"""Registry of the kinds of single runs that `cli/run_single.py` offers as subcommands.

A kind of run is a `Run` subclass that sets its `name`, the name of its subcommand. Registering it
makes it available to the command line and to `config:from-file`, which finds the kind of a
`config.json` by the `__type__` tag, i.e. the class name of its configuration. A study names its run
type directly and needs no registration.

A registry is filled explicitly, from two kinds of sources:

- a package: every module of the package is imported and the named `Run` subclasses defined in it
  are registered (`cli/run_single.py` does this for `single_runs`);
- a file: the file is executed like a script and the named `Run` subclasses defined in it are
  registered (the `--run-file` option of `cli/run_single.py`).

A run in a file needs no registration call:

    # my_run.py
    class MyRun(Run[MyRunConfig]):
        config_type = MyRunConfig
        name = "my-run"
        ...

    pixi run single --run-file path/to/my_run.py --run-dir ... config:my-run ...

The file is loaded like a study module, outside of `sys.modules`. On SLURM the job then receives the
`Run` by value (`submitit` pickles with `cloudpickle`), not by importing the file; this has not been
tested for runs defined in files.

Classes:
    RunKind: A registered kind of run.
    RunRegistry: The registered kinds.
"""

import importlib
import importlib.util
import inspect
import pkgutil
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from bayes_cep.run.config import RunConfig
from bayes_cep.run.template import Run

RUN_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*")


# ==================================================================================================
@dataclass(frozen=True)
class RunKind:
    """A registered kind of run.

    Attributes:
        name (str): Name of the subcommand, lowercase words joined by `-`.
        run_type (type[Run]): The `Run` subclass; its `config_type` is the configuration class.
    """

    name: str
    run_type: type[Run]

    @property
    def config_type(self) -> type[RunConfig]:
        """The configuration class of the run type."""
        return self.run_type.config_type


# ==================================================================================================
class RunRegistry:
    """The registered kinds of run, by subcommand name and by configuration class.

    Methods:
        register: Add one kind of run.
        register_runs_of_module: Add the named runs defined in a module.
        register_runs_in_package: Add the named runs of all modules of a package.
        register_runs_in_file: Add the named runs defined in a Python file.
        run_type_of: The run type that executes a configuration.
    """

    def __init__(self, reserved_names: Iterable[str] = ()) -> None:
        """Create an empty registry.

        Args:
            reserved_names (Iterable[str]): Names that no run may take, e.g. the other subcommands
                of the command line. Defaults to none.
        """
        self._reserved_names = frozenset(reserved_names)
        self._kinds: dict[str, RunKind] = {}

    # ----------------------------------------------------------------------------------------------
    @property
    def kinds(self) -> list[RunKind]:
        """The registered kinds, in order of registration."""
        return list(self._kinds.values())

    # ----------------------------------------------------------------------------------------------
    @property
    def config_types_by_name(self) -> Mapping[str, type[RunConfig]]:
        """The configuration classes by class name, the `__type__` tag of a configuration."""
        return {kind.config_type.__name__: kind.config_type for kind in self._kinds.values()}

    # ----------------------------------------------------------------------------------------------
    def register(self, run_type: type[Run]) -> RunKind:
        """Add a kind of run, named by `run_type.name`; registering the same one again does nothing.

        Args:
            run_type (type[Run]): The `Run` subclass; it must set `name`.

        Returns:
            RunKind: The registered kind.

        Raises:
            ValueError: If `run_type` has no name, the name is malformed, reserved or taken by
                another run type, or the class name of the configuration is that of a different
                registered configuration class (the class name is part of a configuration's JSON
                form).
        """
        name = run_type.name
        if name is None:
            raise ValueError(f"{run_type.__qualname__} has no name to register under.")
        if not RUN_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"A run name must match {RUN_NAME_PATTERN.pattern}, got {name!r}.")
        if name in self._reserved_names:
            raise ValueError(f"The name {name!r} of {run_type.__qualname__} is reserved.")
        kind = RunKind(name, run_type)
        existing = self._kinds.get(name)
        if existing is not None:
            if existing == kind:
                return existing
            raise ValueError(
                f"The name {name!r} of {run_type.__qualname__} is already taken by "
                f"{existing.run_type.__qualname__}."
            )
        other_config = self.config_types_by_name.get(kind.config_type.__name__)
        if other_config is not None and other_config is not kind.config_type:
            raise ValueError(
                f"A different configuration class named {kind.config_type.__name__!r} is already "
                "registered; configuration class names must be unique."
            )
        self._kinds[name] = kind
        return kind

    # ----------------------------------------------------------------------------------------------
    def register_runs_of_module(self, module: ModuleType) -> list[RunKind]:
        """Add the runs that set a `name` and are defined in `module` (not imported into it).

        Returns:
            list[RunKind]: The registered kinds; empty if the module defines no named run.

        Raises:
            ValueError: As `register`.
        """
        run_types = [
            member
            for _, member in inspect.getmembers(module, inspect.isclass)
            if issubclass(member, Run)
            and member.__module__ == module.__name__
            and "name" in vars(member)
            and member.name is not None
        ]
        return [self.register(run_type) for run_type in run_types]

    # ----------------------------------------------------------------------------------------------
    def register_runs_in_package(self, package: ModuleType) -> list[RunKind]:
        """Import every module of `package` and add the named runs defined in them.

        Returns:
            list[RunKind]: The registered kinds.

        Raises:
            ValueError: As `register`.
        """
        kinds: list[RunKind] = []
        for module_info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
            kinds.extend(self.register_runs_of_module(module))
        return kinds

    # ----------------------------------------------------------------------------------------------
    def register_runs_in_file(self, path: Path) -> list[RunKind]:
        """Execute the Python file `path` as a module and add the named runs defined in it.

        Args:
            path (Path): The file; it may import `bayes_cep` and `single_runs`.

        Returns:
            list[RunKind]: The registered kinds.

        Raises:
            ValueError: If the file cannot be loaded or defines no named run, or as `register`.
        """
        spec = importlib.util.spec_from_file_location(f"run_file_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load runs from {path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kinds = self.register_runs_of_module(module)
        if not kinds:
            raise ValueError(f"{path} defines no run with a name.")
        return kinds

    # ----------------------------------------------------------------------------------------------
    def run_type_of(self, config: RunConfig) -> type[Run]:
        """The run type that executes `config`.

        Raises:
            ValueError: If no registered run type has the class of `config` as its configuration.
        """
        for kind in self._kinds.values():
            if type(config) is kind.config_type:
                return kind.run_type
        raise ValueError(f"No registered run type takes a {type(config).__name__}.")

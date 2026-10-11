"""Registry of the run types that `cli/run_single.py` offers as subcommands.

A run type is a `Run` subclass; it is registered under its `name`, the name of its subcommand.
Registering it makes it available to the command line and to `config:from-file`, which finds the
run type of a `config.json` by the `__type__` tag, i.e. the class name of its configuration. A study
names its run type directly and needs no registration.

A registry is filled explicitly, from two kinds of sources:

- a package: every module of the package is imported and the named `Run` subclasses defined in it
  are registered (`cli/run_single.py` does this for `single_runs`);
- a file: the file is executed like a script and the named `Run` subclasses defined in it are
  registered (the `--run-file` option of `cli/run_single.py`).

A run type in a file needs no registration call:

    # my_run.py
    class MyRun(Run[MyRunConfig]):
        config_type = MyRunConfig
        name = "my-run"
        ...

    pixi run single --run-file path/to/my_run.py --run-dir ... config:my-run ...

The file is loaded like a study module, outside of `sys.modules`. On SLURM the job then receives the
`Run` by value (`submitit` pickles with `cloudpickle`), not by importing the file; this has not been
tested for run types defined in files.

Classes:
    RunRegistry: The registered run types.
"""

import importlib
import importlib.util
import inspect
import pkgutil
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType

from bayes_cep.run.config import RunConfig
from bayes_cep.run.template import Run

RUN_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*")


# ==================================================================================================
class RunRegistry:
    """The registered run types, by subcommand name and by configuration class.

    Attributes:
        run_types (list[type[Run]]): The registered run types, in order of registration.
        config_types_by_name (Mapping[str, type[RunConfig]]): The configuration classes of the run
            types by class name, the `__type__` tag of a configuration.

    Methods:
        register: Add one run type.
        register_run_types_of_module: Add the named run types defined in a module.
        register_run_types_in_package: Add the named run types of all modules of a package.
        register_run_types_in_file: Add the named run types defined in a Python file.
        run_type_of: The run type that executes a configuration.
    """

    def __init__(self, reserved_names: Iterable[str] = ()) -> None:
        """Create an empty registry.

        Args:
            reserved_names (Iterable[str]): Names that no run type may take, e.g. the other
                subcommands of the command line. Defaults to none.
        """
        self._reserved_names = frozenset(reserved_names)
        self._run_types_by_name: dict[str, type[Run]] = {}

    # ----------------------------------------------------------------------------------------------
    @property
    def run_types(self) -> list[type[Run]]:
        """The registered run types, in order of registration."""
        return list(self._run_types_by_name.values())

    # ----------------------------------------------------------------------------------------------
    @property
    def config_types_by_name(self) -> Mapping[str, type[RunConfig]]:
        """The configuration classes by class name, the `__type__` tag of a configuration."""
        return {run_type.config_type.__name__: run_type.config_type for run_type in self.run_types}

    # ----------------------------------------------------------------------------------------------
    def register(self, run_type: type[Run]) -> type[Run]:
        """Add a run type, named by `run_type.name`; registering the same one again does nothing.

        Args:
            run_type (type[Run]): The `Run` subclass; it must set `name`.

        Returns:
            type[Run]: `run_type`.

        Raises:
            ValueError: If `run_type` has no name, the name is malformed, reserved or taken by
                another run type, or the class name of its configuration is that of a different
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
        existing = self._run_types_by_name.get(name)
        if existing is not None:
            if existing is run_type:
                return run_type
            raise ValueError(
                f"The name {name!r} of {run_type.__qualname__} is already taken by "
                f"{existing.__qualname__}."
            )
        other_config = self.config_types_by_name.get(run_type.config_type.__name__)
        if other_config is not None and other_config is not run_type.config_type:
            raise ValueError(
                f"A different configuration class named {run_type.config_type.__name__!r} is "
                "already registered; configuration class names must be unique."
            )
        self._run_types_by_name[name] = run_type
        return run_type

    # ----------------------------------------------------------------------------------------------
    def register_run_types_of_module(self, module: ModuleType) -> list[type[Run]]:
        """Add the run types that set a `name` and are defined in `module` (not imported into it).

        Returns:
            list[type[Run]]: The registered run types; empty if the module defines no named one.

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
    def register_run_types_in_package(self, package: ModuleType) -> list[type[Run]]:
        """Import every module of `package` and add the named run types defined in them.

        Returns:
            list[type[Run]]: The registered run types.

        Raises:
            ValueError: As `register`.
        """
        run_types: list[type[Run]] = []
        for module_info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
            run_types.extend(self.register_run_types_of_module(module))
        return run_types

    # ----------------------------------------------------------------------------------------------
    def register_run_types_in_file(self, path: Path) -> list[type[Run]]:
        """Execute the Python file `path` as a module and add the named run types defined in it.

        Args:
            path (Path): The file; it may import `bayes_cep` and `single_runs`.

        Returns:
            list[type[Run]]: The registered run types.

        Raises:
            ValueError: If the file cannot be loaded or defines no named run type, or as `register`.
        """
        spec = importlib.util.spec_from_file_location(f"run_file_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load run types from {path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run_types = self.register_run_types_of_module(module)
        if not run_types:
            raise ValueError(f"{path} defines no run type with a name.")
        return run_types

    # ----------------------------------------------------------------------------------------------
    def run_type_of(self, config: RunConfig) -> type[Run]:
        """The run type that executes `config`.

        Raises:
            ValueError: If no registered run type has the class of `config` as its configuration.
        """
        for run_type in self.run_types:
            if type(config) is run_type.config_type:
                return run_type
        raise ValueError(f"No registered run type takes a {type(config).__name__}.")

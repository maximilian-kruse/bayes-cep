"""Registry of the kinds of single runs that `cli/run_single.py` offers as subcommands.

A kind of run is a `Run` subclass together with the name of its subcommand. Registering one makes
it available to the command line, to `config:from-file` (which finds the kind of a `config.json`
by the `__type__` tag, i.e. the class name of its configuration) and to nothing else: a study names
its run type directly and needs no registration.

The kinds of this repository are registered by `cli/run_single.py`. Another kind, defined anywhere,
registers itself in an importable module:

    # my_runs.py, in a directory on the PYTHONPATH
    from bayes_cep.run.registry import register_run_type
    register_run_type("my-run", MyRun)

and is activated by naming the module in the environment variable `BAYES_CEP_RUN_MODULES` (several
modules separated as in `PATH`):

    BAYES_CEP_RUN_MODULES=my_runs PYTHONPATH=/path/to/dir pixi run single ... config:my-run ...

The module must be importable by name, not loaded from a file, because a SLURM job unpickles the
`Run` and thereby imports its module by name; the directory must be on the `PYTHONPATH` of the job,
as it must be for a study that uses the run type.

Classes:
    RunKind: A registered kind of run.
    RunRegistry: The registered kinds.

Functions:
    register_run_type: Register a kind of run in the default registry.
    import_registration_modules: Import the modules named in `BAYES_CEP_RUN_MODULES`.
"""

import importlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from bayes_cep.run.config import RunConfig
from bayes_cep.run.template import Run

RUN_MODULES_ENVIRONMENT_VARIABLE = "BAYES_CEP_RUN_MODULES"
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
        register: Add a kind of run.
        run_type_of: The run type that executes a configuration.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
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
    def register(self, name: str, run_type: type[Run]) -> RunKind:
        """Add a kind of run; registering the same name and run type again changes nothing.

        Args:
            name (str): Name of the subcommand; lowercase words joined by `-`.
            run_type (type[Run]): The `Run` subclass.

        Returns:
            RunKind: The registered kind.

        Raises:
            ValueError: If the name is malformed or taken by another run type, or the class name
                of the configuration is that of a different registered configuration class (the
                class name is part of a configuration's JSON form).
        """
        if not RUN_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"A run name must match {RUN_NAME_PATTERN.pattern}, got {name!r}.")
        kind = RunKind(name, run_type)
        existing = self._kinds.get(name)
        if existing is not None:
            if existing == kind:
                return existing
            raise ValueError(
                f"The name {name!r} is already registered for {existing.run_type.__qualname__}."
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
    def run_type_of(self, config: RunConfig) -> type[Run]:
        """The run type that executes `config`.

        Raises:
            ValueError: If no registered run type has the class of `config` as its configuration.
        """
        for kind in self._kinds.values():
            if type(config) is kind.config_type:
                return kind.run_type
        raise ValueError(f"No registered run type takes a {type(config).__name__}.")


# ==================================================================================================
RUN_REGISTRY = RunRegistry()


# --------------------------------------------------------------------------------------------------
def register_run_type(name: str, run_type: type[Run]) -> RunKind:
    """Register a kind of run in the default registry; see `RunRegistry.register`."""
    return RUN_REGISTRY.register(name, run_type)


# --------------------------------------------------------------------------------------------------
def import_registration_modules() -> None:
    """Import the modules named in `BAYES_CEP_RUN_MODULES`, so that they register their runs.

    Raises:
        ModuleNotFoundError: If a named module is not importable, e.g. its directory is missing
            from the `PYTHONPATH`.
    """
    for module_name in os.environ.get(RUN_MODULES_ENVIRONMENT_VARIABLE, "").split(os.pathsep):
        if module_name:
            importlib.import_module(module_name)

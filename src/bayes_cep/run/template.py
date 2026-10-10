"""Template of a run: the generic wrapper around the computation of one configuration.

A run leaves a record of itself next to its data: the configuration, metadata, status, log and
metrics, written through its `RunDirectory`.

Classes:
    Run: Abstract run, paired with its configuration type.

Type aliases:
    Metrics: The scalar metrics of a run.
"""

import sys
import traceback
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import ClassVar, get_args, get_origin

from ls_bayesian.common.logging import BaseLogger, LoggerSettings

from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import RunDirectory, RunState
from bayes_cep.run.environment import Environment
from bayes_cep.run.metadata import RunMetadata

type Metrics = dict[str, float | bool | int | str]


# ==================================================================================================
class Run[ConfigT: RunConfig](ABC):
    """A run: a pure function of one configuration, writing everything into its run directory.

    `execute` is the template method: it records the configuration, metadata, status, log and
    metrics, and calls `_execute` for the actual computation. A subclass names its configuration
    type as the generic argument (`class MyRun(Run[MyConfig])`), sets `outputs`, implements
    `_execute` and `report`, and may override `input_files`.

    Attributes:
        config_type (type[RunConfig]): The configuration class this run is paired with; taken from
            the generic argument of the subclass.
        outputs (dict[str, str]): Description of the files a run writes, by path relative to the
            run directory; documented in the study description.
        config (ConfigT): The configuration of this run.
        console (bool): Whether progress is also printed to the console.
    """

    config_type: ClassVar[type[RunConfig]]
    outputs: ClassVar[dict[str, str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Take `config_type` from the generic argument, e.g. `MyRun(Run[MyConfig])`."""
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Run:
                (config_type,) = get_args(base)
                if isinstance(config_type, type):
                    cls.config_type = config_type

    def __init__(self, config: ConfigT, console: bool = True) -> None:
        """Pair the run with its configuration.

        Args:
            config (ConfigT): Run configuration; must be an instance of `config_type`.
            console (bool): Whether progress is also printed to the console. Defaults to `True`.

        Raises:
            TypeError: If the subclass does not define `config_type` (through its generic
                argument) and `outputs`, or if `config` is not an instance of `config_type`.
        """
        for name in ("config_type", "outputs"):
            if not hasattr(type(self), name):
                raise TypeError(f"{type(self).__name__} must define {name}.")
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"{type(self).__name__} needs a {self.config_type.__name__}, "
                f"got {type(config).__name__}."
            )
        self.config = config
        self.console = console

    # ----------------------------------------------------------------------------------------------
    def input_files(self) -> list[Path]:
        """Files the run reads, whose content hashes are recorded in the metadata."""
        return []

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def report(self, run_dir: Path) -> None:
        """Write plots of a finished run to `<run_dir>/plots`.

        Plotting may need packages that are not available on a cluster, so this is a separate
        step from `execute`.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        """Perform the computation, writing result files into the run directory.

        Args:
            run_dir (Path): Run directory; create the result folders in it.
            logger (BaseLogger): Logger for progress output.

        Returns:
            Metrics: Scalar metrics of the run, stored as `metrics.json`.
        """

    # ----------------------------------------------------------------------------------------------
    @contextmanager
    def open_run_logger(self, log_path: Path) -> Generator[BaseLogger]:
        """Provide the logger of the run, after logging which run it is and its configuration.

        The log file is overwritten and its missing parent directories are created.

        Args:
            log_path (Path): Log file.

        Yields:
            BaseLogger: The logger, closed on exit. An exception in the body is logged (with its
                traceback in the log file only) and propagates.
        """
        settings = LoggerSettings(print_to_console=self.console, logfile_path=log_path)
        with BaseLogger(settings) as logger:
            started = datetime.now().astimezone().isoformat(timespec="seconds")
            logger.info(f"Run started {started}")
            logger.info(f"Command: {' '.join(sys.argv)}")
            logger.info(f"Log file: {log_path}")
            logger.info("")
            logger.info(f"{type(self).__name__} {self.config.run_id}")
            logger.info(self.config.describe())
            logger.info("")
            try:
                yield logger
            except BaseException as error:
                logger.error(f"Run failed: {type(error).__name__}: {error}")
                logger.debug(traceback.format_exc())
                raise

    # ----------------------------------------------------------------------------------------------
    def execute(self, run_dir: Path, environment: Environment | None = None) -> RunState:
        """Run in `run_dir` and record everything about the run there.

        Records `config.json` and `metadata.json` first, then runs with the log in `run.log`, and
        finally records `metrics.json` and the end state. An exception, including one while
        recording, is recorded as the `failed` state with its traceback, and not raised, so that a
        study can continue. An interruption (`KeyboardInterrupt`, `SystemExit`) is recorded the
        same way and then raised again. A run that is killed without a chance to record (out of
        memory, `SIGKILL`) stays `running`.

        Args:
            run_dir (Path): Run directory. Files of an earlier attempt are the caller's
                responsibility, except the status and metrics, which are replaced.
            environment (Environment | None): The environment to record. Collected here if `None`;
                a submitter that executes many runs collects it once and passes it on.

        Returns:
            RunState: The final state, `DONE` or `FAILED`.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        directory = RunDirectory(run_dir)
        try:
            if environment is None:
                environment = Environment.collect_from_current_process()
            metadata = RunMetadata.collect_for_run(self.input_files(), environment)
            directory.record_start(self.config, asdict(metadata))
            with self.open_run_logger(directory.log_path) as logger:
                metrics = self._execute(run_dir, logger)
            directory.record_success(metrics)
        except BaseException as error:
            directory.record_failure(error, traceback.format_exc())
            if not isinstance(error, Exception):
                raise
            return RunState.FAILED
        return RunState.DONE

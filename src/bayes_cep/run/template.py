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
from pathlib import Path
from typing import ClassVar

import numpy as np
from ls_bayesian.common.logging import BaseLogger, LoggerSettings

from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import (
    PLOTS_DIR_NAME,
    RESULTS_DIR_NAME,
    RUN_RECORD_NAMES,
    RunDirectory,
    RunState,
    format_current_time,
)
from bayes_cep.run.provenance import Environment, RunMetadata

type Metrics = dict[str, float | bool | int | str]


# ==================================================================================================
class Run[ConfigT: RunConfig](ABC):
    """A run: a pure function of one configuration, writing everything into its run directory.

    `execute` is the template method: it records the configuration, metadata, status, log and
    metrics, and calls `_execute` for the actual computation. A subclass names its configuration
    type as the generic argument (`class MyRun(Run[MyConfig])`), sets `config_type` to the same
    class and `outputs`, and implements `_execute`, `report` and `input_files`.

    Attributes:
        config_type (type[RunConfig]): The configuration class this run is paired with; set by the
            subclass, next to the generic argument that types `config`.
        outputs (dict[str, str]): Description of the files a run writes, by path relative to the
            run directory; documented in the study description. Result files are listed with
            their `results/` prefix, which is dropped with `flat_results`. A key ending in `/` (or
            naming a directory) stands for everything below it. `execute` checks the files that
            were written against it (see `_check_outputs`); the plots are written by `report`, so
            the `plots/` entry is exempt.
        optional_outputs (frozenset[str]): The keys of `outputs` that a run may legitimately not
            write, e.g. because of a setting; empty by default.
        name (str | None): Name of the subcommand of `bcep-single` that runs this
            run type (lowercase words joined by `-`; see `bayes_cep.run.registry`). `None`, the
            default, keeps the run type off the command line; a study needs no name.
        config (ConfigT): The configuration of this run.
        write_to_console (bool): Whether progress is also printed to the console.
        flat_results (bool): Whether result files are written directly into the run directory
            instead of its `results/` subdirectory. Not part of the run's identity; for runs
            whose directory is the result itself, like the example data.
    """

    config_type: ClassVar[type[RunConfig]]
    outputs: ClassVar[dict[str, str]]
    optional_outputs: ClassVar[frozenset[str]] = frozenset()
    name: ClassVar[str | None] = None

    def __init__(
        self, config: ConfigT, write_to_console: bool = True, flat_results: bool = False
    ) -> None:
        """Pair the run with its configuration.

        Args:
            config (ConfigT): Run configuration; must be an instance of `config_type`.
            write_to_console (bool): Whether progress is also printed to the console. Defaults to
                `True`.
            flat_results (bool): Whether result files go directly into the run directory instead
                of `results/`. Defaults to `False`.

        Raises:
            TypeError: If the subclass does not define `config_type` and `outputs`, or if
                `config` is not an instance of `config_type`.
        """
        for name in ("config_type", "outputs"):
            if not hasattr(type(self), name):
                raise TypeError(f"{type(self).__name__} must define {name}.")
        unknown_optional = self.optional_outputs - self.outputs.keys()
        if unknown_optional:
            raise ValueError(
                f"{type(self).__name__}.optional_outputs names files that are not in outputs: "
                f"{sorted(unknown_optional)}."
            )
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"{type(self).__name__} needs a {self.config_type.__name__}, "
                f"got {type(config).__name__}."
            )
        self.config = config
        self.write_to_console = write_to_console
        self.flat_results = flat_results

    # ----------------------------------------------------------------------------------------------
    def results_dir(self, run_dir: Path) -> Path:
        """The directory the result files of the run are in (created by the run when writing)."""
        return run_dir if self.flat_results else run_dir / RESULTS_DIR_NAME

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def input_files(self) -> list[Path]:
        """Files the run reads, whose content hashes are recorded in the metadata.

        Every run states its inputs explicitly; a run that reads no files returns an empty list.
        """

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
    def run_logger(self, log_path: Path) -> Generator[BaseLogger]:
        """Provide the logger of the run, after logging which run it is and its configuration.

        The log file is overwritten and its missing parent directories are created.

        Args:
            log_path (Path): Log file.

        Yields:
            BaseLogger: The logger, closed on exit. An exception in the body is logged (with its
                traceback in the log file only) and propagates.
        """
        settings = LoggerSettings(print_to_console=self.write_to_console, logfile_path=log_path)
        with BaseLogger(settings) as logger:
            logger.info(f"Run started {format_current_time()}")
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
    def _check_outputs(self, run_dir: Path, logger: BaseLogger) -> None:
        """Compare the files the run wrote with `outputs`.

        Exempt are the run records (`RUN_RECORD_NAMES`, e.g. `metrics.json`, written by `execute`
        after `_execute`) and the `plots/` entry (written later by `report`). With `flat_results`
        the `results/` prefix of a key is dropped.

        Args:
            run_dir (Path): Run directory, after `_execute`.
            logger (BaseLogger): Receives a warning for every written file that `outputs` does not
                cover, so that `outputs` stays the complete description of a run.

        Raises:
            FileNotFoundError: If a declared output, not in `optional_outputs`, was not written.
        """

        def output_path(key: str) -> str:
            path = key.rstrip("/")
            return path.removeprefix(f"{RESULTS_DIR_NAME}/") if self.flat_results else path

        declared = {output_path(key) for key in self.outputs}
        declared -= {PLOTS_DIR_NAME} | RUN_RECORD_NAMES
        optional = {output_path(key) for key in self.optional_outputs}
        missing = sorted(path for path in declared - optional if not (run_dir / path).exists())
        if missing:
            raise FileNotFoundError(f"Declared outputs were not written: {missing}")
        written = [path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file()]
        exempt = {Path(name) for name in RUN_RECORD_NAMES}
        undeclared = sorted(
            path.as_posix()
            for path in written
            if path not in exempt
            and path.parts[0] != PLOTS_DIR_NAME
            and not any(path.is_relative_to(declared_path) for declared_path in declared)
        )
        if undeclared:
            logger.warning(f"Files written but not listed in outputs: {undeclared}")

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _as_plain_metrics(metrics: Metrics) -> Metrics:
        """The metrics with numpy scalars converted to Python scalars.

        Raises:
            TypeError: If a metric is not a scalar number, boolean or string, so that a bad metric
                fails the run with a clear message instead of a JSON error while recording.
        """
        plain: Metrics = {}
        for name, value in metrics.items():
            item = value.item() if isinstance(value, np.generic) else value
            if not isinstance(item, float | bool | int | str):
                raise TypeError(f"Metric {name!r} must be a scalar, got {type(value).__name__}.")
            plain[name] = item
        return plain

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
            run_dir (Path): Run directory. Removing the files of an earlier attempt is the
                executor's job (`Executor`), except for the status and metrics, which are replaced.
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
            with self.run_logger(directory.log_path) as logger:
                metrics = self._execute(run_dir, logger)
                self._check_outputs(run_dir, logger)
            directory.record_success(self._as_plain_metrics(metrics))
        except BaseException as error:
            directory.record_failure(error, traceback.format_exc())
            if not isinstance(error, Exception):
                raise
            return RunState.FAILED
        return RunState.DONE

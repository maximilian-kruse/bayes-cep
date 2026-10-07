"""Template of a run: the generic wrapper around the computation of one configuration.

Classes:
    Run: Abstract run, paired with its configuration type.
"""

import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from ls_bayesian.common.logging import BaseLogger

from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import DONE, FAILED, RUNNING, RunDirectory
from bayes_cep.run.logging import run_logger
from bayes_cep.run.metadata import collect_run_metadata

type Metrics = dict[str, float | bool | int | str]


# ==================================================================================================
class Run[ConfigT: RunConfig](ABC):
    """A run: a pure function of one configuration, writing everything into its run directory.

    `execute` is the template method: it records the configuration, metadata, status,
    log and metrics, and calls `_execute` for the actual computation. Subclasses set `config_type`
    and `outputs`, implement `_execute` and `report`, and may override `input_files`.

    Attributes:
        config_type (type[RunConfig]): The configuration class this run is paired with.
        stages (tuple[str, ...]): The parts a run can be restricted to, e.g. to regenerate the
            example data stage by stage; `"all"` is the whole run and the default. Later stages
            read the data of earlier ones from the example layout.
        outputs (dict[str, str]): Description of the files a run writes, by path relative to the
            run directory; documented in the study description.
        config (ConfigT): The configuration of this run.
        console (bool): Whether progress is also printed to the console.
        stage (str): Which part of the run to perform; one of `stages`.
        example_layout (bool): Whether to write the plain example-data layout (data in
            subfolders, logs in `logs/`) instead of the recorded run directory: no `config.json`,
            `metadata.json`, `status.json` or `metrics.json`.
    """

    config_type: ClassVar[type[RunConfig]]
    outputs: ClassVar[dict[str, str]]
    stages: ClassVar[tuple[str, ...]] = ("all",)

    def __init__(
        self,
        config: ConfigT,
        console: bool = True,
        example_layout: bool = False,
        stage: str = "all",
    ) -> None:
        """Pair the run with its configuration.

        Args:
            config (ConfigT): Run configuration; must be an instance of `config_type`.
            console (bool): Whether progress is also printed to the console. Defaults to `True`.
            example_layout (bool): Whether to write the plain example-data layout. Defaults to
                `False`.
            stage (str): The part of the run to perform. Defaults to `"all"`.

        Raises:
            TypeError: If `config` is not an instance of `config_type`.
            ValueError: If `stage` is not one of `stages`, or is not `"all"` without the example
                layout, since a recorded run directory is always complete.
        """
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"{type(self).__name__} needs a {self.config_type.__name__}, "
                f"got {type(config).__name__}."
            )
        if stage not in self.stages:
            raise ValueError(f"stage must be one of {self.stages}, got {stage!r}.")
        if stage != "all" and not example_layout:
            raise ValueError("A stage other than 'all' needs the example layout.")
        self.config = config
        self.stage = stage
        self.console = console
        self.example_layout = example_layout

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
    @property
    def _log_name(self) -> str:
        return "run" if self.stage == "all" else self.stage

    # ----------------------------------------------------------------------------------------------
    def execute(self, run_dir: Path) -> str:
        """Run in `run_dir` and record everything about the run there.

        Writes `config.json` and `metadata.json` first, then runs with the log in `run.log`,
        and finally writes `metrics.json` and the end state. An exception is recorded as the
        `failed` state, with the traceback, and not raised, so that a study can continue. With
        `example_layout`, nothing is recorded but the log (`logs/<stage>.log`) and the data.

        Args:
            run_dir (Path): Run directory; its previous content is the caller's responsibility.

        Returns:
            str: The final state, `DONE` or `FAILED`.
        """
        record = not self.example_layout
        directory = RunDirectory(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if record:
            directory.write("config.json", self.config.to_dict())
            directory.write("metadata.json", collect_run_metadata(self.input_files()))
            directory.set_status(RUNNING)
        logfile_path = run_dir / ("run.log" if record else f"logs/{self._log_name}.log")
        try:
            with run_logger(logfile_path, print_to_console=self.console) as logger:
                logger.info(f"{type(self).__name__} {self.config.run_id}")
                logger.info(self.config.describe())
                logger.info("")
                metrics = self._execute(run_dir, logger)
            if record:
                directory.write("metrics.json", metrics)
        except Exception as error:
            if record:
                directory.set_status(
                    FAILED,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            return FAILED
        if record:
            directory.set_status(DONE)
        return DONE

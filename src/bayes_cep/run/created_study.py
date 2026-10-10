"""A study written to disk: creating it, executing its runs, and plotting the finished ones.

`CreatedStudy.create_from_module` resolves a study module and writes the study directory (see
`directories` for its layout), including a copy of the module and the environment specification.
The run list is fixed at that moment: working on the created study reads the configurations back
from `runs.json`, so later changes of the code never change which runs the study consists of.

Classes:
    CreatedStudy: A study directory together with the study definition archived in it.
"""

import shutil
import warnings
from dataclasses import asdict
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Self

from bayes_cep.run.config import ConfigCodec
from bayes_cep.run.directories import RunState, StudyDirectory, read_json_record, write_json_record
from bayes_cep.run.environment import Environment
from bayes_cep.run.environment_archive import EnvironmentArchive
from bayes_cep.run.executor import Executor, ExecutorSettings, RunOutcome
from bayes_cep.run.study import ResolvedRun, Study


# ==================================================================================================
class CreatedStudy:
    """A study directory together with the study definition archived in it.

    Attributes:
        directory (StudyDirectory): The directory of the study.
        definition (Study): The study as defined by the archived module.
    """

    def __init__(self, directory: StudyDirectory, definition: Study) -> None:
        """Bind a study definition to the directory created for it."""
        self.directory = directory
        self.definition = definition

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def create_from_module(cls, module_path: Path, root: Path) -> Self:
        """Resolve a study module and write its study directory, before any run starts.

        The directory is built under a temporary name and moved into place at the end, so a failure
        or an interruption never leaves a half-written study behind.

        Args:
            module_path (Path): Python file defining `STUDY`; archived with the study.
            root (Path): Directory holding all study directories; created if missing.

        Returns:
            Self: The new study.

        Raises:
            FileExistsError: If the study directory already exists.
            ValueError: If a configuration does not survive its JSON form.
        """
        study = Study.load_from_module_file(module_path)
        study_dir = (root / study.name).resolve()
        if study_dir.exists():
            raise FileExistsError(f"Study directory {study_dir} already exists.")
        runs = study.resolve_runs()
        for run in runs:
            if study.run_type.config_type.from_json_dict(run.config.to_json_dict()) != run.config:
                raise ValueError(f"The configuration of run {run.index} does not survive JSON.")

        temporary_dir = study_dir.with_name(f".{study.name}.creating")
        shutil.rmtree(temporary_dir, ignore_errors=True)
        try:
            cls._write_description(StudyDirectory(temporary_dir), study, runs, module_path)
            temporary_dir.rename(study_dir)
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
        return cls(StudyDirectory(study_dir), study)

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, study_dir: Path) -> Self:
        """Load a created study, with its definition from the module archived in `study_dir`."""
        directory = StudyDirectory(study_dir)
        return cls(directory, Study.load_from_module_file(directory.definition_path))

    # ----------------------------------------------------------------------------------------------
    @cached_property
    def runs(self) -> list[ResolvedRun]:
        """The runs recorded at creation, with their configurations read back from `runs.json`."""
        config_type = self.definition.run_type.config_type
        return [
            ResolvedRun(
                index=entry["index"],
                run_id=entry["id"],
                config=config_type.from_json_dict(entry["config"]),
                overrides=entry["overrides"],
            )
            for entry in self.directory.read_run_index()
        ]

    # ----------------------------------------------------------------------------------------------
    def read_states(self) -> dict[int, RunState]:
        """The state of every run, by run index."""
        return {
            run.index: self.directory.run_directory(run.run_id).read_state() for run in self.runs
        }

    # ----------------------------------------------------------------------------------------------
    def execute_runs(
        self,
        settings: ExecutorSettings,
        indices: list[int] | None = None,
        force: bool = False,
        include_active: bool = False,
        wait: bool = True,
    ) -> dict[int, RunOutcome]:
        """Execute the unfinished runs.

        Finished runs are skipped unless forced, so executing again after failures only repeats
        the others. Runs that are submitted or running are skipped as well, because a second job
        would delete the files of the first; if their jobs have died (killed, node failure,
        interrupted submission), `include_active` restarts them.

        If the code or the environment differs from the one recorded at creation (commit,
        uncommitted changes, `pixi.lock`, editable packages), or the study module would now
        resolve to different runs, a warning lists the differences before any run starts. The
        runs themselves are always the recorded ones, and every run records the environment it
        actually executed in.

        Args:
            settings (ExecutorSettings): Where and with which resources the runs execute.
            indices (list[int] | None): Runs to consider (repeated indices count once); all runs if
                `None`.
            force (bool): Whether to also rerun finished runs. Defaults to `False`.
            include_active (bool): Whether to also restart submitted or running runs. Defaults to
                `False`.
            wait (bool): Whether to wait for the runs to finish; see `Executor.run`. Defaults to
                `True`.

        Returns:
            dict[int, RunOutcome]: Outcome by run index.

        Raises:
            ValueError: If an index is out of range.
        """
        selected = list(range(len(self.runs))) if indices is None else list(dict.fromkeys(indices))
        for index in selected:
            if not 0 <= index < len(self.runs):
                raise ValueError(f"Run index {index} is out of range for {len(self.runs)} runs.")
        outcomes: dict[int, RunOutcome] = {}
        pending: list[int] = []
        for index in selected:
            state = self.directory.run_directory(self.runs[index].run_id).read_state()
            if state == RunState.DONE and not force:
                outcomes[index] = RunOutcome.SKIPPED
            elif state.is_active and not include_active:
                outcomes[index] = RunOutcome.ACTIVE
            else:
                pending.append(index)
        if pending:
            environment = Environment.collect_from_current_process()
            self._warn_if_code_changed(environment)
            executor = Executor(
                settings, self.directory.job_dir, self.directory.path.resolve().name
            )
            results = executor.run(
                [self.definition.run_type(self.runs[index].config) for index in pending],
                [self.directory.run_directory(self.runs[index].run_id).path for index in pending],
                environment,
                wait,
            )
            outcomes.update(zip(pending, results, strict=True))
        return dict(sorted(outcomes.items()))

    # ----------------------------------------------------------------------------------------------
    def plot_finished_runs(self, index: int | None = None) -> list[int]:
        """Plot the finished runs; unfinished ones are skipped.

        A run whose plotting fails is reported and does not stop the others.

        Args:
            index (int | None): Only this run; all runs if `None`.

        Returns:
            list[int]: The indices of the runs that were plotted.

        Raises:
            ValueError: If `index` is out of range.
        """
        if index is not None and not 0 <= index < len(self.runs):
            raise ValueError(f"Run index {index} is out of range for {len(self.runs)} runs.")
        plotted = []
        for run in self.runs:
            if index is not None and run.index != index:
                continue
            run_directory = self.directory.run_directory(run.run_id)
            if run_directory.read_state() != RunState.DONE:
                continue
            print(f"Plotting run {run.index} ({run.run_id})")
            try:
                self.definition.run_type(run.config).report(run_directory.path)
            except Exception as error:
                print(f"Plotting run {run.index} failed: {error!r}")
                continue
            plotted.append(run.index)
        return plotted

    # ----------------------------------------------------------------------------------------------
    def _warn_if_code_changed(self, environment: Environment) -> None:
        """Warn if the environment or the study module differs from the state at creation.

        A study can be continued long after it was created, so its runs may stem from different
        code states; every run records the environment it executed in.
        """
        differences = environment.find_differences(
            read_json_record(self.directory.environment_path)
        )
        recorded_ids = [run.run_id for run in self.runs]
        try:
            current_ids = [run.run_id for run in self.definition.resolve_runs()]
        except (ValueError, TypeError) as error:
            differences.append(f"the study no longer resolves to runs: {error}")
        else:
            if current_ids != recorded_ids:
                differences.append(
                    "the recorded configurations resolve to other run ids with the current code "
                    "(configuration classes or defaults changed)"
                )
        if differences:
            lines = "\n  ".join(differences)
            warnings.warn(
                f"The code differs from the state recorded when the study was created:\n  {lines}",
                stacklevel=2,
            )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _write_description(
        directory: StudyDirectory, study: Study, runs: list[ResolvedRun], module_path: Path
    ) -> None:
        """Write the study description, the run list and the environment into `directory`."""
        directory.description_dir.mkdir(parents=True)
        shutil.copy(module_path, directory.definition_path)
        environment = Environment.collect_from_current_process()
        environment_files = EnvironmentArchive(
            environment, directory.environment_dir
        ).write_specification()
        write_json_record(
            directory.study_record_path,
            {
                "name": study.name,
                "description": study.description,
                "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                "run_type": study.run_type_path,
                "num_runs": len(runs),
                "sweep": study.sweep.to_json_dict(),
                "base_config": study.base.to_json_dict(),
                "outputs": study.run_type.outputs,
                "environment_files": environment_files,
            },
        )
        run_index: list[dict[str, Any]] = [
            {
                "index": run.index,
                "id": run.run_id,
                "overrides": ConfigCodec.encode(run.overrides),
                "config": run.config.to_json_dict(),
            }
            for run in runs
        ]
        write_json_record(directory.run_index_path, run_index)
        write_json_record(directory.environment_path, asdict(environment))

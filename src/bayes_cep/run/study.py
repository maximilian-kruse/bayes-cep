"""A study: a run type, a base configuration and sweeps, resolved into a fixed list of runs.

`Study` is the definition, written in a study module. `create_directory` resolves it and writes the
study directory (see `directories` for its layout), including a copy of the module. Working on a
created study means loading that archived module (`load_from_directory`); its runs must resolve to
the ids recorded at creation. `execute_runs` executes the unfinished runs, `plot_finished_runs`
plots the finished ones.

Classes:
    SweepNode: A variation of configuration parameters, expanding into override dicts.
    Axis: One parameter and its values.
    Zip: Axes varied together.
    Product: All combinations of groups.
    ResolvedRun: One run of a resolved study.
    StudyRecord: The description of a study, stored as `study/study.json`.
    Study: Run type, base configuration, sweep, description and collector.
"""

import hashlib
import importlib
import importlib.util
import itertools
import re
import shutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, override

from bayes_cep.run.collector import Collector, RunTableCollector
from bayes_cep.run.config import ConfigCodec, RunConfig, format_config_tree
from bayes_cep.run.directories import RunState, StudyDirectory
from bayes_cep.run.executor import Executor, ExecutorSettings, RunOutcome
from bayes_cep.run.metadata import Environment, EnvironmentArchive
from bayes_cep.run.template import Run

STUDY_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


# ==================================================================================================
class SweepNode(ABC):
    """A variation of configuration parameters: a tree of axes, zips and products."""

    @abstractmethod
    def expand_overrides(self) -> list[dict[str, Any]]:
        """One override dict (dotted path to value) per combination."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def to_json_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def describe(self, level: int = 0) -> str:
        """Indented text overview, starting at indentation `level`."""


# ==================================================================================================
class Axis(SweepNode):
    """One parameter, addressed by dotted path, and the values it takes.

    Attributes:
        path (str): Dotted path into the base configuration, e.g. `"prior.kappa"`.
        values (tuple[Any, ...]): Values of the parameter.
    """

    def __init__(self, path: str, values: Sequence[Any]) -> None:
        """Vary the parameter at `path` over `values`.

        Raises:
            ValueError: If there are no values.
        """
        if len(values) == 0:
            raise ValueError(f"Axis {path!r} has no values.")
        self.path = path
        self.values = tuple(values)

    @override
    def expand_overrides(self) -> list[dict[str, Any]]:
        return [{self.path: value} for value in self.values]

    @override
    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": "axis", "path": self.path, "values": ConfigCodec.encode(self.values)}

    @override
    def describe(self, level: int = 0) -> str:
        labels = ", ".join(self._label(value) for value in self.values)
        return f"{'  ' * level}{self.path}: {labels}"

    @staticmethod
    def _label(value: Any) -> str:
        """A value for display: scalars as they are, config objects by their class name."""
        return type(value).__name__ if is_dataclass(value) else str(value)


# ==================================================================================================
class Zip(SweepNode):
    """Axes varied together: the i-th value of every axis forms one combination."""

    def __init__(self, *axes: Axis) -> None:
        """Group axes of equal length.

        Raises:
            ValueError: If there are no axes or their lengths differ.
        """
        if not axes:
            raise ValueError("Zip needs at least one axis.")
        lengths = {len(axis.values) for axis in axes}
        if len(lengths) != 1:
            raise ValueError(f"Zip axes must have equal lengths, got {sorted(lengths)}.")
        self.axes = axes

    @override
    def expand_overrides(self) -> list[dict[str, Any]]:
        return [
            {axis.path: axis.values[position] for axis in self.axes}
            for position in range(len(self.axes[0].values))
        ]

    @override
    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": "zip", "axes": [axis.to_json_dict() for axis in self.axes]}

    @override
    def describe(self, level: int = 0) -> str:
        lines = [f"{'  ' * level}zip"] + [axis.describe(level + 1) for axis in self.axes]
        return "\n".join(lines)


# ==================================================================================================
class Product(SweepNode):
    """All combinations of its groups; the last group varies fastest. No groups: a single run."""

    def __init__(self, *groups: Axis | Zip) -> None:
        """Combine axes and zips."""
        self.groups = groups

    @override
    def expand_overrides(self) -> list[dict[str, Any]]:
        merged = []
        for parts in itertools.product(*(group.expand_overrides() for group in self.groups)):
            combination: dict[str, Any] = {}
            for part in parts:
                combination.update(part)
            merged.append(combination)
        return merged

    @override
    def to_json_dict(self) -> dict[str, Any]:
        return {"kind": "product", "groups": [group.to_json_dict() for group in self.groups]}

    @override
    def describe(self, level: int = 0) -> str:
        lines = [f"{'  ' * level}product"] + [group.describe(level + 1) for group in self.groups]
        return "\n".join(lines)


# ==================================================================================================
@dataclass(frozen=True)
class ResolvedRun:
    """One run of a resolved study.

    Attributes:
        index (int): Position in the run list.
        run_id (str): Content hash of `config`.
        config (RunConfig): Complete configuration of the run.
        overrides (dict[str, Any]): The swept parameters of this run, by dotted path.
    """

    index: int
    run_id: str
    config: RunConfig
    overrides: dict[str, Any]


# ==================================================================================================
@dataclass(frozen=True)
class StudyRecord:
    """The description of a study, stored as `study/study.json`.

    Attributes:
        name (str): Name of the study.
        description (str): What the study investigates and computes.
        created (str): Creation time, ISO 8601 with time zone.
        run_type (str): The run type as `module:Class`.
        collector (str): Class name of the study's collector.
        num_runs (int): Number of runs.
        sweep (dict[str, Any]): The sweep in JSON form.
        base_config (dict[str, Any]): The base configuration in JSON form.
        outputs (dict[str, str]): The files a run writes, as described by the run type.
        definition_sha256 (str): Hash of the archived study module.
        environment_files (list[str]): The archived environment specification, see
            `EnvironmentArchive`.
    """

    name: str
    description: str
    created: str
    run_type: str
    collector: str
    num_runs: int
    sweep: dict[str, Any]
    base_config: dict[str, Any]
    outputs: dict[str, str]
    definition_sha256: str
    environment_files: list[str]


# ==================================================================================================
@dataclass(frozen=True)
class Study:
    """A run type, the base configuration of its runs, the sweep over it, and a description.

    Attributes:
        name (str): Name of the study, also the name of its directory.
        description (str): What the study investigates and computes.
        run_type (type[Run]): The kind of run all runs of the study are.
        base (RunConfig): Configuration of all runs before the sweep is applied; fixes all
            parameters that are common to the runs.
        sweep (SweepNode): The parameter variations.
        collector (type[Collector]): Analysis of the finished runs, matching this study. Defaults
            to writing the run table only.
    """

    name: str
    description: str
    run_type: type[Run]
    base: RunConfig
    sweep: SweepNode
    collector: type[Collector] = RunTableCollector

    def __post_init__(self) -> None:
        if not STUDY_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                f"Study name must match {STUDY_NAME_PATTERN.pattern}, got {self.name!r}."
            )
        if not isinstance(self.base, self.run_type.config_type):
            raise ValueError(
                f"{self.run_type.__name__} needs a {self.run_type.config_type.__name__} as base, "
                f"got {type(self.base).__name__}."
            )

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load_from_module_file(cls, module_path: Path) -> Self:
        """Load the `STUDY` attribute of a study module file.

        The module is executed as a script and must be self-contained apart from `bayes_cep`.

        Args:
            module_path (Path): Python file defining `STUDY`.

        Returns:
            Self: The study.

        Raises:
            ValueError: If the file does not define a `Study` named `STUDY`.
        """
        spec = importlib.util.spec_from_file_location(f"study_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load a study from {module_path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        study = getattr(module, "STUDY", None)
        if not isinstance(study, cls):
            raise ValueError(f"{module_path} must define STUDY as a {cls.__name__}.")
        return study

    # ----------------------------------------------------------------------------------------------
    def resolve_runs(self) -> list[ResolvedRun]:
        """Expand the sweep into the ordered run list.

        Returns:
            list[ResolvedRun]: One entry per run.

        Raises:
            ValueError: If two runs resolve to the same configuration, or an override path is
                invalid.
        """
        runs: list[ResolvedRun] = []
        seen: dict[str, int] = {}
        for index, overrides in enumerate(self.sweep.expand_overrides()):
            config = self.base.with_overrides(overrides)
            run_id = config.run_id
            if run_id in seen:
                raise ValueError(f"Runs {seen[run_id]} and {index} are identical.")
            seen[run_id] = index
            runs.append(ResolvedRun(index, run_id, config, overrides))
        return runs

    # ----------------------------------------------------------------------------------------------
    def describe(self) -> str:
        """Text overview: description, run type, number of runs, base configuration and sweep."""
        return "\n".join(
            [
                f"Study {self.name}",
                self.description,
                "",
                f"run type  : {self._run_type_path}",
                f"collector : {self.collector.__name__}",
                f"num runs  : {len(self.resolve_runs())}",
                "",
                "Base configuration",
                format_config_tree(self.base.to_json_dict()),
                "",
                "Sweep",
                self.sweep.describe(level=1),
            ]
        )

    # ----------------------------------------------------------------------------------------------
    def create_directory(self, definition_path: Path, root: Path) -> Path:
        """Resolve the study and write its study directory, before any run starts.

        The directory holds the study description, the configurations of all runs, the archived
        study module and the environment specification (see `EnvironmentArchive`), so that the
        study can be reproduced elsewhere.

        Args:
            definition_path (Path): The module file that defines the study; archived with it.
            root (Path): Directory holding all study directories.

        Returns:
            Path: The new study directory.

        Raises:
            FileExistsError: If the study directory already exists.
            ValueError: If a configuration does not survive its JSON form.
        """
        study_dir = (root / self.name).resolve()
        if study_dir.exists():
            raise FileExistsError(f"Study directory {study_dir} already exists.")
        runs = self.resolve_runs()
        self._check_configurations_survive_json(runs)

        directory = StudyDirectory(study_dir)
        directory.definition_path.parent.mkdir(parents=True)
        shutil.copy(definition_path, directory.definition_path)
        environment = Environment.collect_from_current_process()
        environment_files = EnvironmentArchive(
            environment, directory.environment_dir
        ).write_specification()
        directory.description_records.write_record(
            "study.json",
            asdict(self._describe_for_record(runs, definition_path, environment_files)),
        )
        directory.description_records.write_record("runs.json", self._index_runs(runs))
        directory.description_records.write_record("environment.json", asdict(environment))
        return study_dir

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load_from_directory(cls, study_dir: Path) -> Self:
        """Load the study from the module archived in its study directory."""
        return cls.load_from_module_file(StudyDirectory(study_dir).definition_path)

    # ----------------------------------------------------------------------------------------------
    def execute_runs(
        self,
        study_dir: Path,
        settings: ExecutorSettings,
        indices: list[int] | None = None,
        force: bool = False,
        wait: bool = True,
    ) -> dict[int, RunOutcome]:
        """Execute the unfinished runs of a created study.

        Finished runs are skipped unless forced, so executing again after failures or timeouts
        only repeats the others.

        If the code or the environment differs from the one recorded at creation (commit,
        uncommitted changes, `pixi.lock`, editable packages), a warning lists the differences before
        any run starts; every run records the environment it actually executed in.

        Args:
            study_dir (Path): Directory of the created study.
            settings (ExecutorSettings): Where and with which resources the runs execute.
            indices (list[int] | None): Runs to consider (repeated indices count once); all runs if
                `None`.
            force (bool): Whether to also rerun finished runs. Defaults to `False`.
            wait (bool): Whether to wait for the runs to finish; see `Executor.run`. Defaults to
                `True`.

        Returns:
            dict[int, RunOutcome]: Outcome by run index.

        Raises:
            ValueError: If an index is out of range, or if the study no longer resolves to the runs
                recorded in `study_dir`.
        """
        directory = StudyDirectory(study_dir)
        runs = self._resolve_recorded_runs(directory)
        selected = list(range(len(runs))) if indices is None else list(dict.fromkeys(indices))
        for index in selected:
            if not 0 <= index < len(runs):
                raise ValueError(f"Run index {index} is out of range for {len(runs)} runs.")
        outcomes: dict[int, RunOutcome] = {}
        pending: list[int] = []
        for index in selected:
            run_directory = directory.run_directory(runs[index].run_id)
            if not force and run_directory.read_state() == RunState.DONE:
                outcomes[index] = RunOutcome.SKIPPED
            else:
                pending.append(index)
        if pending:
            self._warn_if_environment_changed(directory)
        executor = Executor(settings, directory.job_dir, study_dir.resolve().name)
        results = executor.run(
            [self.run_type(runs[index].config) for index in pending],
            [directory.run_directory(runs[index].run_id).path for index in pending],
            wait,
        )
        outcomes.update(zip(pending, results, strict=True))
        return dict(sorted(outcomes.items()))

    # ----------------------------------------------------------------------------------------------
    def plot_finished_runs(self, study_dir: Path, index: int | None = None) -> list[int]:
        """Plot the finished runs of a created study; unfinished ones are skipped.

        Args:
            study_dir (Path): Directory of the created study.
            index (int | None): Only this run; all runs if `None`.

        Returns:
            list[int]: The indices of the runs that were plotted.

        Raises:
            ValueError: If the study no longer resolves to the runs recorded in `study_dir`.
        """
        directory = StudyDirectory(study_dir)
        plotted = []
        for run in self._resolve_recorded_runs(directory):
            if index is not None and run.index != index:
                continue
            run_directory = directory.run_directory(run.run_id)
            if run_directory.read_state() != RunState.DONE:
                continue
            print(f"Plotting run {run.index} ({run.run_id})")
            self.run_type(run.config).report(run_directory.path)
            plotted.append(run.index)
        return plotted

    # ----------------------------------------------------------------------------------------------
    def _resolve_recorded_runs(self, directory: StudyDirectory) -> list[ResolvedRun]:
        """Resolve the runs, checking that they are the ones recorded when the study was created."""
        runs = self.resolve_runs()
        recorded_ids = [entry["id"] for entry in directory.read_run_index()]
        if [run.run_id for run in runs] != recorded_ids:
            raise ValueError(
                f"The study no longer resolves to the runs recorded in {directory.path}: the study "
                "definition or the configuration classes changed since it was created."
            )
        return runs

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _warn_if_environment_changed(directory: StudyDirectory) -> None:
        """Warn if the code or the environment differs from the one recorded at creation.

        A study can be continued long after it was created, so its runs may stem from different
        code states; every run records the environment it executed in.
        """
        recorded = directory.description_records.read_record("environment.json")
        differences = Environment.collect_from_current_process().find_differences(recorded)
        if differences:
            lines = "\n  ".join(differences)
            warnings.warn(
                f"The environment differs from the one recorded when the study was created:\n  "
                f"{lines}",
                stacklevel=3,
            )

    # ----------------------------------------------------------------------------------------------
    @property
    def _run_type_path(self) -> str:
        """The run type as `module:Class`, by which a created study imports it."""
        return f"{self.run_type.__module__}:{self.run_type.__qualname__}"

    # ----------------------------------------------------------------------------------------------
    def _check_configurations_survive_json(self, runs: list[ResolvedRun]) -> None:
        """Raise if a configuration changes when written to JSON and read back.

        A created study reads its configurations back from `runs.json`.
        """
        for run in runs:
            if self.run_type.config_type.from_json_dict(run.config.to_json_dict()) != run.config:
                raise ValueError(f"The configuration of run {run.index} does not survive JSON.")

    # ----------------------------------------------------------------------------------------------
    def _describe_for_record(
        self, runs: list[ResolvedRun], definition_path: Path, environment_files: list[str]
    ) -> StudyRecord:
        return StudyRecord(
            name=self.name,
            description=self.description,
            created=datetime.now().astimezone().isoformat(timespec="seconds"),
            run_type=self._run_type_path,
            collector=self.collector.__name__,
            num_runs=len(runs),
            sweep=self.sweep.to_json_dict(),
            base_config=self.base.to_json_dict(),
            outputs=self.run_type.outputs,
            definition_sha256=hashlib.sha256(definition_path.read_bytes()).hexdigest(),
            environment_files=environment_files,
        )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _index_runs(runs: list[ResolvedRun]) -> list[dict[str, Any]]:
        """The run list as stored in `runs.json`."""
        return [
            {
                "index": run.index,
                "id": run.run_id,
                "overrides": ConfigCodec.encode(run.overrides),
                "config": run.config.to_json_dict(),
            }
            for run in runs
        ]

"""Studies: the definition of a study, and a study written to disk.

A study is a run type, a base configuration and a sweep over it. `StudySetup` is what a study module
(`studies/<name>.py`) defines as `STUDY`, together with the settings for executing it (local or
SLURM resources) and the root directory of the study. It resolves into a fixed,
ordered list of runs. A sweep is a tree of nodes, each expanding into a list of override dicts
(dotted configuration path to value) that are applied to the base configuration.

`Study` is a study written to disk: creating it, executing its runs, plotting and
summarizing them. A study is identified by its module; its directory is the root of its setup
(see `directories` for the layout).
`Study.create` resolves the module into the fixed run list and writes the study directory,
including the environment specification. Working on the study resolves the module again;
the runs must have the ids recorded at creation, since the results in the directory belong to them.

Classes:
    ResolvedRun: One run of a resolved study.
    SweepNode: A variation of configuration parameters, expanding into override dicts.
    Axis: One parameter and its values.
    Zip: Axes varied together.
    Product: All combinations of groups.
    StudySetup: Run type, base configuration, sweep and description.
    Study: A study directory together with the study setup it was created from.
"""

import importlib.util
import itertools
import json
import re
import shutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Self, override

import pandas as pd

from bayes_cep.run.config import TYPE_KEY, ConfigCodec, RunConfig, format_config_tree
from bayes_cep.run.directories import (
    CONFIG_RECORD,
    REPOSITORY_ROOT,
    RESULTS_DIR_NAME,
    RunState,
    StudyDirectory,
    format_current_time,
    repository_relative_path,
    resolve_repository_path,
    write_json_record,
)
from bayes_cep.run.executor import Executor, ExecutorSettings, RunOutcome
from bayes_cep.run.provenance import Environment, EnvironmentArchive
from bayes_cep.run.template import Run

STUDY_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


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

    def __init__(self, *groups: SweepNode) -> None:
        """Combine axes, zips and further products."""
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
class StudySetup:
    """A run type, the base configuration of its runs, the sweep over it, and a description.

    Attributes:
        name (str): Name of the study, also the name of its directory.
        description (str): What the study investigates and computes.
        run_type (type[Run]): The run type of all runs of the study.
        base (RunConfig): Configuration of all runs before the sweep is applied; fixes all
            parameters that are common to the runs.
        sweep (SweepNode): The parameter variations.
        root (Path): The study directory, conventionally named after the study. Other studies find
            the runs of this one through it. Not part of a run's identity. A study module sets it
            relative to itself, `module_relative_path(__file__, "../working_data/<name>")`, so that
            it works from any working directory (a bare relative path would be relative to the
            working directory). After construction `root` is relative to the repository root if the
            directory is inside it, else absolute (see `repository_relative_path`). This keeps the
            run ids independent of where the repository is checked out: the run id is a hash of
            the configuration, paths included, and a study that reads the results of another one
            (see `results_directories`) gets the paths in its configurations from that study's
            `root`. With an absolute `root` the checkout location would end up in the run ids: the
            same study would have other run ids in another checkout, a worktree or on the cluster,
            and a study created elsewhere (or before) would fail `Study.load`, which requires the
            run ids recorded at creation.
        executor (ExecutorSettings): Where and with which resources the runs execute, e.g. the
            SLURM resources. Not part of a run's identity.
    """

    name: str
    description: str
    run_type: type[Run]
    base: RunConfig
    sweep: SweepNode
    root: Path
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)

    def __post_init__(self) -> None:
        if not STUDY_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                f"Study name must match {STUDY_NAME_PATTERN.pattern}, got {self.name!r}."
            )
        if not isinstance(self.base, self.run_type.config_type):
            raise TypeError(
                f"{self.run_type.__name__} needs a {self.run_type.config_type.__name__} as base, "
                f"got {type(self.base).__name__}."
            )
        object.__setattr__(self, "root", repository_relative_path(self.root))

    # ----------------------------------------------------------------------------------------------
    @property
    def run_type_path(self) -> str:
        """The run type as `module:Class`."""
        return f"{self.run_type.__module__}:{self.run_type.__qualname__}"

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load_from_module_file(cls, module_path: Path) -> Self:
        """Load the study a study module file defines.

        The module is executed as a script. It should import only from `bayes_cep`, from the
        run types of `single_runs`, which are found through the `PYTHONPATH` of the pixi
        environment; those are not archived with a study (the commit and the patch of
        uncommitted changes in the recorded environment cover them).

        Args:
            module_path (Path): Python file defining `STUDY`.

        Returns:
            Self: The study.

        Raises:
            ValueError: If the file does not define a `StudySetup` named `STUDY`.
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
            TypeError: If an override value does not fit the type of its field.
        """
        runs: list[ResolvedRun] = []
        seen: dict[str, int] = {}
        for index, overrides in enumerate(self.sweep.expand_overrides()):
            config = self.base.with_overrides(overrides)
            run_id = config.run_id
            if run_id in seen:
                raise ValueError(
                    f"Runs {seen[run_id]} and {index} have the same run id {run_id}: identical "
                    "configurations (or, very unlikely, a hash collision)."
                )
            seen[run_id] = index
            runs.append(ResolvedRun(index, run_id, config, overrides))
        return runs

    # ----------------------------------------------------------------------------------------------
    def run_directories(self) -> list[Path]:
        """The run directories of all runs, in order, as they exist once the study is created.

        This lets a study refer to the results of another one, e.g. a MAP study to the
        preprocessing runs that produced its data. The paths are relative if `root` is.

        Returns:
            list[Path]: One run directory per run.
        """
        study_directory = StudyDirectory(self.root)
        return [study_directory.run_directory(run.run_id).path for run in self.resolve_runs()]

    # ----------------------------------------------------------------------------------------------
    def results_directories(self) -> list[Path]:
        """The result directories of all runs, in order (see `run_directories`).

        A study that reads the output files of another one points at these.
        """
        return [run_directory / RESULTS_DIR_NAME for run_directory in self.run_directories()]

    # ----------------------------------------------------------------------------------------------
    def results_directory(self, config: RunConfig) -> Path:
        """The result directory of the run with this configuration.

        Raises:
            ValueError: If the study has no run with this configuration.
        """
        return self.run_directory(config) / RESULTS_DIR_NAME

    # ----------------------------------------------------------------------------------------------
    def run_directory(self, config: RunConfig) -> Path:
        """The run directory of the run with this configuration (see `run_directories`).

        Raises:
            ValueError: If the study has no run with this configuration.
        """
        for run, run_directory in zip(self.resolve_runs(), self.run_directories(), strict=True):
            if run.run_id == config.run_id:
                return run_directory
        raise ValueError(f"Study {self.name!r} has no run with the configuration {config.run_id}.")

    # ----------------------------------------------------------------------------------------------
    def describe(self) -> str:
        """Text overview: description, run type, number of runs, base configuration and sweep."""
        return "\n".join(
            [
                f"Study {self.name}",
                self.description,
                "",
                f"run type  : {self.run_type_path}",
                f"num runs  : {len(self.resolve_runs())}",
                "executor  : "
                + ", ".join(f"{key} = {value}" for key, value in asdict(self.executor).items()),
                "",
                "Base configuration",
                format_config_tree(self.base.to_json_dict()),
                "",
                "Sweep",
                self.sweep.describe(level=1),
            ]
        )


# ==================================================================================================
class Study:
    """A study directory together with the study definition it was created from.

    Attributes:
        directory (StudyDirectory): The directory of the study.
        definition (StudySetup): The study as defined by its module.
    """

    def __init__(
        self,
        directory: StudyDirectory,
        definition: StudySetup,
        runs: list[ResolvedRun] | None = None,
        runs_verified: bool = True,
    ) -> None:
        """Bind a study definition to the directory created for it.

        Args:
            directory (StudyDirectory): The study directory.
            definition (StudySetup): The study definition.
            runs (list[ResolvedRun] | None): The runs of the study; resolved from `definition` if
                `None`.
            runs_verified (bool): Whether the run ids are those the current code resolves to. If
                not, the study only supports reading (see `load`).
        """
        self.directory = directory
        self.definition = definition
        self.runs = definition.resolve_runs() if runs is None else runs
        self.runs_verified = runs_verified

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def create(cls, module_path: Path) -> Self:
        """Resolve a study module and write its study record, before any run starts.

        The study directory (`root`) may already exist, e.g. as the folder holding the study
        module; the record `<root>/study/` is built under a temporary name inside it and moved into
        place at the end, so a failure or an interruption never leaves a half-written study behind.

        Args:
            module_path (Path): Python file defining `STUDY`.

        Returns:
            Self: The new study.

        Raises:
            FileExistsError: If the study has already been created, i.e. `<root>/study/` exists.
            FileNotFoundError: If input files of a run are missing, e.g. because the preprocessing
                study it reads from has not been run.
            ValueError: If a configuration does not survive its JSON form.
        """
        study = StudySetup.load_from_module_file(module_path)
        directory = StudyDirectory(resolve_repository_path(study.root))
        if directory.record_dir.exists():
            raise FileExistsError(f"Study already created, {directory.record_dir} exists.")
        runs = study.resolve_runs()
        missing_files: set[Path] = set()
        for run in runs:
            if study.run_type.config_type.from_json_dict(run.config.to_json_dict()) != run.config:
                raise ValueError(f"The configuration of run {run.index} does not survive JSON.")
            input_files = study.run_type(run.config).input_files()
            missing_files.update(path for path in input_files if not path.exists())
        if missing_files:
            listed = "\n  ".join(str(path) for path in sorted(missing_files))
            raise FileNotFoundError(f"Input files of the runs are missing:\n  {listed}")

        temporary_directory = StudyDirectory(directory.path / ".creating")
        shutil.rmtree(temporary_directory.path, ignore_errors=True)
        try:
            cls._write_description(temporary_directory, study, runs, module_path)
            temporary_directory.record_dir.rename(directory.record_dir)
        finally:
            shutil.rmtree(temporary_directory.path, ignore_errors=True)
        return cls(directory, study, runs)

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, module_path: Path, strict: bool = True) -> Self:
        """Load a study from its module.

        A study outlives the code version it was created with. Executing runs needs the run ids
        the current code resolves to (`strict`), since it writes into the run directories. Reading
        (status, collect, report) only needs the runs of the study: with `strict=False`, a module
        that resolves to other run ids, whatever the reason (the configuration classes, the module,
        or a study it reads from changed), is ignored for the runs; they are read from the study
        record and the run directories (`_read_recorded_runs`). Such a study has
        `runs_verified == False` and cannot execute runs; the configuration of a run is the one
        recorded in its run directory, not the one of `ResolvedRun.config`.

        Args:
            module_path (Path): Python file defining `STUDY`.
            strict (bool): Whether the module must resolve to the recorded run ids. Defaults to
                `True`.

        Raises:
            FileNotFoundError: If the study has not been created in the root of its module.
            ValueError: If `strict` and the module no longer resolves to the runs recorded at
                creation.
        """
        study = StudySetup.load_from_module_file(module_path)
        directory = StudyDirectory(resolve_repository_path(study.root))
        module_directory = StudyDirectory(module_path.resolve().parent)
        if not directory.study_record_path.exists() and module_directory.study_record_path.exists():
            # Another checkout of the code puts the root elsewhere than where the study is.
            directory = module_directory
        if not directory.study_record_path.exists():
            raise FileNotFoundError(f"Study {study.name!r} has not been created in {study.root}.")
        record = json.loads(directory.study_record_path.read_text())
        recorded_ids: list[str] = record["run_ids"]
        try:
            runs = study.resolve_runs()
        except (ValueError, TypeError):
            if strict:
                raise
            runs = []  # the module does not resolve under this code; read the record instead
        if [run.run_id for run in runs] == recorded_ids:
            return cls(directory, study, runs)
        if strict:
            raise ValueError(
                f"{module_path} no longer resolves to the runs recorded in {directory.path}: the "
                "study module or the configuration classes changed since it was created. "
                "Commands that only read the study (status, collect, report) accept this."
            )
        return cls(directory, study, cls._read_recorded_runs(directory, study, record), False)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _read_recorded_runs(
        directory: StudyDirectory, study: StudySetup, record: dict[str, Any]
    ) -> list[ResolvedRun]:
        """The runs of the study as recorded: their ids, and their swept parameters read from the
        `config.json` of each run directory.

        The module is not consulted, so this works however the module or the code changed. The
        `config` of the returned runs is the base configuration of the module, a placeholder; the
        configuration of a run is the one in its directory. A run that never started has no
        `config.json`, and `None` for its swept parameters.
        """
        paths: list[str] = []

        def collect_paths(node: dict[str, Any]) -> None:
            if node["kind"] == "axis":
                paths.append(node["path"])
            for child in node.get("axes", []) + node.get("groups", []):
                collect_paths(child)

        collect_paths(record["sweep"])
        runs = []
        for index, run_id in enumerate(record["run_ids"]):
            config_path = directory.run_directory(run_id).path / CONFIG_RECORD
            config = json.loads(config_path.read_text()) if config_path.exists() else None
            overrides: dict[str, Any] = {}
            for path in dict.fromkeys(paths):
                value = config
                for part in path.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                overrides[path] = value
            runs.append(ResolvedRun(index, run_id, study.base, overrides))
        return runs

    # ----------------------------------------------------------------------------------------------
    def read_states(self) -> dict[int, RunState]:
        """The state of every run, by run index."""
        return {
            run.index: self.directory.run_directory(run.run_id).read_state() for run in self.runs
        }

    # ----------------------------------------------------------------------------------------------
    def execute_runs(
        self,
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
        uncommitted changes, `pixi.lock`, editable packages), a warning lists the differences before
        any run starts; every run records the environment it actually executed in.

        Args:
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
            ValueError: If an index is out of range, or if the study was loaded with run ids
                other than the current code resolves to (see `load`).
        """
        if not self.runs_verified:
            raise ValueError("Runs can only be executed with the code the study was created with.")
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
            self._warn_if_environment_changed(environment)
            executor = Executor(
                self.definition.executor, self.directory.job_dir, self.directory.path.name
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
                self.definition.run_type(self._config_of(run, run_directory.path)).report(
                    run_directory.path
                )
            except Exception as error:
                print(f"Plotting run {run.index} failed: {error!r}")
                continue
            plotted.append(run.index)
        return plotted

    # ----------------------------------------------------------------------------------------------
    def _config_of(self, run: ResolvedRun, run_path: Path) -> RunConfig:
        """The configuration of a run: the resolved one, or, if the code changed since the study
        was created, the one recorded in the run directory."""
        if self.runs_verified:
            return run.config
        data = json.loads((run_path / CONFIG_RECORD).read_text())
        return self.definition.run_type.config_type.from_json_dict(data)

    # ----------------------------------------------------------------------------------------------
    def build_run_table(self) -> pd.DataFrame:
        """The cross-run table: one row per run with `index`, `run_id`, `state`, the swept
        parameters and the metrics.

        Swept parameters are columns named by their dotted path (config objects by class name),
        metrics are columns of their own. Runs that are not `done` have empty metrics. The table is
        regenerated from the run directories whenever it is wanted.

        Raises:
            ValueError: If a metric has the name of another column.
        """
        states = self.read_states()
        rows = []
        for run in self.runs:
            row: dict[str, Any] = {
                "index": run.index,
                "run_id": run.run_id,
                "state": states[run.index].value,
            }
            for path, value in run.overrides.items():
                row[path] = self._to_table_value(ConfigCodec.encode(value))
            metrics = self.directory.run_directory(run.run_id).read_metrics()
            clashes = row.keys() & metrics.keys()
            if clashes:
                raise ValueError(
                    f"Metrics of run {run.index} clash with columns: {sorted(clashes)}."
                )
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------------------------------------
    def write_run_table(self) -> Path:
        """Write the run table to `<study>/summary/run_table.parquet` and return its path."""
        self.directory.summary_dir.mkdir(exist_ok=True)
        table_path = self.directory.summary_dir / "run_table.parquet"
        self.build_run_table().to_parquet(table_path, index=False)
        return table_path

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _to_table_value(value: Any) -> Any:
        """Convert a swept value to a table cell; config objects become their class name."""
        if isinstance(value, dict) and TYPE_KEY in value:
            return value[TYPE_KEY]
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return str(value)

    # ----------------------------------------------------------------------------------------------
    def _warn_if_environment_changed(self, environment: Environment) -> None:
        """Warn if the code or the environment differs from the state at creation.

        A study can be continued long after it was created, so its runs may stem from different
        code states; every run records the environment it executed in.
        """
        recorded = json.loads(self.directory.environment_path.read_text())
        differences = environment.find_differences(recorded)
        if differences:
            lines = "\n  ".join(differences)
            warnings.warn(
                f"The code differs from the state recorded when the study was created:\n  {lines}",
                stacklevel=2,
            )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _write_description(
        directory: StudyDirectory,
        study: StudySetup,
        runs: list[ResolvedRun],
        module_path: Path,
    ) -> None:
        """Write the study description and the environment into `directory`."""
        environment = Environment.collect_from_current_process()
        environment_files = EnvironmentArchive(
            environment, directory.environment_dir
        ).write_specification()
        resolved_module = module_path.resolve()
        module = (
            resolved_module.relative_to(REPOSITORY_ROOT)
            if resolved_module.is_relative_to(REPOSITORY_ROOT)
            else resolved_module
        )
        write_json_record(
            directory.study_record_path,
            {
                "name": study.name,
                "description": study.description,
                "created": format_current_time(),
                "module": str(module),
                "executor": asdict(study.executor),
                "run_type": study.run_type_path,
                "run_ids": [run.run_id for run in runs],
                "sweep": study.sweep.to_json_dict(),
                "base_config": study.base.to_json_dict(),
                "outputs": study.run_type.outputs,
                "environment_files": environment_files,
            },
        )
        write_json_record(directory.environment_path, asdict(environment))

"""A study: a run type, a base configuration and sweeps, resolved into a fixed list of runs.

See `directories` for the layout of a study directory.

Classes:
    Axis: One parameter and its values.
    Zip: Axes varied together.
    Product: All combinations of groups.
    ResolvedRun: One run of a resolved study.
    Study: Run type, base configuration, sweep, description and collector.

Functions:
    apply_overrides: Replace nested configuration fields by dotted path.
    format_sweep: Text overview of a sweep.
    load_study: Load the `STUDY` of a study module file.
    load_resolved_runs: Read the runs of a created study back from its directory.
"""

import hashlib
import importlib
import importlib.util
import itertools
import re
import shutil
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from bayes_cep.run.collector import Collector, RunTableCollector
from bayes_cep.run.config import TYPE_KEY, RunConfig, format_tree, to_jsonable
from bayes_cep.run.directories import StudyDirectory
from bayes_cep.run.metadata import collect_environment
from bayes_cep.run.template import Run

STUDY_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


# ==================================================================================================
@dataclass(frozen=True)
class Axis:
    """One parameter, addressed by dotted path, and the values it takes.

    Attributes:
        path (str): Dotted path into the base configuration, e.g. `"prior.kappa"`.
        values (tuple[Any, ...]): Values of the parameter; a list is converted to a tuple.
    """

    path: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError(f"Axis {self.path!r} has no values.")

    def combinations(self) -> list[dict[str, Any]]:
        """One override per value."""
        return [{self.path: value} for value in self.values]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {"kind": "axis", "path": self.path, "values": to_jsonable(self.values)}


# ==================================================================================================
class Zip:
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

    def combinations(self) -> list[dict[str, Any]]:
        """One override dict per position."""
        return [
            {axis.path: axis.values[position] for axis in self.axes}
            for position in range(len(self.axes[0].values))
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {"kind": "zip", "axes": [axis.to_dict() for axis in self.axes]}


# ==================================================================================================
class Product:
    """All combinations of its groups; the last group varies fastest. No groups: a single run."""

    def __init__(self, *groups: Axis | Zip) -> None:
        """Combine axes and zips."""
        self.groups = groups

    def combinations(self) -> list[dict[str, Any]]:
        """One merged override dict per combination."""
        merged = []
        for parts in itertools.product(*(group.combinations() for group in self.groups)):
            combination: dict[str, Any] = {}
            for part in parts:
                combination.update(part)
            merged.append(combination)
        return merged

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return {"kind": "product", "groups": [group.to_dict() for group in self.groups]}


# ==================================================================================================
def apply_overrides(config: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    """Replace nested configuration fields addressed by dotted paths.

    Args:
        config (RunConfig): Base configuration; not modified.
        overrides (dict[str, Any]): Dotted path to new value.

    Returns:
        RunConfig: The updated configuration.

    Raises:
        ValueError: If a path does not address a field of the configuration.
    """
    for path, value in overrides.items():
        config = _replace_path(config, path, path.split("."), value)
    return config


# --------------------------------------------------------------------------------------------------
def _replace_path(obj: Any, full_path: str, parts: list[str], value: Any) -> Any:
    head, rest = parts[0], parts[1:]
    if not is_dataclass(obj) or head not in {field.name for field in fields(obj)}:
        raise ValueError(f"{full_path!r} is not a field of the configuration (at {head!r}).")
    if rest:
        value = _replace_path(getattr(obj, head), full_path, rest, value)
    return replace(obj, **{head: value})


# ==================================================================================================
def format_sweep(node: dict[str, Any], level: int = 0) -> str:
    """Indented text overview of a sweep in its `to_dict` form."""
    pad = "  " * level
    match node["kind"]:
        case "axis":
            labels = ", ".join(_label(value) for value in node["values"])
            return f"{pad}{node['path']}: {labels}"
        case kind:
            children = node["axes"] if kind == "zip" else node["groups"]
            lines = [f"{pad}{kind}"] + [format_sweep(child, level + 1) for child in children]
            return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
def _label(value: Any) -> str:
    """A swept value for display: scalars as they are, config objects by their class name."""
    return str(value[TYPE_KEY]) if isinstance(value, dict) and TYPE_KEY in value else str(value)


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
class Study:
    """A run type, the base configuration of its runs, the sweep over it, and a description.

    Attributes:
        name (str): Name of the study, also the name of its directory.
        description (str): What the study investigates and computes.
        run_type (type[Run]): The kind of run all runs of the study are.
        base (RunConfig): Configuration of all runs before the sweep is applied; fixes all
            parameters that are common to the runs.
        sweep (Axis | Zip | Product): The parameter variations.
        collector (type[Collector]): Analysis of the finished runs, matching this study. Defaults
            to writing the run table only.
    """

    name: str
    description: str
    run_type: type[Run]
    base: RunConfig
    sweep: Axis | Zip | Product
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
    def resolve(self) -> list[ResolvedRun]:
        """Expand the sweep into the ordered run list.

        Returns:
            list[ResolvedRun]: One entry per run.

        Raises:
            ValueError: If two runs resolve to the same configuration, or an override path is
                invalid.
        """
        runs: list[ResolvedRun] = []
        seen: dict[str, int] = {}
        for index, overrides in enumerate(self.sweep.combinations()):
            config = apply_overrides(self.base, overrides)
            if config.run_id in seen:
                raise ValueError(f"Runs {seen[config.run_id]} and {index} are identical.")
            seen[config.run_id] = index
            runs.append(ResolvedRun(index, config.run_id, config, overrides))
        return runs

    # ----------------------------------------------------------------------------------------------
    def describe(self) -> str:
        """Text overview: description, run type, number of runs, base configuration and sweep."""
        return "\n".join(
            [
                f"Study {self.name}",
                self.description,
                "",
                f"run type  : {_class_path(self.run_type)}",
                f"collector : {self.collector.__name__}",
                f"num runs  : {len(self.resolve())}",
                "",
                "Base configuration",
                format_tree(self.base.to_dict()),
                "",
                "Sweep",
                format_sweep(self.sweep.to_dict(), 1),
            ]
        )

    # ----------------------------------------------------------------------------------------------
    def create(self, definition_path: Path, root: Path) -> Path:
        """Resolve the study and write its description, before any run starts.

        Every configuration is checked to survive its JSON form unchanged, since workers read the
        configurations back from `runs.json`.

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
        runs = self.resolve()
        for run in runs:
            if self.run_type.config_type.from_dict(run.config.to_dict()) != run.config:
                raise ValueError(f"The configuration of run {run.index} does not survive JSON.")

        directory = StudyDirectory(study_dir)
        directory.definition_path.parent.mkdir(parents=True)
        shutil.copy(definition_path, directory.definition_path)
        directory.write(
            "study.json",
            {
                "name": self.name,
                "description": self.description,
                "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                "run_type": _class_path(self.run_type),
                "collector": self.collector.__name__,
                "num_runs": len(runs),
                "sweep": self.sweep.to_dict(),
                "base_config": self.base.to_dict(),
                "outputs": self.run_type.outputs,
                "definition_sha256": hashlib.sha256(definition_path.read_bytes()).hexdigest(),
            },
        )
        directory.write(
            "runs.json",
            [
                {
                    "index": run.index,
                    "id": run.run_id,
                    "overrides": to_jsonable(run.overrides),
                    "config": run.config.to_dict(),
                }
                for run in runs
            ],
        )
        directory.write("environment.json", collect_environment())
        return study_dir


# ==================================================================================================
def _class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


# ==================================================================================================
def load_study(module_path: Path) -> Study:
    """Load the `STUDY` attribute of a study module file.

    The module is executed as a script and must be self-contained apart from `bayes_cep`.

    Args:
        module_path (Path): Python file defining `STUDY`.

    Returns:
        Study: The study.

    Raises:
        ValueError: If the file does not define a `Study` named `STUDY`.
    """
    spec = importlib.util.spec_from_file_location(f"study_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load a study from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = getattr(module, "STUDY", None)
    if not isinstance(study, Study):
        raise ValueError(f"{module_path} must define STUDY as a Study.")
    return study


# ==================================================================================================
def load_resolved_runs(study_dir: Path) -> tuple[type[Run], list[ResolvedRun]]:
    """Read the run type and the runs of a created study from `study/study.json` and `runs.json`.

    The study definition is not imported: the configurations are decoded from their JSON form and
    checked against the recorded run ids.

    Args:
        study_dir (Path): Study directory.

    Returns:
        tuple[type[Run], list[ResolvedRun]]: The run type and the runs, ordered by index.

    Raises:
        ValueError: If a decoded configuration no longer has its recorded id, i.e. the
            configuration classes changed since the study was created.
    """
    directory = StudyDirectory(study_dir)
    module_name, _, class_name = directory.read("study.json")["run_type"].partition(":")
    run_type: type[Run] = getattr(importlib.import_module(module_name), class_name)
    runs = []
    for entry in directory.run_index():
        config = run_type.config_type.from_dict(entry["config"])
        if config.run_id != entry["id"]:
            raise ValueError(
                f"Run {entry['index']} no longer has id {entry['id']}: the configuration "
                "classes or their defaults changed since the study was created."
            )
        runs.append(ResolvedRun(entry["index"], entry["id"], config, entry["overrides"]))
    return run_type, runs

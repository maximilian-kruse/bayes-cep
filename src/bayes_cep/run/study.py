"""The definition of a study: a run type, a base configuration and a sweep over it.

`Study` is what a study module defines as `STUDY`. It resolves into a fixed, ordered list of runs.
Writing a study to disk and working with it afterwards is done by
[`CreatedStudy`][bayes_cep.run.created_study.CreatedStudy].

Classes:
    ResolvedRun: One run of a resolved study.
    Study: Run type, base configuration, sweep and description.
"""

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from bayes_cep.run.config import RunConfig, format_config_tree
from bayes_cep.run.sweep import SweepNode
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
    """

    name: str
    description: str
    run_type: type[Run]
    base: RunConfig
    sweep: SweepNode

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

    # ----------------------------------------------------------------------------------------------
    @property
    def run_type_path(self) -> str:
        """The run type as `module:Class`."""
        return f"{self.run_type.__module__}:{self.run_type.__qualname__}"

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load_from_module_file(cls, module_path: Path) -> Self:
        """Load the `STUDY` attribute of a study module file.

        The module is executed as a script. It should import only from `bayes_cep` and from the
        run kinds of `single_runs`, which are found through the `PYTHONPATH` of the pixi
        environment; those are not archived with a study (the commit and the patch of
        uncommitted changes in the recorded environment cover them).

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
    def describe(self) -> str:
        """Text overview: description, run type, number of runs, base configuration and sweep."""
        return "\n".join(
            [
                f"Study {self.name}",
                self.description,
                "",
                f"run type  : {self.run_type_path}",
                f"num runs  : {len(self.resolve_runs())}",
                "",
                "Base configuration",
                format_config_tree(self.base.to_json_dict()),
                "",
                "Sweep",
                self.sweep.describe(level=1),
            ]
        )

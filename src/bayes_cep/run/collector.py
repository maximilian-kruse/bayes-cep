"""Collection and analysis of the results of a study, after its runs have finished.

Everything here is regenerated from the run directories whenever it is wanted; runs never write to
the summary. A study-specific analysis is a subclass of `Collector`.

Classes:
    Collector: Abstract analysis of a study, writing to `<study>/summary`.
    RunTableCollector: The run table only.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from bayes_cep.run.directories import StudyDirectory


# ==================================================================================================
class Collector(ABC):
    """Analysis of the finished runs of one study.

    `collect` is the template method: it writes the run table `summary/run_table.parquet`, then
    calls `_analyze` for the study-specific part.

    Attributes:
        study_dir (Path): Study directory.
        summary_dir (Path): Where the table and the analysis are written.
    """

    def __init__(self, study_dir: Path) -> None:
        """Bind the collector to a study directory."""
        self.study = StudyDirectory(study_dir)
        self.study_dir = study_dir
        self.summary_dir = self.study.summary_dir

    # ----------------------------------------------------------------------------------------------
    def run_table(self) -> pd.DataFrame:
        """One row per run: `index`, `run_id`, `state`, swept parameters and metrics.

        Swept parameters are columns named by their dotted path (config objects by class name),
        metrics are columns of their own. Pending and failed runs have empty metrics.
        """
        rows = []
        for entry in self.study.run_index():
            run_dir = self.study.run(entry["id"])
            row: dict[str, Any] = {
                "index": entry["index"],
                "run_id": entry["id"],
                "state": run_dir.state(),
            }
            row.update({path: _cell(value) for path, value in entry["overrides"].items()})
            row.update(run_dir.metrics())
            rows.append(row)
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _analyze(self, table: pd.DataFrame) -> None:
        """Study-specific analysis, e.g. figures; write results to `self.summary_dir`.

        Args:
            table (pd.DataFrame): The run table.
        """

    # ----------------------------------------------------------------------------------------------
    def collect(self) -> Path:
        """Write the run table and run the analysis.

        Returns:
            Path: The run table file.
        """
        table = self.run_table()
        self.summary_dir.mkdir(exist_ok=True)
        table_path = self.summary_dir / "run_table.parquet"
        table.to_parquet(table_path, index=False)
        self._analyze(table)
        return table_path


# ==================================================================================================
class RunTableCollector(Collector):
    """The default collector: the run table, no further analysis."""

    def _analyze(self, table: pd.DataFrame) -> None:
        """Nothing to analyze beyond the table."""


# ==================================================================================================
def _cell(value: Any) -> Any:
    """A table cell for a swept value: scalars as they are, config objects by their class name."""
    if isinstance(value, dict) and "__type__" in value:
        return value["__type__"]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)

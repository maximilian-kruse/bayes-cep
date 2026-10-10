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

    Subclasses read the runs through `self._study` and write their results to `self._summary_dir`.
    """

    def __init__(self, study_dir: Path) -> None:
        """Bind the collector to a study directory."""
        self._study = StudyDirectory(study_dir)
        self._summary_dir = self._study.summary_dir

    # ----------------------------------------------------------------------------------------------
    def run_table(self) -> pd.DataFrame:
        """One row per run: `index`, `run_id`, `state`, swept parameters and metrics.

        Swept parameters are columns named by their dotted path (config objects by class name),
        metrics are columns of their own. Pending and failed runs have empty metrics.
        """
        rows = []
        for entry in self._study.read_run_index():
            run_dir = self._study.run_directory(entry["id"])
            row: dict[str, Any] = {
                "index": entry["index"],
                "run_id": entry["id"],
                "state": run_dir.read_state().value,
            }
            row.update(
                {
                    path: self._override_to_table_value(value)
                    for path, value in entry["overrides"].items()
                }
            )
            row.update(run_dir.read_metrics())
            rows.append(row)
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _override_to_table_value(value: Any) -> Any:
        """Convert a swept value to a table cell; config objects become their class name."""
        if isinstance(value, dict) and "__type__" in value:
            return value["__type__"]
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return str(value)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _analyze(self, table: pd.DataFrame) -> None:
        """Study-specific analysis, e.g. figures; write results to `self._summary_dir`.

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
        self._summary_dir.mkdir(exist_ok=True)
        table_path = self._summary_dir / "run_table.parquet"
        table.to_parquet(table_path, index=False)
        self._analyze(table)
        return table_path


# ==================================================================================================
class RunTableCollector(Collector):
    """The default collector: the run table, no further analysis."""

    def _analyze(self, table: pd.DataFrame) -> None:
        """Nothing to analyze beyond the table."""

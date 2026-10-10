"""The cross-run table of a study: one row per run, with its swept parameters and metrics.

It is regenerated from the run directories whenever it is wanted; runs never write to the summary.
Study-specific analyses read the table (or the run directories) themselves.

Functions:
    build_run_table: The table of a created study.
    write_run_table: Write the table to `<study>/summary/run_table.parquet`.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from bayes_cep.run.config import TYPE_KEY
from bayes_cep.run.created_study import CreatedStudy


# ==================================================================================================
def build_run_table(study: CreatedStudy) -> pd.DataFrame:
    """One row per run: `index`, `run_id`, `state`, swept parameters and metrics.

    Swept parameters are columns named by their dotted path (config objects by class name),
    metrics are columns of their own. Runs that are not `done` have empty metrics.

    Raises:
        ValueError: If a metric has the name of another column.
    """
    states = study.read_states()
    rows = []
    for run in study.runs:
        row: dict[str, Any] = {
            "index": run.index,
            "run_id": run.run_id,
            "state": states[run.index].value,
        }
        row.update({path: _to_table_value(value) for path, value in run.overrides.items()})
        metrics = study.directory.run_directory(run.run_id).read_metrics()
        clashes = row.keys() & metrics.keys()
        if clashes:
            raise ValueError(f"Metrics of run {run.index} clash with columns: {sorted(clashes)}.")
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
def write_run_table(study: CreatedStudy) -> Path:
    """Write the run table of `study` and return its path."""
    study.directory.summary_dir.mkdir(exist_ok=True)
    table_path = study.directory.summary_dir / "run_table.parquet"
    build_run_table(study).to_parquet(table_path, index=False)
    return table_path


# --------------------------------------------------------------------------------------------------
def _to_table_value(value: Any) -> Any:
    """Convert a swept value to a table cell; config objects become their class name."""
    if isinstance(value, dict) and TYPE_KEY in value:
        return value[TYPE_KEY]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)

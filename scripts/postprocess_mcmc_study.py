"""Plots that compare the samplers of one or more MCMC studies over the step width.

Reads `<study>/summary/run_table.parquet` (written by `bcep-study collect`) of every given study
and writes into the `summary/` directory of the first one:

- `mcmc_metrics_summary.csv`: for every study, algorithm and step width, the mean, minimum and
  maximum over the chain seeds of every metric in `METRICS`, and the number of runs.
- `mcmc_step_width_comparison.pdf`: one panel per metric against the step width (logarithmic),
  one line per algorithm and study (the mean over the seeds, with the range as a band); give the
  `mcmc_baseline` and the `mcmc_laplace` study together to compare the preconditioned samplers
  with the plain ones.

Only runs in the state `done` are used. The metrics are those of the run (`acceptance_rate`,
effective sample size per second, autocorrelation time), which makes the comparison between the
plain and the preconditioned run fair only if the chains have the same length.

Usage: python scripts/postprocess_mcmc_study.py STUDY_DIR [STUDY_DIR ...]
"""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from bayes_cep.run.directories import StudyDirectory

ALGORITHM = "sampler.algorithm"
STEP_WIDTH = "sampler.step_width"

# (column, axis label, logarithmic axis)
METRICS = (
    ("acceptance_rate", "acceptance rate", False),
    ("effective_sample_size_min", "minimum effective sample size", True),
    ("effective_sample_size_per_second_min", "minimum effective sample size per second", True),
    ("autocorrelation_time_max", "maximum autocorrelation time [samples]", True),
)
FIGURE_DPI = 150


# ==================================================================================================
def main() -> None:
    arguments = _parse_arguments()
    tables = {}
    for study_dir in arguments.study_dirs:
        study = StudyDirectory(study_dir)
        table = pd.read_parquet(study.summary_dir / "run_table.parquet")
        done = table[table["state"] == "done"]
        if done.empty:
            raise SystemExit(f"No finished run in {study_dir}.")
        tables[study_dir.name] = done
    summary_dir = StudyDirectory(arguments.study_dirs[0]).summary_dir

    summary = _summarize(tables)
    summary.to_csv(summary_dir / "mcmc_metrics_summary.csv", index=False)

    figure = Figure(figsize=(5 * len(METRICS), 4))
    axes = figure.subplots(1, len(METRICS))
    for axis, (column, label, logarithmic) in zip(axes, METRICS, strict=True):
        for (study_name, algorithm), group in summary.groupby(["study", "algorithm"]):
            group = group.sort_values("step_width")
            axis.plot(
                group["step_width"],
                group[f"{column}_mean"],
                marker="o",
                label=f"{study_name}: {algorithm}",
            )
            axis.fill_between(
                group["step_width"], group[f"{column}_min"], group[f"{column}_max"], alpha=0.2
            )
        axis.set_xscale("log")
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xlabel("step width")
        axis.set_ylabel(label)
    axes[0].legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(summary_dir / "mcmc_step_width_comparison.pdf", dpi=FIGURE_DPI)
    print(f"Wrote the summary to {summary_dir}.")


# ==================================================================================================
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "study_dirs", type=Path, nargs="+", help="The directories of the studies to compare."
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------------------
def _summarize(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean, minimum and maximum over the seeds of the metrics, per study, algorithm and width."""
    rows = []
    for study_name, table in tables.items():
        table = table.assign(
            **{ALGORITHM: table[ALGORITHM].astype(str).str.removeprefix("MCMCAlgorithmName.")}
        )
        for (algorithm, step_width), group in table.groupby([ALGORITHM, STEP_WIDTH]):
            row = {
                "study": study_name,
                "algorithm": algorithm,
                "step_width": float(step_width),
                "num_runs": len(group),
            }
            for column, _, _ in METRICS:
                row[f"{column}_mean"] = group[column].mean()
                row[f"{column}_min"] = group[column].min()
                row[f"{column}_max"] = group[column].max()
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

"""Summarize a prior recovery study: how well kappa and tau are recovered from known truths.

Reads `<study>/summary/run_table.parquet` (written by `bcep-study collect`), uses the runs in the
state `done`, and writes into `<study>/summary/`:

- `recovery_summary.csv`: for every smoothing scale and true (kappa, tau), the mean and standard
  deviation over the samples and meshes of the log errors of kappa, tau and kappa * tau (natural
  log of estimate over truth, so 0 is exact and 0.25 is 28% too large), the mean share of the grid
  within a factor 1.1 of the smallest loss (the width of the valley), and the number of runs.
- `estimates_versus_truth.pdf`: the estimates against the truths of kappa, tau and kappa * tau, one
  color per smoothing scale. Points on the dashed line are exact.

The numbers are printed per smoothing scale as well: the log errors of kappa * tau are the ones to
compare with the others, since the loss identifies the variance better than the correlation length
if the valley follows it.

Usage: python scripts/postprocess_prior_recovery_study.py STUDY_DIR
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bayes_cep.postprocessing.summary_plots import plot_estimates_versus_truth
from bayes_cep.run.directories import StudyDirectory

SCALE = "fit.smoothing_scale"
LOG_ERRORS = ("log_kappa_error", "log_tau_error", "log_kappa_tau_error")


# ==================================================================================================
def main() -> None:
    """Write the summary table and the plot of the study given on the command line."""
    arguments = _parse_arguments()
    study = StudyDirectory(arguments.study_dir)
    table = pd.read_parquet(study.summary_dir / "run_table.parquet")
    done = table[table["state"] == "done"]
    if done.empty:
        raise SystemExit("The study has no run in the state done.")

    grouped = done.groupby([SCALE, "true_kappa", "true_tau"])
    summary = grouped[[*LOG_ERRORS, "fraction_near_minimum"]].agg(["mean", "std"])
    summary.columns = [f"{column}_{statistic}" for column, statistic in summary.columns]
    summary.insert(0, "num_runs", grouped["run_id"].count())
    summary.reset_index().to_csv(study.summary_dir / "recovery_summary.csv", index=False)

    for scale, group in done.groupby(SCALE):
        print(f"smoothing scale {scale:g}: {len(group)} runs")
        for column in LOG_ERRORS:
            print(
                f"  {column:22s} mean {group[column].mean():+.3f}, "
                f"std {group[column].std():.3f}, mean |error| {group[column].abs().mean():.3f}"
            )
        print(f"  share of the grid near the minimum: {group['fraction_near_minimum'].mean():.3f}")

    plot_estimates_versus_truth(
        ["kappa", "tau", "kappa * tau"],
        np.column_stack(
            [done["true_kappa"], done["true_tau"], done["true_kappa"] * done["true_tau"]]
        ),
        np.column_stack(
            [done["best_kappa"], done["best_tau"], done["best_kappa"] * done["best_tau"]]
        ),
        [f"smoothing scale {scale:g}" for scale in done[SCALE]],
        study.summary_dir / "estimates_versus_truth.pdf",
    )
    print(f"Wrote recovery_summary.csv and estimates_versus_truth.pdf into {study.summary_dir}")


# --------------------------------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study_dir", type=Path, help="The directory of the study.")
    return parser.parse_args()


# ==================================================================================================
if __name__ == "__main__":
    main()

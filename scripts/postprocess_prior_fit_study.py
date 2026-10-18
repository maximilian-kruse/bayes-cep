"""Summarize a prior parameter study: loss maps of the kappa/tau plane per smoothing scale.

Reads `<study>/summary/run_table.parquet` (written by `bcep-study collect`) and the loss surfaces
of the runs in the state `done`, and writes into `<study>/summary/`, for every distance metric:

- `loss_maps_pooled_<metric>.pdf`: the loss summed over the patients, one panel per smoothing scale
  (`0` is the raw field). The star is the minimum and the dashed curve the line of constant
  kappa * tau through it; the valley is the region of small loss along it.
- `loss_maps_<metric>_scale<scale>.pdf`: for one smoothing scale, the loss of every patient and the
  pooled one.
- `best_parameters.csv`: the grid point of the smallest loss of every patient, of the pooled loss,
  and of the loss pooled over the other patients (`leave_one_out`, the estimate for a held-out
  patient).

Usage: python scripts/postprocess_prior_fit_study.py STUDY_DIR
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bayes_cep.postprocessing.summary_plots import plot_loss_maps
from bayes_cep.run.directories import StudyDirectory
from bayes_cep.statistics.prior_moment_fit import MomentFitResult, pool_losses
from single_runs.prior_fit import MOMENT_FIT_FILE_NAME

PATIENT = "raw_dir"
SCALE = "fit.smoothing_scale"
METRIC = "fit.distance_metric"


# ==================================================================================================
def main() -> None:
    """Write the loss maps and the table of best parameters of the study given on the command
    line."""
    arguments = _parse_arguments()
    study = StudyDirectory(arguments.study_dir)
    table = pd.read_parquet(study.summary_dir / "run_table.parquet")
    done = table[table["state"] == "done"]
    if done.empty:
        raise SystemExit("The study has no run in the state done.")

    best_rows = []
    for metric, metric_group in done.groupby(METRIC):
        pooled_by_scale = {}
        for scale, scale_group in metric_group.groupby(SCALE):
            results = {
                Path(str(row[PATIENT])).name: MomentFitResult.load(
                    study.run_directory(row["run_id"]).path / "results" / MOMENT_FIT_FILE_NAME
                )
                for _, row in scale_group.iterrows()
            }
            patients = sorted(results)
            reference = results[patients[0]]
            pooled = pool_losses(list(results.values()))
            pooled_by_scale[scale] = pooled
            title = f"{metric} distance, smoothing scale {scale:g}"
            plot_loss_maps(
                reference.kappas,
                reference.taus,
                [results[patient].loss for patient in patients] + [pooled],
                [*patients, f"pooled over {len(patients)} patients"],
                study.summary_dir / f"loss_maps_{metric}_scale{scale:g}.pdf",
                title,
            )
            best_rows.append(_best_row(metric, scale, "pooled", reference, pooled, None))
            for patient in patients:
                others = [results[other] for other in patients if other != patient]
                best_rows.append(
                    _best_row(
                        metric,
                        scale,
                        patient,
                        reference,
                        results[patient].loss,
                        pool_losses(others) if others else None,
                    )
                )
            kappas, taus = reference.kappas, reference.taus
        scales = sorted(pooled_by_scale)
        plot_loss_maps(
            kappas,
            taus,
            [pooled_by_scale[scale] for scale in scales],
            [f"smoothing scale {scale:g}" for scale in scales],
            study.summary_dir / f"loss_maps_pooled_{metric}.pdf",
            f"{metric} distance, loss pooled over the patients",
        )
    pd.DataFrame(best_rows).to_csv(study.summary_dir / "best_parameters.csv", index=False)
    print(f"Wrote the plots and best_parameters.csv into {study.summary_dir}")


# --------------------------------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study_dir", type=Path, help="The directory of the study.")
    return parser.parse_args()


# --------------------------------------------------------------------------------------------------
def _best_row(
    metric: str,
    scale: float,
    patient: str,
    grid: MomentFitResult,
    loss: np.ndarray,
    leave_one_out_loss: np.ndarray | None,
) -> dict[str, object]:
    """One row of `best_parameters.csv`: the smallest loss of a surface and of the surface pooled
    over the other patients."""
    row, column = np.unravel_index(np.argmin(loss), loss.shape)
    entry: dict[str, object] = {
        "distance_metric": metric,
        "smoothing_scale": scale,
        "patient": patient,
        "best_kappa": grid.kappas[row],
        "best_tau": grid.taus[column],
        "loss_min": loss[row, column],
        "leave_one_out_kappa": np.nan,
        "leave_one_out_tau": np.nan,
    }
    if leave_one_out_loss is not None:
        row, column = np.unravel_index(np.argmin(leave_one_out_loss), loss.shape)
        entry["leave_one_out_kappa"] = grid.kappas[row]
        entry["leave_one_out_tau"] = grid.taus[column]
    return entry


# ==================================================================================================
if __name__ == "__main__":
    main()

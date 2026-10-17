"""Plots that summarize a MAP study: heatmaps over the prior parameters and run comparisons.

Reads `<study>/summary/run_table.parquet` (written by `bcep-study collect`) and the run directories
of the study, and writes into `<study>/summary/`:

- `metrics_summary.csv`: for every preprocessing run (data set) and prior, the mean, minimum and
  maximum over the initial-guess seeds of every numeric metric of the runs, and the number of runs.
  Without a seed sweep, each prior has one run, and the three statistics coincide.
- `heatmap_<data>_<metric>.pdf`: one heatmap per preprocessing run (data set) and metric over
  `kappa` (columns) and `tau` (rows), each cell the mean over the initial-guess seeds, for every
  metric in `HEATMAP_METRICS`. With a seed sweep, also `error_spread_deg` (largest minus smallest
  mean error over the seeds, a measure of the sensitivity to the initial guess) and
  `loss_final_min` (the smallest final loss over the seeds).
- `comparison/<data>_..._error_cdf.pdf` and `..._optimization_history.pdf`: with a seed sweep, the
  runs of all initial guesses of one prior and data set in one figure; without one, the runs of all
  priors of one data set.

Only runs in the state `done` are used.

Usage: python scripts/postprocess_map_study.py STUDY_DIR [--raw-dir DIR] [--data-root DIR]

The configurations of a study that ran on a cluster name directories there. `--raw-dir` replaces the
raw data directory, and `--data-root` is a directory holding the preprocessing runs (`<run id>/
results`), used when the data directory of a run does not exist.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bayes_cep.postprocessing.curve_plots import plot_error_cdf, plot_optimization_histories
from bayes_cep.postprocessing.summary_plots import plot_parameter_heatmap
from bayes_cep.run.directories import CONFIG_RECORD, StudyDirectory
from bayes_cep.statistics.axial_error_metrics import ERROR_THRESHOLDS_DEGREES
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff
from single_runs.data import PreprocessedData, RawDataDirectory
from single_runs.map import MapResults, MapRunConfig

KAPPA = "problem.prior.kappa"
TAU = "problem.prior.tau"
DATA_DIR = "problem.data_dir"
SEED = "initial_guess.seed"

# Metrics shown as heatmaps of the mean over the seeds: (column, color bar label, log scale)
HEATMAP_METRICS = (
    ("error_mean_deg", "mean error [deg]", False),
    ("error_median_deg", "median error [deg]", False),
    ("error_p90_deg", "90th percentile of the error [deg]", False),
    ("error_p95_deg", "95th percentile of the error [deg]", False),
    ("error_p99_deg", "99th percentile of the error [deg]", False),
    ("error_max_deg", "maximum error [deg]", False),
    ("fraction_below_5_deg", "fraction of simplices below 5 deg", False),
    ("fraction_below_10_deg", "fraction of simplices below 10 deg", False),
    ("fraction_below_20_deg", "fraction of simplices below 20 deg", False),
    ("axial_concentration", "axial concentration", False),
    ("skill_score", "skill score", False),
    ("activation_time_rmse_all", "activation-time RMSE", False),
    ("activation_time_rmse_observed", "activation-time RMSE, observed", False),
    ("activation_time_rmse_unobserved", "activation-time RMSE, unobserved", False),
    ("loss_final", "mean final loss", True),
    ("loss_final_likelihood", "mean final likelihood part of the loss", True),
    ("loss_final_prior", "mean final prior part of the loss", True),
    ("loss_truth", "loss of the ground truth", True),
    ("num_iterations", "number of iterations", False),
)

# (file name part, column, how to reduce the seeds, color bar label, log scale)
HEATMAPS = tuple(
    (column, column, "mean", label, log_scale) for column, label, log_scale in HEATMAP_METRICS
)
# Only with a seed sweep, as they compare the runs of the seeds of one prior
SEED_HEATMAPS = (
    ("error_spread_deg", "error_mean_deg", "spread", "spread of the mean error [deg]", False),
    ("loss_final_min", "loss_final", "min", "smallest final loss", True),
)


# ==================================================================================================
def main() -> None:
    """Write the heatmaps and the run comparisons of the study given on the command line."""
    arguments = _parse_arguments()
    study = StudyDirectory(arguments.study_dir)
    table = pd.read_parquet(study.summary_dir / "run_table.parquet")
    done = table[table["state"] == "done"]
    if done.empty:
        raise SystemExit("The study has no run in the state done.")
    has_seeds = SEED in done.columns
    (study.summary_dir / "comparison").mkdir(exist_ok=True)
    _write_metrics_summary(done, study.summary_dir / "metrics_summary.csv", has_seeds)

    for data_dir, data_group in done.groupby(DATA_DIR):
        data_label = _data_label(Path(str(data_dir)), arguments.data_root)
        for file_part, column, reduction, label, log_scale in (
            HEATMAPS + SEED_HEATMAPS if has_seeds else HEATMAPS
        ):
            kappas = sorted(data_group[KAPPA].unique())
            taus = sorted(data_group[TAU].unique())
            values = np.full((len(taus), len(kappas)), np.nan)
            for (kappa, tau), prior_group in data_group.groupby([KAPPA, TAU]):
                values[taus.index(tau), kappas.index(kappa)] = _reduce(
                    prior_group[column], reduction
                )
            plot_parameter_heatmap(
                values,
                taus,
                kappas,
                "tau",
                "kappa",
                label,
                study.summary_dir / f"heatmap_{data_label}_{file_part}.pdf",
                title=data_label,
                log_scale=log_scale,
            )

    if has_seeds:
        for (data_dir, kappa, tau), group in done.groupby([DATA_DIR, KAPPA, TAU]):
            _plot_comparison(
                study,
                group.sort_values(SEED),
                Path(str(data_dir)),
                f"_kappa{kappa:g}_tau{tau:g}",
                f"kappa {kappa:g}, tau {tau:g}",
                [f"seed {int(seed)}" for seed in group.sort_values(SEED)[SEED]],
                arguments,
            )
    else:
        for data_dir, group in done.groupby(DATA_DIR):
            group = group.sort_values([KAPPA, TAU])
            _plot_comparison(
                study,
                group,
                Path(str(data_dir)),
                "",
                "all priors",
                [
                    f"kappa {kappa:g}, tau {tau:g}"
                    for kappa, tau in zip(group[KAPPA], group[TAU], strict=True)
                ],
                arguments,
            )
    print(f"Wrote the plots into {study.summary_dir}")


# --------------------------------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study_dir", type=Path, help="The directory of the study.")
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="Replaces the raw data directory."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory with the preprocessing runs, used if a data directory does not exist.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------------------
def _write_metrics_summary(done: pd.DataFrame, path: Path, has_seeds: bool) -> None:
    """Write the mean, minimum and maximum over the seeds of every numeric metric, per data set and
    prior."""
    excluded = ("index", KAPPA, TAU, *((SEED,) if has_seeds else ()))
    metric_columns = [
        column
        for column in done.select_dtypes("number").columns
        if column not in excluded and not column.startswith("problem.")
    ]
    summary = done.groupby([DATA_DIR, KAPPA, TAU])[metric_columns].agg(["mean", "min", "max"])
    summary.columns = [f"{column}_{statistic}" for column, statistic in summary.columns]
    summary.insert(0, "num_runs", done.groupby([DATA_DIR, KAPPA, TAU])["run_id"].count())
    summary.reset_index().to_csv(path, index=False)


# --------------------------------------------------------------------------------------------------
def _reduce(values: pd.Series, reduction: str) -> float:
    """Reduce the values of the seeds of one prior to one number."""
    finite = values[np.isfinite(values)]
    if finite.empty:
        return float("nan")
    match reduction:
        case "mean":
            return float(finite.mean())
        case "min":
            return float(finite.min())
        case "spread":
            return float(finite.max() - finite.min())
    raise ValueError(f"Unknown reduction {reduction!r}.")


# --------------------------------------------------------------------------------------------------
def _resolve_data_dir(data_dir: Path, data_root: Path | None) -> Path:
    """The directory with the preprocessed data, looked up in `data_root` if it does not exist."""
    if data_dir.exists() or data_root is None:
        return data_dir
    return data_root / data_dir.parent.name / data_dir.name


# --------------------------------------------------------------------------------------------------
def _data_label(data_dir: Path, data_root: Path | None) -> str:
    """Name a preprocessing run by its number of observations and noise variance if its
    configuration can be read, else by the name of its directory."""
    config_path = _resolve_data_dir(data_dir, data_root).parent / CONFIG_RECORD
    if config_path.exists():
        observations = json.loads(config_path.read_text()).get("observations", {})
        if observations:
            return f"n{observations['num_observations']}_var{observations['noise_variance']:g}"
    return data_dir.parent.name


# --------------------------------------------------------------------------------------------------
def _plot_comparison(
    study: StudyDirectory,
    group: pd.DataFrame,
    data_dir: Path,
    name_suffix: str,
    title: str,
    labels: list[str],
    arguments: argparse.Namespace,
) -> None:
    """Plot the error distributions and the optimization histories of the runs in `group`, one
    curve per run labeled by `labels`, into `comparison/<data><name_suffix>_*.pdf`."""
    data = PreprocessedData.load(_resolve_data_dir(data_dir, arguments.data_root))
    errors, losses, gradient_norms = [], [], []
    for run_id in group["run_id"]:
        run_path = study.run_directory(run_id).path
        config = MapRunConfig.from_json_dict(json.loads((run_path / CONFIG_RECORD).read_text()))
        results = MapResults.load(run_path / "results")
        if not errors:
            raw_dir = arguments.raw_dir or config.problem.raw_dir
            mesh = RawDataDirectory.from_configured_path(raw_dir).load_mesh()
            to_simplices = config.problem.interpolation.assemble_matrix(
                mesh.points, np.asarray(mesh.cells).reshape(-1, 4)[:, 1:]
            )
            ground_truth_simplices = to_simplices @ data.ground_truth
        difference = compute_axial_data_diff(
            to_simplices @ results.estimate, ground_truth_simplices
        )
        errors.append(np.degrees(np.abs(difference)))
        losses.append(results.loss_history)
        gradient_norms.append(results.gradient_norm_history)
    name = f"{_data_label(data_dir, arguments.data_root)}{name_suffix}"
    plot_error_cdf(
        errors,
        labels,
        study.summary_dir / "comparison" / f"{name}_error_cdf.pdf",
        ERROR_THRESHOLDS_DEGREES,
        title,
    )
    plot_optimization_histories(
        losses,
        gradient_norms,
        labels,
        study.summary_dir / "comparison" / f"{name}_optimization_history.pdf",
        title,
    )


# ==================================================================================================
if __name__ == "__main__":
    main()

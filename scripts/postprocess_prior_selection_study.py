"""Summarize a prior selection study: which (kappa, tau) is best, with and without the truth.

Reads `<study>/summary/run_table.parquet` (written by `bcep-study collect`) of a MAP study over a
(kappa, tau) grid and, optionally, of the Laplace study of its MAP estimates and of the
cross-validation study (held-out MAP runs over the same grid and data sets), and writes:

Into `<map study>/summary/` (the choice by the known ground truth):
- `prior_selection_known_truth.csv`: for every group of data sets and every (kappa, tau), the mean
  and the standard deviation over the data sets of the skill score and the mean over them of the
  mean axial error. A group is the data sets with the same ground truth: the same true
  (kappa, tau) for prior samples (the sample seeds and observation seeds are averaged), else one
  group.
- `heatmap_<group>_mean_skill.pdf` and `heatmap_<group>_mean_error.pdf`.

With `--laplace-study-dir`, into `<laplace study>/summary/` (the choice without the truth):
- `prior_selection_evidence.csv`: the log evidence relative to the best pair of each data set,
  averaged over the data sets of a group, per (kappa, tau). The log evidence is
  `-loss_final(MAP) + 1/2 log_determinant_ratio(Laplace)` (see
  `bayes_cep.statistics.prior_selection.compute_laplace_log_evidence`).
- `prior_selection_evidence_validation.csv`: for every data set the pair of highest evidence, the
  pair of highest skill, the skill lost by following the evidence (regret) and the rank correlation
  of evidence and skill over the grid.
- `heatmap_<group>_mean_evidence.pdf`.

With `--cv-study-dir`, into `<cv study>/summary/` (the choice without the truth, by prediction):
- `prior_selection_cross_validation.csv`, `prior_selection_cross_validation_validation.csv` and
  `heatmap_<group>_mean_cross_validation.pdf`: the same, with the score
  `-mean negative log predictive density` of the held-out observations averaged over the folds
  (pairs that miss a fold are left out).

All runs in the state `done` are used, converged or not. The pairs found are printed, with the true
pair for synthetic ground truths.

Usage: python scripts/postprocess_prior_selection_study.py MAP_STUDY_DIR
           [--laplace-study-dir DIR] [--cv-study-dir DIR] [--data-root DIR]

The configurations of a study that ran on a cluster name directories there. `--data-root` is a
directory holding the preprocessing runs (`<run id>/results`), used when a data directory does not
exist, to read the configuration of the preprocessing run.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bayes_cep.postprocessing.summary_plots import plot_parameter_heatmap
from bayes_cep.run.directories import CONFIG_RECORD, StudyDirectory
from bayes_cep.statistics.prior_selection import compute_laplace_log_evidence, validate_selection

KAPPA = "problem.prior.kappa"
TAU = "problem.prior.tau"
DATA_DIR = "problem.data_dir"
GROUP = "group"
DATA_SET = "data_set"


# ==================================================================================================
@dataclass(frozen=True)
class DataSetInfo:
    """What the preprocessing run of a data directory says about its data set.

    Attributes:
        group (str): Name of the ground truth: the true (kappa, tau) of a prior sample, else
            `patient_field` (`unknown` if the configuration cannot be read).
        label (str): Name of the data set within the group (observations, seeds).
        true_pair (tuple[float, float] | None): True (kappa, tau) of a prior sample, else `None`.
        num_observations (int | None): Number of observations, `None` if the configuration cannot
            be read.
    """

    group: str
    label: str
    true_pair: tuple[float, float] | None
    num_observations: int | None


# ==================================================================================================
def main() -> None:
    """Write the summaries of the study given on the command line."""
    arguments = _parse_arguments()
    map_study = StudyDirectory(arguments.map_study_dir)
    runs = _keep_best_start(_read_done_runs(map_study))
    infos = {
        data_dir: _describe_data_set(Path(str(data_dir)), arguments.data_root)
        for data_dir in runs[DATA_DIR].unique()
    }
    runs[GROUP] = runs[DATA_DIR].map(lambda data_dir: infos[data_dir].group)
    runs[DATA_SET] = runs[DATA_DIR].map(lambda data_dir: infos[data_dir].label)
    runs["num_observations"] = runs[DATA_DIR].map(lambda data_dir: infos[data_dir].num_observations)
    not_converged = int((~runs["success"].astype(bool)).sum())
    print(
        f"{len(runs)} MAP runs, {not_converged} of them stopped before the gradient norm "
        "tolerance; all are used"
    )
    true_pairs = {info.group: info.true_pair for info in infos.values()}

    map_study.summary_dir.mkdir(exist_ok=True)
    print("== Known ground truth: best pair by the skill of the MAP estimate ==")
    _write_known_truth(runs, map_study, true_pairs)

    if arguments.laplace_study_dir is not None:
        laplace_study = StudyDirectory(arguments.laplace_study_dir)
        laplace_study.summary_dir.mkdir(exist_ok=True)
        print("\n== Unknown ground truth: best pair by the Laplace evidence ==")
        _write_evidence(runs, laplace_study, true_pairs)
    if arguments.cv_study_dir is not None:
        cv_study = StudyDirectory(arguments.cv_study_dir)
        cv_study.summary_dir.mkdir(exist_ok=True)
        print("\n== Unknown ground truth: best pair by cross-validation ==")
        _write_cross_validation(runs, cv_study, true_pairs)


# --------------------------------------------------------------------------------------------------
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("map_study_dir", type=Path, help="The directory of the MAP study.")
    parser.add_argument(
        "--laplace-study-dir",
        type=Path,
        default=None,
        help="The directory of the Laplace study of the MAP estimates.",
    )
    parser.add_argument(
        "--cv-study-dir",
        type=Path,
        default=None,
        help="The directory of the cross-validation study (held-out MAP runs over the folds).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory with the preprocessing runs, used if a data directory does not exist.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------------------
def _read_done_runs(study: StudyDirectory) -> pd.DataFrame:
    """The runs of a study in the state done, from its run table."""
    table = pd.read_parquet(study.summary_dir / "run_table.parquet")
    done = table[table["state"] == "done"].copy()
    if done.empty:
        raise SystemExit(f"The study {study.path} has no run in the state done.")
    return done


# --------------------------------------------------------------------------------------------------
def _keep_best_start(runs: pd.DataFrame) -> pd.DataFrame:
    """One run per (data set, kappa, tau): the one with the lowest final loss.

    A MAP study with several initial guesses has several runs per pair; the best optimum is the
    one to judge the pair by (and the one the Laplace approximation should be centered at).
    """
    keys = [DATA_DIR, KAPPA, TAU]
    best = runs.sort_values("loss_final").drop_duplicates(keys, keep="first")
    if len(best) < len(runs):
        print(
            f"Kept the run of lowest final loss for each of {len(best)} pairs "
            f"and dropped {len(runs) - len(best)} others"
        )
    return best.sort_index()


# --------------------------------------------------------------------------------------------------
def _describe_data_set(data_dir: Path, data_root: Path | None) -> DataSetInfo:
    """Name the ground truth and the data set of a data directory from its preprocessing run."""
    if not data_dir.exists() and data_root is not None:
        data_dir = data_root / data_dir.parent.name / data_dir.name
    config_path = data_dir.parent / CONFIG_RECORD
    if not config_path.exists():
        return DataSetInfo("unknown", data_dir.parent.name, None, None)
    config = json.loads(config_path.read_text())
    observations = config["observations"]
    num_observations = int(observations["num_observations"])
    label = f"n{num_observations}_seed{observations['seed']}"
    truth = config["ground_truth"].get("prior")
    if truth is None:
        return DataSetInfo("patient_field", label, None, num_observations)
    pair = (float(truth["kappa"]), float(truth["tau"]))
    return DataSetInfo(
        f"truth_kappa{pair[0]:g}_tau{pair[1]:g}",
        f"{label}_truthseed{truth['seed']}",
        pair,
        num_observations,
    )


# --------------------------------------------------------------------------------------------------
def _grid(frame: pd.DataFrame, column: str) -> tuple[list[float], list[float], np.ndarray]:
    """The values of `column` of runs with distinct (kappa, tau) on the grid of the parameters.

    Returns:
        The sorted taus (rows), the sorted kappas (columns) and the values with shape
        `(len(taus), len(kappas))`, `nan` where there is no run.
    """
    kappas = sorted(frame[KAPPA].unique())
    taus = sorted(frame[TAU].unique())
    values = np.full((len(taus), len(kappas)), np.nan)
    for kappa, tau, value in zip(frame[KAPPA], frame[TAU], frame[column], strict=True):
        values[taus.index(tau), kappas.index(kappa)] = value
    return taus, kappas, values


# --------------------------------------------------------------------------------------------------
def _average_over_data_sets(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean and standard deviation of `column` over the data sets per (kappa, tau), only for pairs
    that have a finite value in every data set of the frame."""
    num_data_sets = frame[DATA_SET].nunique()
    finite = frame[np.isfinite(frame[column])]
    grouped = finite.groupby([KAPPA, TAU])[column].agg(["mean", "std", "count"])
    complete = grouped[grouped["count"] == num_data_sets].copy()
    complete["std"] = complete["std"].fillna(0.0)
    return complete.reset_index()


# --------------------------------------------------------------------------------------------------
def _write_known_truth(
    runs: pd.DataFrame, study: StudyDirectory, true_pairs: dict[str, tuple[float, float] | None]
) -> None:
    """Average the skill and the error over the data sets of each ground truth and write them."""
    rows = []
    for group, group_runs in runs.groupby(GROUP):
        skill = _average_over_data_sets(group_runs, "skill_score")
        error = _average_over_data_sets(group_runs, "error_mean_deg")
        table = skill.merge(error, on=[KAPPA, TAU], suffixes=("_skill", "_error"))
        table = table.rename(
            columns={"mean_error": "mean_error_mean_deg", "count_skill": "num_data_sets"}
        ).drop(columns=["std_error", "count_error"])
        table.insert(0, GROUP, group)
        rows.append(table)
        for column, label, name in (
            ("mean_skill", "mean skill score", "mean_skill"),
            ("mean_error_mean_deg", "mean axial error [deg]", "mean_error"),
        ):
            taus, kappas, values = _grid(table, column)
            plot_parameter_heatmap(
                values,
                taus,
                kappas,
                "tau",
                "kappa",
                label,
                study.summary_dir / f"heatmap_{group}_{name}.pdf",
                title=f"{group}, {group_runs[DATA_SET].nunique()} data sets",
            )
        _print_best(group, table, "mean_skill", "mean skill", true_pairs[group])
    pd.concat(rows).to_csv(study.summary_dir / "prior_selection_known_truth.csv", index=False)
    print(f"Wrote the known truth summary into {study.summary_dir}")


# --------------------------------------------------------------------------------------------------
def _print_best(
    group: str,
    table: pd.DataFrame,
    column: str,
    name: str,
    true_pair: tuple[float, float] | None,
) -> None:
    """Print the pair with the largest value of `column`, with the true pair if there is one."""
    if table.empty:
        print(f"{group}: no pair has a finite value in every data set")
        return
    best = table.loc[table[column].idxmax()]
    suffix = "" if true_pair is None else f" (true: kappa {true_pair[0]:g}, tau {true_pair[1]:g})"
    print(f"{group}: kappa {best[KAPPA]:g}, tau {best[TAU]:g}, {name} {best[column]:.3f}{suffix}")


# --------------------------------------------------------------------------------------------------
def _write_evidence(
    map_runs: pd.DataFrame,
    study: StudyDirectory,
    true_pairs: dict[str, tuple[float, float] | None],
) -> None:
    """Form the log evidence of every MAP run from its Laplace run and select by it."""
    laplace = _read_done_runs(study)
    # The Laplace run names the MAP estimate it started at: <map study>/runs/<run id>/results/...
    laplace["map_run_id"] = laplace["map_estimate_path"].map(
        lambda path: Path(path).parent.parent.name
    )
    scored = map_runs.merge(
        laplace[["map_run_id", "log_determinant_ratio", "laplace.num_eigenvalues"]],
        left_on="run_id",
        right_on="map_run_id",
        how="inner",
    )
    if scored.empty:
        raise SystemExit("No Laplace run belongs to a finished run of the MAP study.")
    too_small = scored["laplace.num_eigenvalues"] < scored["num_observations"]
    if too_small.any():
        print(
            f"WARNING: {int(too_small.sum())} Laplace runs keep fewer eigenpairs than there are "
            "observations; their log-determinant ratio is truncated and the evidence too high"
        )
    scored["score"] = compute_laplace_log_evidence(
        scored["loss_final"].to_numpy(), scored["log_determinant_ratio"].to_numpy()
    )
    _write_selection(
        scored,
        study,
        true_pairs,
        Selection("evidence", "log evidence", "mean relative log evidence"),
    )


# --------------------------------------------------------------------------------------------------
def _write_cross_validation(
    map_runs: pd.DataFrame,
    study: StudyDirectory,
    true_pairs: dict[str, tuple[float, float] | None],
) -> None:
    """Average the held-out score of every pair over the folds and select by it."""
    folds = _read_done_runs(study)
    keys = [DATA_DIR, KAPPA, TAU]
    per_pair = folds.groupby(keys).agg(
        mean_negative_log_predictive=("heldout_negative_log_predictive", "mean"),
        mean_heldout_rmse=("heldout_rmse", "mean"),
        num_folds=("run_id", "count"),
    )
    # A pair with a missing fold is not comparable with the others
    per_pair = per_pair[per_pair["num_folds"] == per_pair["num_folds"].max()].reset_index()
    scored = map_runs.merge(per_pair, on=keys, how="inner")
    if scored.empty:
        raise SystemExit("No cross-validated pair belongs to a finished run of the MAP study.")
    # Larger is better, as for the evidence
    scored["score"] = -scored["mean_negative_log_predictive"]
    _write_selection(
        scored,
        study,
        true_pairs,
        Selection("cross_validation", "cross-validation score", "mean relative CV score"),
    )


# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Selection:
    """A way to select the prior parameters without the truth.

    Attributes:
        name (str): Used in the file names, e.g. `evidence`.
        score_label (str): Names the score (larger is better) in the color bars.
        print_label (str): Names the averaged relative score in the printed best pair.
    """

    name: str
    score_label: str
    print_label: str


# --------------------------------------------------------------------------------------------------
def _write_selection(
    scored: pd.DataFrame,
    study: StudyDirectory,
    true_pairs: dict[str, tuple[float, float] | None],
    selection: Selection,
) -> None:
    """Rank the pairs of each data set by `score`, average the ranking over the data sets of a
    ground truth, write it and check it against the skill of the MAP estimates.

    Args:
        scored (pd.DataFrame): The runs of the MAP study with the column `score` (larger is
            better) of the pair, one row per pair and data set.
        study (StudyDirectory): The study the summary is written into.
        true_pairs (dict): The true (kappa, tau) of each group, `None` if there is none.
        selection (Selection): Names of the way to select.
    """
    # Relative to the best pair of the data set, so that data sets can be averaged
    scored["score_relative"] = scored["score"] - scored.groupby(DATA_DIR)["score"].transform("max")
    mean_column = f"mean_{selection.name}_score_relative_to_best"
    averaged, validations = [], []
    for group, group_runs in scored.groupby(GROUP):
        table = _average_over_data_sets(group_runs, "score_relative")
        table = table.rename(
            columns={
                "mean": mean_column,
                "std": f"std_{selection.name}_score_relative_to_best",
                "count": "num_data_sets",
            }
        )
        table.insert(0, GROUP, group)
        averaged.append(table)
        taus, kappas, values = _grid(table, mean_column)
        plot_parameter_heatmap(
            values,
            taus,
            kappas,
            "tau",
            "kappa",
            f"mean {selection.score_label} relative to the best pair",
            study.summary_dir / f"heatmap_{group}_mean_{selection.name}.pdf",
            title=f"{group}, {group_runs[DATA_SET].nunique()} data sets",
        )
        _print_best(group, table, mean_column, selection.print_label, true_pairs[group])
        for data_set, data_runs in group_runs.groupby(DATA_SET):
            validation_row = _validate_data_set(group, str(data_set), data_runs)
            if validation_row is not None:
                validations.append(validation_row)
    pd.concat(averaged).to_csv(
        study.summary_dir / f"prior_selection_{selection.name}.csv", index=False
    )
    validation = pd.DataFrame(validations)
    validation.to_csv(
        study.summary_dir / f"prior_selection_{selection.name}_validation.csv", index=False
    )
    print(f"Wrote the {selection.name} summary into {study.summary_dir}")
    print(
        f"\nPer data set: the pair of highest {selection.score_label} against the pair of "
        "highest skill"
    )
    print(validation.to_string(index=False, float_format=lambda value: f"{value:.3g}"))


# --------------------------------------------------------------------------------------------------
def _validate_data_set(group: str, data_set: str, runs: pd.DataFrame) -> dict[str, object] | None:
    """Check the choice by the score against the choice by the skill for one data set.

    Pairs without a finite score or skill are left out. Returns `None`, with a message, if fewer
    than three pairs remain.
    """
    runs = runs[np.isfinite(runs["score"]) & np.isfinite(runs["skill_score"])].reset_index(
        drop=True
    )
    if len(runs) < 3:
        print(
            f"Skipped the validation of {group}/{data_set}: fewer than 3 pairs with a finite skill"
        )
        return None
    result = validate_selection(
        runs["score"].to_numpy(dtype=float), runs["skill_score"].to_numpy(dtype=float)
    )
    selected_run, skill_run = runs.loc[result.selected_index], runs.loc[result.skill_index]
    return {
        GROUP: group,
        DATA_SET: data_set,
        "selected_kappa": selected_run[KAPPA],
        "selected_tau": selected_run[TAU],
        "skill_kappa": skill_run[KAPPA],
        "skill_tau": skill_run[TAU],
        "skill_at_selected_pair": selected_run["skill_score"],
        "best_skill": skill_run["skill_score"],
        "regret": result.regret,
        "rank_correlation": result.rank_correlation,
    }


# ==================================================================================================
if __name__ == "__main__":
    main()

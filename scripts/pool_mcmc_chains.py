r"""Pool the chains of MCMC runs into one posterior summary of the fiber-angle field.

A production result is the posterior of many chains together, not of one. The script reads finished
runs (their directories, e.g. `<study>/runs/*`) and writes into `--output`:

- `pooled_axial_mean.npy`, `pooled_axial_variance.npy`: per vertex, the axial mean and variance of
  all samples after the burn-in of every chain (the mean resultant of the doubled angles, pooled
  with the number of samples of each chain; see `bayes_cep.statistics.axial_statistics`).
- `rhat_summary.npy`: a between/within estimate $\sqrt{1 + B/W}$ per vertex from the axial means and
  variances of the chains, with $B$ the variance of the chain means about the pooled mean and $W$
  the mean axial variance. It needs no samples and approximates the split R-hat for concentrated
  posteriors.
- `effective_sample_size_sum.npy`: the sum of the effective sample sizes of the chains (from the
  runs); a pooled ESS if the chains agree, i.e. the R-hat is close to 1.
- with `--samples`, which streams the Zarr stores of the chains: `rhat_split.npy` (the split R-hat
  of $s = \sin(2(\theta - \mu))$ with $\mu$ the pooled mean, from the first and the last
  `min length / 2` samples of every chain), `interval_lower.npy` and `interval_upper.npy` (the
  equal-tailed credible interval of the angle at the level `--level`, from every `--thinning`-th
  sample, about the pooled mean on the branch $(-\pi/2, \pi/2]$ of the deviation).
- with `--ground-truth`, the `.npy` of a ground truth angle field per vertex: the error metrics of
  the pooled axial mean (`bayes_cep.statistics.axial_error_metrics`; per vertex, not per simplex)
  and the fraction of vertices whose interval contains the truth (with `--samples`).
- `pooled_summary.json`: the number of chains and samples, the quantiles of the R-hats and ESS and
  the metrics.

Only runs in the state `done` are used. Chains may differ in length and burn-in. The posterior of
one chain is a sample of the same distribution only if the chains agree: look at the R-hat first.

Usage: python scripts/pool_mcmc_chains.py --output DIR [--samples] [--ground-truth FILE]
           [--level 0.9] [--thinning 10] RUN_DIR [RUN_DIR ...]
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.mcmc.builder import SAMPLE_STORE_NAME
from bayes_cep.statistics.axial_error_metrics import compute_axial_error_metrics
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff

ESS_FILE = "effective_sample_size.npy"
MEAN_FILE = "posterior_axial_mean.npy"
VARIANCE_FILE = "posterior_axial_variance.npy"
BLOCK_SIZE = 200
RHAT_THRESHOLDS = (1.01, 1.05, 1.1)


# ==================================================================================================
@dataclass(frozen=True)
class ChainSummary:
    """What a finished run records about its chain.

    Attributes:
        run_dir (Path): The run directory.
        num_kept (int): Number of samples after the burn-in.
        burn_in (int): Number of samples discarded.
        axial_mean (np.ndarray): Axial mean per vertex, shape `(V,)`.
        axial_variance (np.ndarray): Axial variance per vertex, shape `(V,)`.
        effective_sample_size (np.ndarray): Effective sample size per vertex, shape `(V,)`.
    """

    run_dir: Path
    num_kept: int
    burn_in: int
    axial_mean: np.ndarray
    axial_variance: np.ndarray
    effective_sample_size: np.ndarray

    @property
    def store_path(self) -> Path:
        """The Zarr store of the chain."""
        return self.run_dir / "results" / SAMPLE_STORE_NAME


# ==================================================================================================
def read_chain_summary(run_dir: Path) -> ChainSummary | None:
    """The summary of a run, or `None` if it is not in the state `done`."""
    status = json.loads((run_dir / "status.json").read_text())
    if status["state"] != "done":
        return None
    config = json.loads((run_dir / "config.json").read_text())
    burn_in = int(config["diagnostics"]["burn_in"])
    results = run_dir / "results"
    return ChainSummary(
        run_dir=run_dir,
        num_kept=int(config["num_samples"]) - burn_in,
        burn_in=burn_in,
        axial_mean=np.load(results / MEAN_FILE),
        axial_variance=np.load(results / VARIANCE_FILE),
        effective_sample_size=np.load(results / ESS_FILE),
    )


# ==================================================================================================
def wrap_deviation(angle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    r"""The axial deviation $\frac12\arg e^{2i(\theta - \mu)}\in(-\pi/2, \pi/2]$."""
    return 0.5 * np.angle(np.exp(2j * (angle - reference)))


# ==================================================================================================
def pool_summaries(
    chains: list[ChainSummary],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Pool the axial means and variances of chains.

    The mean resultant of the doubled angles of a chain is $R e^{2i\mu}$ with $R = e^{-2\sigma^2}$
    (the axial variance is $\sigma^2 = -\frac12\ln R$); the pooled one is the mean of these,
    weighted by the number of samples.

    Args:
        chains (list[ChainSummary]): The chains, at least one.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The pooled axial mean and variance,
            the between/within R-hat estimate $\sqrt{1 + B/W}$ and the sum of the effective sample
            sizes, each of shape `(V,)`.
    """
    weights = np.array([chain.num_kept for chain in chains], dtype=float)
    weights /= weights.sum()
    resultants = np.stack(
        [np.exp(-2.0 * chain.axial_variance) * np.exp(2j * chain.axial_mean) for chain in chains]
    )
    pooled = np.tensordot(weights, resultants, axes=1)
    pooled_mean = 0.5 * np.angle(pooled)
    pooled_variance = -0.5 * np.log(np.maximum(np.abs(pooled), 1e-300))
    deviations = np.stack([wrap_deviation(chain.axial_mean, pooled_mean) for chain in chains])
    variances = np.stack([chain.axial_variance for chain in chains])
    if len(chains) > 1:
        between = np.tensordot(weights, deviations**2, axes=1)
        within = np.tensordot(weights, variances, axes=1)
        rhat = np.sqrt(1.0 + between / np.maximum(within, 1e-12))
    else:
        rhat = np.full(pooled_mean.shape, np.nan)
    effective = np.sum([chain.effective_sample_size for chain in chains], axis=0)
    return pooled_mean, pooled_variance, rhat, effective


# ==================================================================================================
def stream_chains(
    chains: list[ChainSummary],
    pooled_mean: np.ndarray,
    thinning: int,
    level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Split R-hat and credible interval from the samples of the chains.

    $s = \sin(2(\theta - \mu))$ with $\mu$ the pooled axial mean; every chain contributes its first
    and its last `min length // 2` samples (after the burn-in) as two chains to the R-hat of
    Gelman and Rubin. The interval is the equal-tailed quantile interval of the axial deviation
    $\theta - \mu$ of every `thinning`-th sample of all chains, added to $\mu$.

    Args:
        chains (list[ChainSummary]): The chains, at least one.
        pooled_mean (np.ndarray): Pooled axial mean, shape `(V,)`.
        thinning (int): Every `thinning`-th sample enters the interval.
        level (float): Credible level in $(0, 1)$.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The split R-hat and the lower and upper end of
            the interval, each of shape `(V,)`.
    """
    half = min(chain.num_kept for chain in chains) // 2
    num_vertices = pooled_mean.size
    sums = np.zeros((len(chains), 2, num_vertices))
    squares = np.zeros((len(chains), 2, num_vertices))
    thinned = []
    for position, chain in enumerate(chains):
        store = open_zarr_samples(chain.store_path)
        kept = store.shape[0] - chain.burn_in
        for start in range(0, kept, BLOCK_SIZE):
            stop = min(start + BLOCK_SIZE, kept)
            block = np.asarray(store[chain.burn_in + start : chain.burn_in + stop])
            deviation = wrap_deviation(block, pooled_mean)
            sine = np.sin(2.0 * deviation)
            rows = np.arange(start, stop)
            for part, selected in enumerate(
                (rows < half, rows >= kept - half)
            ):  # first and last half samples
                if selected.any():
                    sums[position, part] += sine[selected].sum(axis=0)
                    squares[position, part] += (sine[selected] ** 2).sum(axis=0)
            keep = (rows % thinning) == 0
            thinned.append(deviation[keep].astype(np.float32))
    means = sums / half
    variances = (squares - half * means**2) / (half - 1)
    means = means.reshape(-1, num_vertices)
    within = variances.reshape(-1, num_vertices).mean(axis=0)
    between = half * means.var(axis=0, ddof=1)
    rhat = np.sqrt(((half - 1) / half * within + between / half) / np.maximum(within, 1e-300))
    deviations = np.concatenate(thinned)
    tail = 0.5 * (1.0 - level)
    lower, upper = np.quantile(deviations, [tail, 1.0 - tail], axis=0)
    return rhat, pooled_mean + lower, pooled_mean + upper


# ==================================================================================================
def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {
        "min": float(finite.min()),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


# ==================================================================================================
def main() -> None:
    """Pool the chains of the given runs."""
    arguments = _parse_arguments()
    summaries = [read_chain_summary(Path(path)) for path in arguments.run_dirs]
    chains = [summary for summary in summaries if summary is not None]
    skipped = len(summaries) - len(chains)
    if not chains:
        raise SystemExit("No run in the state `done`.")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    mean, variance, rhat_summary, effective = pool_summaries(chains)
    arrays = {
        "pooled_axial_mean": mean,
        "pooled_axial_variance": variance,
        "rhat_summary": rhat_summary,
        "effective_sample_size_sum": effective,
    }
    summary: dict = {
        "num_chains": len(chains),
        "num_runs_skipped": skipped,
        "num_samples": int(sum(chain.num_kept for chain in chains)),
        "rhat_summary": _quantiles(rhat_summary),
        "effective_sample_size_sum": _quantiles(effective),
    }
    for threshold in RHAT_THRESHOLDS:
        summary[f"fraction_rhat_summary_above_{threshold}"] = float(
            np.mean(rhat_summary > threshold)
        )

    lower = upper = None
    if arguments.samples:
        rhat, lower, upper = stream_chains(chains, mean, arguments.thinning, arguments.level)
        arrays |= {"rhat_split": rhat, "interval_lower": lower, "interval_upper": upper}
        summary["rhat_split"] = _quantiles(rhat)
        for threshold in RHAT_THRESHOLDS:
            summary[f"fraction_rhat_split_above_{threshold}"] = float(np.mean(rhat > threshold))
        width = wrap_deviation(upper, lower)
        summary["interval_width_deg"] = _quantiles(np.degrees(np.abs(width)))

    if arguments.ground_truth:
        truth = np.load(arguments.ground_truth)
        difference = compute_axial_data_diff(mean, truth)
        summary["error_of_pooled_mean"] = compute_axial_error_metrics(difference)
        if lower is not None and upper is not None:
            offset = wrap_deviation(truth, mean)
            covered = (offset >= wrap_deviation(lower, mean)) & (
                offset <= wrap_deviation(upper, mean)
            )
            summary["interval_coverage_of_truth"] = float(np.mean(covered))

    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)
    (output / "pooled_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ==================================================================================================
def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", help="Run directories of the chains.")
    parser.add_argument("--output", required=True, help="Directory the results are written to.")
    parser.add_argument("--samples", action="store_true", help="Stream the chains for R-hat, CI.")
    parser.add_argument("--ground-truth", help=".npy of the ground truth angle per vertex.")
    parser.add_argument("--level", type=float, default=0.9, help="Credible level.")
    parser.add_argument("--thinning", type=int, default=10, help="Thinning of the interval.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

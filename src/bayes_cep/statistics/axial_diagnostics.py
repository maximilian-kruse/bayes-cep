r"""Convergence diagnostics of MCMC samples of fiber-angle fields.

The fiber angle is only defined modulo $\pi$ (see `bayes_cep.statistics.axial_statistics`), so the
autocorrelation of the raw angle is meaningless: a chain that hops between $\theta$ and
$\theta \pm \pi$ would look uncorrelated although it describes one orientation. As in
`bayes_cep.statistics.correlation_length`, each vertex is therefore represented by

$$
s_t = \sin\bigl(2(\theta_t - \mu)\bigr),
$$

with $\mu$ the axial mean of the chain at that vertex. $s$ is invariant under
$\theta \to \theta + \pi$, and for a concentrated posterior $s \approx 2(\theta - \mu)$, so that its
effective sample size is that of the angle itself. The effective sample size and autocorrelation
time of $s$ are estimated by `ls_bayesian.mcmc.diagnostics` (Vehtari et al., 2021; the baseline is
`arviz.ess(..., method="mean")`), streamed block by block, so that a Zarr chain on disk (see
`ls_bayesian.mcmc.storage.open_zarr_samples`) never has to fit in memory.

Functions:
    compute_axial_effective_sample_size: Effective sample size and autocorrelation time per vertex.
    compute_axial_autocorrelation: Autocorrelation function per vertex.
"""

import numpy as np
from ls_bayesian.mcmc.diagnostics import (
    DEFAULT_BLOCK_SIZE,
    EffectiveSampleSizeResult,
    compute_autocorrelation,
    estimate_effective_sample_size,
)

from bayes_cep.statistics.axial_statistics import compute_axial_mean_and_variance
from bayes_cep.statistics.sample_blocks import MappedSampleStore, SampleStore


# ==================================================================================================
def compute_axial_effective_sample_size(
    samples: SampleStore,
    max_lag: int,
    burn_in: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> EffectiveSampleSizeResult:
    r"""Compute the effective sample size and autocorrelation time of angle samples, per vertex.

    The chain, after discarding `burn_in` samples, is split into two halves and treated as two
    chains; see `ls_bayesian.mcmc.diagnostics.estimate_effective_sample_size` for the estimator and
    the meaning of `max_lag` and of the `truncated` flag. Reads the store block by block, in three
    passes over it: for the axial mean, and two for the mean and the autocovariance of $s$. The
    work is $O(N K d)$ for $N$ samples, $d$ vertices and $K$ = `max_lag`; memory is independent
    of $N$.

    Args:
        samples (SampleStore): Angle samples in radians, shape `(num_samples, num_vertices)`,
            e.g. a `numpy` or `zarr` array.
        max_lag (int): Largest lag of the autocovariance, smaller than half the number of samples
            after the burn-in. If the autocorrelation has not died out by then, the result is
            flagged as truncated; increase it.
        burn_in (int): Number of leading samples to discard. Defaults to `0`.
        block_size (int): Number of samples read at once. Defaults to `DEFAULT_BLOCK_SIZE`.

    Returns:
        EffectiveSampleSizeResult: Effective sample size, integrated autocorrelation time (both in
            numbers of samples) and truncation flag, each of shape `(num_vertices,)`. `nan` for a
            vertex whose angle is constant.

    Raises:
        ValueError: If `burn_in`, `max_lag` or `block_size` are invalid for the samples.
    """
    centered_sine = _build_centered_sine_store(samples, burn_in, block_size)
    return estimate_effective_sample_size(
        centered_sine, max_lag=max_lag, burn_in=burn_in, block_size=block_size
    )


# ==================================================================================================
def compute_axial_autocorrelation(
    samples: SampleStore,
    max_lag: int,
    burn_in: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""Compute the autocorrelation function of angle samples, per vertex.

    The ordinary autocorrelation (see `ls_bayesian.mcmc.diagnostics.compute_autocorrelation`) of
    $s = \sin(2(\theta - \mu))$ over the whole chain after the burn-in, not of the raw angle, which
    is meaningless modulo $\pi$. Reads the store block by block, in three passes: the axial mean,
    and the mean and the autocovariance of $s$.

    Args:
        samples (SampleStore): Angle samples in radians, shape `(num_samples, num_vertices)`,
            e.g. a `numpy` or `zarr` array.
        max_lag (int): Largest lag, smaller than the number of samples after the burn-in.
        burn_in (int): Number of leading samples to discard. Defaults to `0`.
        block_size (int): Number of samples read at once. Defaults to `DEFAULT_BLOCK_SIZE`.

    Returns:
        np.ndarray: Autocorrelation for the lags `0, ..., max_lag`, shape
            `(max_lag + 1, num_vertices)`; `nan` for a vertex whose angle is constant.

    Raises:
        ValueError: If `burn_in`, `max_lag` or `block_size` are invalid for the samples.
    """
    centered_sine = _build_centered_sine_store(samples, burn_in, block_size)
    return compute_autocorrelation(
        centered_sine, max_lag=max_lag, start=burn_in, block_size=block_size
    )


# ==================================================================================================
def _build_centered_sine_store(
    samples: SampleStore, burn_in: int, block_size: int
) -> MappedSampleStore:
    r"""Lazy view of the samples as $\sin(2(\theta - \mu))$, with $\mu$ the axial mean after the
    burn-in."""
    axial_mean, _ = compute_axial_mean_and_variance(samples, burn_in, block_size)
    return MappedSampleStore(samples, lambda block: np.sin(2 * (block - axial_mean)))

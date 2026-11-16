r"""Convergence diagnostics of several chains: rank-normalized split R-hat, bulk and tail ESS.

The estimators of Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021), "Rank-normalization,
folding, and localization: an improved $\hat R$ for assessing convergence of MCMC":

- `rhat`: every chain is split into two halves, which are treated as separate chains; the
  split $\hat R$ is computed from the rank-normalized draws and from the rank-normalized draws
  folded about the median ($|x - \mathrm{median}|$, which detects differences in scale and tails),
  and the larger of the two is reported.
- `ess_bulk`: effective sample size of the rank-normalized draws, which describes the efficiency of
  the posterior mean and median (and, through the ranks, of the bulk of the distribution).
- `ess_tail`: the smaller effective sample size of the indicators $1[x \le q_{0.05}]$ and
  $1[x \le q_{0.95}]$, which describes the efficiency of the 5 % and 95 % quantiles (equal-tailed
  90 % intervals).

The effective sample sizes use the autocorrelation of the split chains with Geyer's initial
monotone sequence, so they are not capped at a maximum lag, and they are the pooled values of all
chains. Everything is vectorized over a parameter axis, computed in blocks of parameters to bound
the memory.

For fiber angles (only defined modulo $\pi$) the draws have to be an unwrapped, linear quantity,
e.g. the signed axial deviation from the pooled axial mean, `compute_axial_data_diff(theta, mean)`.

Functions:
    compute_multichain_diagnostics: $\hat R$, bulk and tail ESS of draws of several chains.
"""

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

TAIL_QUANTILES = (0.05, 0.95)
DEFAULT_PARAMETER_BLOCK_SIZE = 512


# ==================================================================================================
@dataclass(frozen=True)
class MultichainDiagnostics:
    """The diagnostics per parameter.

    Attributes:
        rhat (np.ndarray): Rank-normalized split $\\hat R$ (maximum of the plain and the folded
            version), shape `(num_parameters,)`. `nan` for a constant parameter.
        ess_bulk (np.ndarray): Bulk effective sample size over all chains, same shape.
        ess_tail (np.ndarray): Tail effective sample size, same shape.
    """

    rhat: np.ndarray
    ess_bulk: np.ndarray
    ess_tail: np.ndarray


# ==================================================================================================
def compute_multichain_diagnostics(
    draws: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
    block_size: int = DEFAULT_PARAMETER_BLOCK_SIZE,
) -> MultichainDiagnostics:
    """Compute rank-normalized split $\\hat R$, bulk and tail ESS per parameter.

    Args:
        draws (np.ndarray): Draws of all chains after the burn-in and thinning, shape
            `(num_chains, num_draws, num_parameters)`. The chains must have equal length.
        block_size (int): Number of parameters processed at a time. Defaults to
            `DEFAULT_PARAMETER_BLOCK_SIZE`.

    Returns:
        MultichainDiagnostics: The diagnostics.

    Raises:
        ValueError: If the draws do not have three axes, a chain has fewer than 8 draws, or
            `block_size` is not positive.
    """
    if draws.ndim != 3:
        raise ValueError(f"Expected draws of shape (chains, draws, parameters), got {draws.shape}.")
    if draws.shape[1] < 8:
        raise ValueError(f"Each chain needs at least 8 draws, got {draws.shape[1]}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    num_parameters = draws.shape[2]
    rhat = np.empty(num_parameters)
    ess_bulk = np.empty(num_parameters)
    ess_tail = np.empty(num_parameters)
    for start in range(0, num_parameters, block_size):
        block = np.asarray(draws[:, :, start : start + block_size], dtype=np.float64)
        split = _split_chains(block)
        rhat[start : start + block_size] = _compute_rank_normalized_rhat(split)
        ess_bulk[start : start + block_size] = _compute_ess(_rank_normalize(split))
        ess_tail[start : start + block_size] = np.fmin.reduce(
            [
                _compute_ess(
                    (split <= np.quantile(split, level, axis=(0, 1), keepdims=True)).astype(float)
                )
                for level in TAIL_QUANTILES
            ]
        )
    return MultichainDiagnostics(rhat=rhat, ess_bulk=ess_bulk, ess_tail=ess_tail)


# ==================================================================================================
def _split_chains(
    draws: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """Cut every chain into two halves of equal length (an odd last draw is dropped)."""
    num_chains, num_draws, num_parameters = draws.shape
    half = num_draws // 2
    return np.concatenate([draws[:, :half], draws[:, half : 2 * half]], axis=0).reshape(
        2 * num_chains, half, num_parameters
    )


# --------------------------------------------------------------------------------------------------
def _rank_normalize(
    draws: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """Replace the draws of each parameter by normal scores of their pooled (average) ranks."""
    num_chains, num_draws, num_parameters = draws.shape
    pooled = draws.reshape(num_chains * num_draws, num_parameters)
    ranks = stats.rankdata(pooled, axis=0, method="average")
    scores = special.ndtri((ranks - 3.0 / 8.0) / (pooled.shape[0] + 1.0 / 4.0))
    return scores.reshape(draws.shape)


# --------------------------------------------------------------------------------------------------
def _compute_rank_normalized_rhat(
    split_draws: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """The larger of the $\\hat R$ of the rank-normalized and of the folded draws."""
    folded = np.abs(split_draws - np.median(split_draws, axis=(0, 1), keepdims=True))
    return np.fmax(
        _compute_rhat(_rank_normalize(split_draws)), _compute_rhat(_rank_normalize(folded))
    )


# --------------------------------------------------------------------------------------------------
def _compute_rhat(
    chains: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Plain $\\hat R$ from the between- and within-chain variance, shape `(num_parameters,)`."""
    num_draws = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean(axis=0)
    between = num_draws * chains.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(((num_draws - 1) / num_draws * within + between / num_draws) / within)


# --------------------------------------------------------------------------------------------------
def _compute_ess(
    chains: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Effective sample size of (split) chains by Geyer's initial monotone sequence.

    Follows the definition of Vehtari et al. (2021): with the within-chain variance $W$, the
    between-chain variance $B$ and the autocovariance $\\gamma_t$ averaged over the chains,
    $\\hat\\rho_t = 1 - (W - \\bar\\gamma_t) / \\hat v$, $\\hat v = \\frac{n-1}{n}W + B/n$, the
    sums of consecutive pairs of $\\hat\\rho$ are truncated at the first negative one and made
    monotone, and $\\mathrm{ESS} = mn / (-1 + 2\\sum \\text{pairs})$. `nan` for a constant
    parameter.
    """
    num_chains, num_draws, _ = chains.shape
    centered = chains - chains.mean(axis=1, keepdims=True)
    padded_length = 2 * num_draws
    spectrum = np.fft.rfft(centered, n=padded_length, axis=1)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum), n=padded_length, axis=1)[
        :, :num_draws
    ]
    autocovariance = autocovariance / num_draws  # the biased estimate of every chain
    mean_autocovariance = autocovariance.mean(axis=0)  # (draws, parameters)
    within = autocovariance[:, 0].mean(axis=0) * num_draws / (num_draws - 1)
    variance_estimate = within * (num_draws - 1) / num_draws
    if num_chains > 1:
        variance_estimate = variance_estimate + chains.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1.0 - (within[None, :] - mean_autocovariance) / variance_estimate[None, :]
    rho[0] = 1.0
    num_pairs = num_draws // 2
    pair_sums = rho[: 2 * num_pairs : 2] + rho[1 : 2 * num_pairs : 2]  # (pairs, parameters)
    negative = pair_sums < 0.0
    first_negative = np.where(negative.any(axis=0), negative.argmax(axis=0), num_pairs)
    keep = np.arange(num_pairs)[:, None] < first_negative[None, :]
    monotone = np.minimum.accumulate(np.where(keep, pair_sums, np.inf), axis=0)
    monotone = np.where(keep, monotone, 0.0)
    integrated_time = -1.0 + 2.0 * monotone.sum(axis=0)
    integrated_time = np.maximum(integrated_time, 1.0 / np.log10(num_chains * num_draws))
    ess = num_chains * num_draws / integrated_time
    ess[~np.isfinite(variance_estimate) | (variance_estimate <= 0.0)] = np.nan
    return ess

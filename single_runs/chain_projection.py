r"""Projection of a stored MCMC chain onto the eigenvectors of the Laplace approximation.

Which part of the posterior mixes slowly, and does the Laplace approximation describe it? The run
recomputes the eigenpairs $(\lambda_i, v_i)$ of $Hv = \lambda\Gamma_0^{-1}v$ at a stored MAP
estimate (as the preconditioned MCMC run, but with a rank that may exceed the one of the chain) and
projects every sample $m$ of a finished chain onto them,

$$c_i = v_i^T\Gamma_0^{-1}(m - m_\mathrm{MAP}), \qquad
r = (m - m_\mathrm{MAP})^T\Gamma_0^{-1}(m - m_\mathrm{MAP}) - \sum_i c_i^2.$$

The $v_i$ are $\Gamma_0^{-1}$-orthonormal, so under the Laplace approximation the $c_i$ are
independent with mean 0 and variance $1/(1+\lambda_i)$, and $r\ge 0$ is the squared norm of the part
outside the retained subspace (prior variance, if the approximation is right there). The run writes,
per coefficient, the mean, variance and integrated autocorrelation time of the chain (initial
monotone sequence estimator on the whole chain, no lag cap), and the same for $r$.

Classes:
    ChainProjectionRunConfig: The chain, the posterior it samples and the eigenproblem.
    ChainProjectionResults: The arrays the run writes.
    ChainProjectionRun: The projection run.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.laplace.builder import LaplaceBuilder, LaplaceSettings
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components


# ==================================================================================================
@dataclass(frozen=True)
class ChainProjectionRunConfig(RunConfig):
    """The chain, the posterior it samples and the eigenproblem it is projected onto.

    Attributes:
        problem (InferenceProblemConfig): The posterior the chain samples.
        map_estimate_path (Path): `.npy` file with the MAP estimate the approximation of the chain
            was centered at.
        laplace (LaplaceSettings): Eigensolver and rank of the eigenproblem. A rank above the one
            of the chain also shows directions the preconditioner did not correct.
        chain_path (Path): The Zarr store (`samples.zarr`) of a finished MCMC run of `problem`.
        burn_in (int): Number of samples discarded from the start of the chain.
        block_size (int): Number of samples projected at a time.
        thinning (int): Every `thinning`-th sample after the burn-in is kept in
            `coefficient_chain`.
        acf_lags (tuple[int, ...]): Lags of the stored autocorrelation of the coefficients.
    """

    problem: InferenceProblemConfig
    map_estimate_path: Path
    laplace: LaplaceSettings
    chain_path: Path
    burn_in: int
    block_size: int
    thinning: int
    acf_lags: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.burn_in < 0 or self.block_size < 1 or self.thinning < 1:
            raise ValueError(
                "burn_in must be non-negative, block_size and thinning positive, got "
                f"{self.burn_in}, {self.block_size} and {self.thinning}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class ChainProjectionResults(ArrayRecord):
    r"""The arrays a projection run writes; $K$ is the rank, $N$ the number of samples after the
    burn-in.

    Attributes:
        eigenvalues (np.ndarray): Eigenvalues $\lambda_i$, descending, shape `(K,)`.
        coefficient_mean (np.ndarray): Chain mean of $c_i$, shape `(K,)`; 0 under the Laplace
            approximation.
        coefficient_variance (np.ndarray): Chain variance of $c_i$, shape `(K,)`; compare with
            $1/(1+\lambda_i)$.
        coefficient_autocorrelation_time (np.ndarray): Integrated autocorrelation time of $c_i$ in
            samples, shape `(K,)`.
        coefficient_autocorrelation (np.ndarray): Autocorrelation of $c_i$ at `acf_lags`, shape
            `(len(acf_lags), K)`.
        coefficient_chain (np.ndarray): Every `thinning`-th $c$ after the burn-in, float32, shape
            `(N // thinning, K)`.
        residual_chain (np.ndarray): The same samples of $r$, shape `(N // thinning,)`.
        residual_statistics (np.ndarray): Chain mean, variance and autocorrelation time of $r$,
            shape `(3,)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "eigenvalues": "hessian_eigenvalues.npy",
        "coefficient_mean": "coefficient_mean.npy",
        "coefficient_variance": "coefficient_variance.npy",
        "coefficient_autocorrelation_time": "coefficient_autocorrelation_time.npy",
        "coefficient_autocorrelation": "coefficient_autocorrelation.npy",
        "coefficient_chain": "coefficient_chain.npy",
        "residual_chain": "residual_chain.npy",
        "residual_statistics": "residual_statistics.npy",
    }

    eigenvalues: np.ndarray
    coefficient_mean: np.ndarray
    coefficient_variance: np.ndarray
    coefficient_autocorrelation_time: np.ndarray
    coefficient_autocorrelation: np.ndarray
    coefficient_chain: np.ndarray
    residual_chain: np.ndarray
    residual_statistics: np.ndarray


_FILES = ChainProjectionResults.FILE_NAMES  # the names of the result files, defined once there


# ==================================================================================================
def compute_autocorrelation_time(
    series: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    lags: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Integrated autocorrelation time and autocorrelation at `lags` of every column.

    The time is $\tau = -1 + 2\sum_{k\ge 0}\Gamma_k$ over the pairs $\Gamma_k = \rho_{2k} +
    \rho_{2k+1}$ of the autocorrelation, cut at the first negative pair and made monotone
    non-increasing (Geyer's initial monotone sequence), so there is no maximal lag. A column of
    a chain that does not move has no autocorrelation; its entries are NaN.

    Args:
        series (np.ndarray): Chain, shape `(N, K)`.
        lags (tuple[int, ...]): Lags of the returned autocorrelation, each below `N`.

    Returns:
        tuple[np.ndarray, np.ndarray]: The times, shape `(K,)`, and the autocorrelation at `lags`,
            shape `(len(lags), K)`.
    """
    num_samples, num_columns = series.shape
    centered = series - series.mean(axis=0)
    fft_size = 1 << (2 * num_samples - 1).bit_length()
    times = np.full(num_columns, np.nan)
    autocorrelation = np.full((len(lags), num_columns), np.nan)
    for start in range(0, num_columns, 50):
        block = centered[:, start : start + 50]
        spectrum = np.fft.rfft(block, fft_size, axis=0)
        covariance = np.fft.irfft(spectrum * spectrum.conj(), fft_size, axis=0)[:num_samples]
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = covariance / covariance[0]
        for column in range(block.shape[1]):
            rho = correlation[:, column]
            if not np.isfinite(rho).all():
                continue
            pairs = rho[: 2 * (num_samples // 2)].reshape(-1, 2).sum(axis=1)
            negative = np.flatnonzero(pairs < 0)
            pairs = pairs[: negative[0] if negative.size else pairs.size]
            pairs = np.minimum.accumulate(pairs)
            times[start + column] = -1.0 + 2.0 * pairs.sum()
            autocorrelation[:, start + column] = rho[list(lags)]
    return times, autocorrelation


# ==================================================================================================
class ChainProjectionRun(Run[ChainProjectionRunConfig]):
    """Project a stored chain onto the eigenvectors of the Laplace approximation."""

    config_type = ChainProjectionRunConfig
    name = "chain-projection"
    outputs = {
        f"results/{_FILES['eigenvalues']}": "Eigenvalues of the misfit Hessian relative to the "
        "prior precision, descending.",
        f"results/{_FILES['coefficient_mean']}": "Chain mean of the coefficients.",
        f"results/{_FILES['coefficient_variance']}": "Chain variance of the coefficients.",
        f"results/{_FILES['coefficient_autocorrelation_time']}": "Integrated autocorrelation time "
        "of the coefficients, in samples.",
        f"results/{_FILES['coefficient_autocorrelation']}": "Autocorrelation of the coefficients "
        "at the configured lags.",
        f"results/{_FILES['coefficient_chain']}": "Thinned chain of the coefficients.",
        f"results/{_FILES['residual_chain']}": "Thinned chain of the squared norm outside the "
        "retained subspace.",
        f"results/{_FILES['residual_statistics']}": "Mean, variance and autocorrelation time of "
        "that squared norm.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        # The chain is a directory (a Zarr store) and not hashed.
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.map_estimate_path),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed across the runs of a study."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        prior = assembled.components.prior
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))

        with steps.step(f"Computing {config.laplace.num_eigenvalues} eigenpairs"):
            approximation = LaplaceBuilder(config.laplace, assembled.components).build(map_estimate)
        eigenvalues = approximation.eigenpairs.eigenvalues
        eigenvectors = approximation.eigenpairs.eigenvectors
        logger.info(f"      {describe_array('eigenvalues', eigenvalues)}")

        with steps.step("Applying the prior precision to the eigenvectors"):
            weighted_vectors = np.stack(
                [prior.apply_precision_operator(vector) for vector in eigenvectors.T], axis=1
            )

        chain = open_zarr_samples(resolve_repository_path(config.chain_path))
        num_samples, num_vertices = chain.shape
        if num_vertices != map_estimate.size:
            raise ValueError(
                f"The chain has {num_vertices} vertices, the MAP estimate {map_estimate.size}."
            )
        num_kept = num_samples - config.burn_in
        coefficients = np.empty((num_kept, eigenvalues.size))
        residual = np.empty(num_kept)
        start = time.perf_counter()
        with steps.step(f"Projecting {num_kept} samples"):
            for first in range(config.burn_in, num_samples, config.block_size):
                last = min(first + config.block_size, num_samples)
                offsets = np.asarray(chain[first:last]) - map_estimate
                rows = slice(first - config.burn_in, last - config.burn_in)
                coefficients[rows] = offsets @ weighted_vectors
                norms = np.array(
                    [offset @ prior.apply_precision_operator(offset) for offset in offsets]
                )
                residual[rows] = norms - np.sum(coefficients[rows] ** 2, axis=1)
        seconds_projection = time.perf_counter() - start

        with steps.step("Computing the autocorrelation times"):
            times, autocorrelation = compute_autocorrelation_time(coefficients, config.acf_lags)
            residual_time, _ = compute_autocorrelation_time(residual[:, None], config.acf_lags[:1])
        residual_statistics = np.array([residual.mean(), residual.var(ddof=1), residual_time[0]])
        logger.info(f"      {describe_array('coefficient autocorrelation time', times)}")
        logger.info(f"      residual mean, variance, time: {residual_statistics}")

        ChainProjectionResults(
            eigenvalues=eigenvalues,
            coefficient_mean=coefficients.mean(axis=0),
            coefficient_variance=coefficients.var(axis=0, ddof=1),
            coefficient_autocorrelation_time=times,
            coefficient_autocorrelation=autocorrelation,
            coefficient_chain=coefficients[:: config.thinning].astype(np.float32),
            residual_chain=residual[:: config.thinning],
            residual_statistics=residual_statistics,
        ).save(self.results_dir(run_dir))
        return {
            "num_eigenpairs": int(eigenvalues.size),
            "num_samples": int(num_kept),
            "seconds_projection": seconds_projection,
            "coefficient_time_max": float(np.nanmax(times)),
            "coefficient_time_median": float(np.nanmedian(times)),
            "residual_mean": float(residual_statistics[0]),
            "residual_time": float(residual_statistics[2]),
        }

r"""Covariance of the Laplace coefficients from the Gauss-Newton Hessian averaged over a chain.

DILI (Cui, Law and Marzouk, 2016) builds its likelihood-informed subspace from the Gauss-Newton
Hessian averaged over the posterior, not from the empirical covariance of a pilot chain: the
Hessian average needs no effective sample size in the stiff directions, and it follows the
rotation of the eigenvectors along the chain. Does it describe the chain better than the Laplace
covariance, and better than the calibration of the pilot chains?

The run computes the eigenpairs $(\lambda_i, v_i)$ of $Hv = \lambda\Gamma_0^{-1}v$ at the MAP
estimate. In the coefficients $c = V^T\Gamma_0^{-1}(u - m)$ the prior precision is the identity,
and the Gauss-Newton Hessian $H(u_s)$ at a state $u_s$ of the chain is $K_s = V^TH(u_s)V$ (one
product per eigenvector and state, about a millisecond each). With $\bar K$ the mean of $K_s$ over
the states, the candidates for the covariance of the leading $k$ coefficients are the leading
blocks of

- the inverse of the averaged precision, $(I + \bar K)^{-1}$,
- the average of the inverses, $\frac1S\sum_s(I + K_s)^{-1}$,

both marginals of the full $K\times K$ matrix (the coupling to the other coefficients is kept).
Each is scored on every chain of the pilot set like the candidates of
[`CalibrationValidationRun`][single_runs.calibration_validation.CalibrationValidationRun], next to
the Laplace covariance and the empirical covariance of the pooled chains (not held out: a ceiling
for a Gaussian fitted to these chains). The Gaussian is centered at the pooled mean of the chains.

Classes:
    HessianAverageRunConfig: The posterior, the chain whose states are used and the scored chains.
    HessianAverageResults: The arrays the run writes.
    HessianAverageRun: The Hessian average run.
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
from bayes_cep.laplace.hessian import GaussNewtonMisfitHessian
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.calibration_validation import SPECTRUM_QUANTILES, score_covariance
from single_runs.chain_projection import ChainProjectionResults
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components

CANDIDATE_NAMES = ("laplace", "averaged_precision", "averaged_covariance", "empirical")


# ==================================================================================================
@dataclass(frozen=True)
class HessianAverageRunConfig(RunConfig):
    """The posterior, the chain whose states are used and the chains the covariances are scored on.

    Attributes:
        problem (InferenceProblemConfig): The posterior the chains sample.
        map_estimate_path (Path): `.npy` file with the MAP estimate the Laplace approximation of
            the chains was centered at.
        laplace (LaplaceSettings): Eigensolver and rank of the eigenproblem at the MAP estimate;
            it must give the eigenvectors the scored chains were projected onto.
        chain_path (Path): The Zarr store (`samples.zarr`) of a finished MCMC run of `problem`;
            its states carry the Hessians.
        burn_in (int): Number of samples discarded from the start of that chain.
        num_states (int): Number of states, evenly spaced over the chain after the burn-in.
        num_coefficients (tuple[int, ...]): Numbers $k$ of leading coefficients that are scored.
        projection_paths (tuple[Path, ...]): The `results` directories of the projection runs of
            the chains that are scored (all chains of the pilot set).
    """

    problem: InferenceProblemConfig
    map_estimate_path: Path
    laplace: LaplaceSettings
    chain_path: Path
    burn_in: int
    num_states: int
    num_coefficients: tuple[int, ...]
    projection_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.burn_in < 0 or self.num_states < 1:
            raise ValueError(
                f"burn_in must be non-negative and num_states positive, got {self.burn_in} "
                f"and {self.num_states}."
            )
        if len(self.num_coefficients) == 0 or min(self.num_coefficients) < 2:
            raise ValueError(
                f"num_coefficients must be non-empty with entries of at least 2, got "
                f"{self.num_coefficients}."
            )
        if len(self.projection_paths) < 1:
            raise ValueError("At least one projected chain is needed.")


# ==================================================================================================
@dataclass(frozen=True)
class HessianAverageResults(ArrayRecord):
    """The arrays a Hessian average run writes; $K$ is the rank, $S$ the number of states, $k$ the
    largest number of scored coefficients, $F$ the number of scored chains, $N$ the number of
    `num_coefficients`, $Q$ the number of spectrum quantiles.

    Attributes:
        eigenvalues (np.ndarray): Eigenvalues of the Laplace approximation, shape `(K,)`.
        state_indices (np.ndarray): Sample index of each state, shape `(S,)`.
        averaged_hessian (np.ndarray): $\\bar K = \\frac1S\\sum_s V^TH(u_s)V$, shape `(K, K)`.
        state_diagonal (np.ndarray): Diagonal of $V^TH(u_s)V$ at each state, shape `(S, K)`.
        averaged_precision_covariance (np.ndarray): Leading $k\\times k$ block of $(I +
            \\bar K)^{-1}$, shape `(k, k)`.
        averaged_covariance (np.ndarray): Leading block of the average of $(I + K_s)^{-1}$,
            shape `(k, k)`.
        log_likelihood (np.ndarray): Mean log density of each scored chain under each candidate
            (`CANDIDATE_NAMES`), shape `(N, 4, F)`.
        spectrum (np.ndarray): Quantiles of the whitened spectrum of each chain, shape
            `(N, 4, F, Q)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "eigenvalues": "hessian_eigenvalues.npy",
        "state_indices": "state_indices.npy",
        "averaged_hessian": "averaged_hessian.npy",
        "state_diagonal": "state_diagonal.npy",
        "averaged_precision_covariance": "averaged_precision_covariance.npy",
        "averaged_covariance": "averaged_covariance.npy",
        "log_likelihood": "log_likelihood.npy",
        "spectrum": "spectrum.npy",
    }

    eigenvalues: np.ndarray
    state_indices: np.ndarray
    averaged_hessian: np.ndarray
    state_diagonal: np.ndarray
    averaged_precision_covariance: np.ndarray
    averaged_covariance: np.ndarray
    log_likelihood: np.ndarray
    spectrum: np.ndarray


_FILES = HessianAverageResults.FILE_NAMES  # the names of the result files


# ==================================================================================================
class HessianAverageRun(Run[HessianAverageRunConfig]):
    """Average the Gauss-Newton Hessian over chain states and score the resulting covariance."""

    config_type = HessianAverageRunConfig
    name = "hessian-average"
    outputs = {
        f"results/{_FILES['eigenvalues']}": "Eigenvalues of the Laplace approximation.",
        f"results/{_FILES['state_indices']}": "Sample index of each state.",
        f"results/{_FILES['averaged_hessian']}": "Gauss-Newton Hessian in the coefficients, "
        "averaged over the states.",
        f"results/{_FILES['state_diagonal']}": "Its diagonal at each state.",
        f"results/{_FILES['averaged_precision_covariance']}": "Leading block of the inverse of "
        "the averaged precision.",
        f"results/{_FILES['averaged_covariance']}": "Leading block of the average of the inverses.",
        f"results/{_FILES['log_likelihood']}": "Mean log density of the scored chains under "
        "each candidate.",
        f"results/{_FILES['spectrum']}": "Quantiles of the whitened spectrum of the scored chains.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        # The chain is a directory (a Zarr store) and not hashed.
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.map_estimate_path),
            *(
                resolve_repository_path(path) / ChainProjectionResults.FILE_NAMES[name]
                for path in self.config.projection_paths
                for name in ("eigenvalues", "coefficient_chain")
            ),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed in a notebook."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        names = ChainProjectionResults.FILE_NAMES
        assembled = assemble_posterior_components(config.problem, steps, logger)
        components = assembled.components
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))

        with steps.step(f"Eigenproblem of rank {config.laplace.num_eigenvalues} at the MAP"):
            pairs = LaplaceBuilder(config.laplace, components).build(map_estimate).eigenpairs
        eigenvalues, eigenvectors = pairs.eigenvalues, pairs.eigenvectors
        rank = eigenvalues.size
        logger.info(f"      {describe_array('eigenvalues', eigenvalues)}")
        if max(config.num_coefficients) > rank:
            raise ValueError(
                f"num_coefficients up to {max(config.num_coefficients)} exceed the {rank} "
                "eigenpairs."
            )

        chain = open_zarr_samples(resolve_repository_path(config.chain_path))
        indices = np.linspace(config.burn_in, chain.shape[0] - 1, config.num_states).astype(int)
        identity = np.eye(rank)
        hessian_sum = np.zeros((rank, rank))
        covariance_sum = np.zeros((rank, rank))
        state_diagonal = np.empty((indices.size, rank))
        for position, index in enumerate(indices):
            start = time.perf_counter()
            hessian = GaussNewtonMisfitHessian(
                components.forward_map, components.likelihood, np.asarray(chain[int(index)])
            )
            with steps.step(f"State {position + 1} of {indices.size} (sample {index})"):
                products = np.stack([hessian.apply(vector) for vector in eigenvectors.T], axis=1)
                projected = eigenvectors.T @ products
                projected = 0.5 * (projected + projected.T)
            hessian_sum += projected
            covariance_sum += np.linalg.inv(identity + projected)
            state_diagonal[position] = np.diag(projected)
            logger.info(f"      {time.perf_counter() - start:.1f} s")
        averaged_hessian = hessian_sum / indices.size
        averaged_covariance_full = covariance_sum / indices.size
        averaged_precision_covariance_full = np.linalg.inv(identity + averaged_hessian)

        k_max = max(config.num_coefficients)
        with steps.step(f"Scoring the covariances on {len(config.projection_paths)} chains"):
            directories = [resolve_repository_path(path) for path in config.projection_paths]
            chains = []
            for directory in directories:
                if not np.array_equal(np.load(directory / names["eigenvalues"]), eigenvalues):
                    raise ValueError(
                        f"The eigenvalues in {directory} differ from those of the eigenproblem "
                        "of this run: the chains are not projected onto the same eigenvectors."
                    )
                chains.append(np.load(directory / names["coefficient_chain"]).astype(np.float64))
            pooled = np.concatenate(chains)
            log_likelihood = np.empty(
                (len(config.num_coefficients), len(CANDIDATE_NAMES), len(chains))
            )
            spectrum = np.empty(
                (
                    len(config.num_coefficients),
                    len(CANDIDATE_NAMES),
                    len(chains),
                    len(SPECTRUM_QUANTILES),
                )
            )
            for row, k in enumerate(config.num_coefficients):
                mean = pooled[:, :k].mean(axis=0)
                covariances = (
                    np.diag(1.0 / (1.0 + eigenvalues[:k])),
                    averaged_precision_covariance_full[:k, :k],
                    averaged_covariance_full[:k, :k],
                    np.cov(pooled[:, :k], rowvar=False),
                )
                for column, covariance in enumerate(covariances):
                    for held, samples in enumerate(chains):
                        log_likelihood[row, column, held], _, spectrum[row, column, held] = (
                            score_covariance(samples[:, :k], mean, covariance)
                        )

        HessianAverageResults(
            eigenvalues=eigenvalues,
            state_indices=indices,
            averaged_hessian=averaged_hessian,
            state_diagonal=state_diagonal,
            averaged_precision_covariance=averaged_precision_covariance_full[:k_max, :k_max],
            averaged_covariance=averaged_covariance_full[:k_max, :k_max],
            log_likelihood=log_likelihood,
            spectrum=spectrum,
        ).save(self.results_dir(run_dir))

        metrics: Metrics = {
            "num_eigenpairs": int(rank),
            "num_states": int(indices.size),
            "state_diagonal_ratio_median": float(
                np.median(state_diagonal.mean(axis=0)[:k_max] / eigenvalues[:k_max])
            ),
        }
        mean_score = log_likelihood.mean(axis=2)
        for row, k in enumerate(config.num_coefficients):
            for column, name in enumerate(CANDIDATE_NAMES):
                metrics[f"k{k}_{name}_log_likelihood"] = float(mean_score[row, column])
        return metrics

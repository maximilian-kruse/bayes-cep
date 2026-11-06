r"""Leave-one-chain-out validation of the covariance calibration of a Laplace approximation.

The calibration ([`calibrate_approximation`][bayes_cep.laplace.calibration.calibrate_approximation])
replaces the covariance of the leading coefficients of the Laplace eigenvectors by a regularized
empirical covariance of pilot chains, with settings (coupling shrinkage, variance inflation, lower
bound) that are guesses. The stiffest coefficients have an effective sample size of about 10 per
chain, so the empirical covariance is noisy there. This run scores candidates on chains that were
not used to estimate them:

- Each of the $F$ chains is held out in turn; the covariance $\Sigma$ of a candidate is estimated
  from the other $F - 1$ chains (pooled, as the calibration does) and the held-out chain is scored
  under the Gaussian $\mathcal N(\mu, \Sigma)$, with $\mu$ the pooled mean of the training chains.
  The score is the mean log density of the held-out samples; the mean shift of the held-out chain
  is part of it.
- The whitened spectrum is the eigenvalues of $\Sigma^{-1/2} S_h \Sigma^{-1/2}$ with $S_h$ the
  covariance of the held-out chain about its own mean: all 1 for a covariance that describes the
  held-out chain, above 1 where the candidate is too narrow, below 1 where it is too wide. Its
  quantiles are stored.
- The candidates are the Laplace covariance (baseline), the unregularized empirical covariance, the
  regularization of the calibration on a grid, and a low-rank update: the whitened empirical
  covariance keeps only its $r$ largest eigenvalues above 1 (each raised to $1 + a(\mu - 1)$ for an
  inflation $a$) and leaves the Laplace covariance in the other directions, so far fewer numbers
  are estimated than for the full covariance.

The coefficients are those of the projection of the chains onto the Laplace eigenvectors
([`ChainProjectionRun`][single_runs.chain_projection.ChainProjectionRun]), thinned and after the
burn-in; the chains must have been projected onto the same eigenvectors. No forward solves are
needed, the run takes seconds.

Classes:
    CalibrationValidationRunConfig: The projected chains and the grid of candidates.
    TargetCalibrationValidationRunConfig: The same with a shrinkage target.
    CalibrationValidationResults: The arrays the run writes.
    CalibrationValidationRun: The validation run.
    TargetCalibrationValidationRun: The validation run with a shrinkage target.

Functions:
    low_rank_update_covariance: The low-rank candidate.
    score_covariance: Log density and whitened spectrum of samples under a Gaussian.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from pathlib import Path
from typing import ClassVar, cast, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from scipy.linalg import solve_triangular

from bayes_cep.laplace.calibration import regularize_coefficient_covariance
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from single_runs.chain_projection import ChainProjectionResults

SPECTRUM_QUANTILES = (0.0, 0.1, 0.5, 0.9, 1.0)
_TARGET_FILE_NAME = "averaged_covariance.npy"  # in the results of a `HessianAverageRun`
_JITTER = 1e-12  # relative to the mean variance, against a covariance that is not definite


# ==================================================================================================
class CandidateKind(IntEnum):
    """The kinds of covariance that are scored."""

    LAPLACE = 0
    EMPIRICAL = 1
    REGULARIZED = 2
    LOW_RANK = 3
    TARGET = 4


# ==================================================================================================
@dataclass(frozen=True)
class CalibrationValidationRunConfig(RunConfig):
    """The projected chains and the grid of candidates.

    Attributes:
        projection_paths (tuple[Path, ...]): The `results` directories of the projection runs, one
            per chain (at least three). Each holds `hessian_eigenvalues.npy` and
            `coefficient_chain.npy`; a relative path is relative to the repository. The
            eigenvalues must agree, i.e. the chains are projected onto the same eigenvectors.
        num_coefficients (tuple[int, ...]): Numbers $k$ of leading coefficients that are
            calibrated.
        coupling_shrinkages (tuple[float, ...]): Coupling shrinkages of the regularized candidates.
        variance_inflations (tuple[float, ...]): Variance inflations of the regularized candidates
            and inflations $a$ of the excess of the low-rank candidates.
        minimum_variance_ratios (tuple[float, ...]): Lower bounds of the regularized candidates,
            in units of the Laplace covariance.
        ranks (tuple[int, ...]): Ranks $r$ of the low-rank candidates; none if empty.
        reference_shrinkage (float): Coupling shrinkage of the setting that is in use, for the
            comparison in the metrics.
        reference_inflation (float): Its variance inflation.
        reference_minimum_ratio (float): Its lower bound.
    """

    projection_paths: tuple[Path, ...]
    num_coefficients: tuple[int, ...]
    coupling_shrinkages: tuple[float, ...]
    variance_inflations: tuple[float, ...]
    minimum_variance_ratios: tuple[float, ...]
    ranks: tuple[int, ...]
    reference_shrinkage: float
    reference_inflation: float
    reference_minimum_ratio: float

    def __post_init__(self) -> None:
        if len(self.projection_paths) < 3:
            raise ValueError(f"At least three chains are needed, got {len(self.projection_paths)}.")
        if len(self.num_coefficients) == 0 or min(self.num_coefficients) < 2:
            raise ValueError(
                f"num_coefficients must be non-empty with entries of at least 2, got "
                f"{self.num_coefficients}."
            )
        if min(self.ranks, default=1) < 1:
            raise ValueError(f"ranks must be positive, got {self.ranks}.")
        if len(self.coupling_shrinkages) * len(self.variance_inflations) == 0:
            raise ValueError("coupling_shrinkages and variance_inflations must not be empty.")


# ==================================================================================================
@dataclass(frozen=True)
class CalibrationValidationResults(ArrayRecord):
    """The arrays a validation run writes; $C$ is the number of candidates, $F$ the number of
    chains, $Q$ the number of spectrum quantiles.

    Attributes:
        candidate_kinds (np.ndarray): `CandidateKind` of each candidate, shape `(C,)`.
        candidate_parameters (np.ndarray): Columns `num_coefficients`, `coupling_shrinkage`,
            `variance_inflation`, `minimum_variance_ratio`, `rank` (0 where not used), shape
            `(C, 5)`.
        log_likelihood (np.ndarray): Mean log density of the held-out chain under the candidate
            estimated from the others, shape `(C, F)`.
        mean_shift (np.ndarray): Squared Mahalanobis distance of the held-out chain mean from the
            training mean, in the candidate covariance, shape `(C, F)`.
        spectrum (np.ndarray): Quantiles (`SPECTRUM_QUANTILES`) of the whitened spectrum of the
            held-out covariance, shape `(C, F, Q)`.
        eigenvalues (np.ndarray): Eigenvalues of the Laplace approximation, shape `(K,)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "candidate_kinds": "candidate_kinds.npy",
        "candidate_parameters": "candidate_parameters.npy",
        "log_likelihood": "log_likelihood.npy",
        "mean_shift": "mean_shift.npy",
        "spectrum": "spectrum.npy",
        "eigenvalues": "hessian_eigenvalues.npy",
    }

    candidate_kinds: np.ndarray
    candidate_parameters: np.ndarray
    log_likelihood: np.ndarray
    mean_shift: np.ndarray
    spectrum: np.ndarray
    eigenvalues: np.ndarray


_FILES = CalibrationValidationResults.FILE_NAMES  # the names of the result files


# ==================================================================================================
def low_rank_update_covariance(
    empirical: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    laplace_variances: np.ndarray[tuple[int], np.dtype[np.float64]],
    rank: int,
    inflation: float,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""The Laplace covariance with a rank-$r$ update towards the empirical one.

    With $L = \operatorname{diag}(1/(1+\lambda_i))$ and the eigendecomposition $L^{-1/2}\Sigma
    L^{-1/2} = Q\operatorname{diag}(\mu)Q^T$, the result is $L^{1/2}Q\operatorname{diag}(\tilde\mu)
    Q^TL^{1/2}$ with $\tilde\mu = 1 + a(\max(\mu, 1) - 1)$ for the $r$ largest $\mu$ and 1 for the
    others: the Laplace covariance where the empirical one is not clearly wider, never narrower.

    Args:
        empirical (np.ndarray): Symmetric empirical covariance, shape `(k, k)`.
        laplace_variances (np.ndarray): Laplace variances of the coefficients, shape `(k,)`.
        rank (int): Number $r$ of eigenvalues that are kept, at most $k$.
        inflation (float): Factor $a$ of the excess over 1.

    Returns:
        np.ndarray: The covariance, shape `(k, k)`.
    """
    scale = np.sqrt(laplace_variances)
    outer = np.outer(scale, scale)
    values, vectors = np.linalg.eigh(0.5 * (empirical + empirical.T) / outer)
    updated = np.ones_like(values)
    kept = min(rank, values.size)
    updated[-kept:] = 1.0 + inflation * (np.maximum(values[-kept:], 1.0) - 1.0)
    return (vectors * updated) @ vectors.T * outer


# ==================================================================================================
def score_covariance(
    samples: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    mean: np.ndarray[tuple[int], np.dtype[np.float64]],
    covariance: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> tuple[float, float, np.ndarray]:
    r"""Mean log density, mean shift and whitened spectrum of samples under a Gaussian.

    Args:
        samples (np.ndarray): Samples, shape `(n, k)`.
        mean (np.ndarray): Mean of the Gaussian, shape `(k,)`.
        covariance (np.ndarray): Covariance $\Sigma$ of the Gaussian, shape `(k, k)`.

    Returns:
        tuple[float, float, np.ndarray]: The mean log density of the samples; the squared
            Mahalanobis distance of their mean from `mean`; and the quantiles
            (`SPECTRUM_QUANTILES`) of the eigenvalues of $\Sigma^{-1/2} S \Sigma^{-1/2}$ with $S$
            the covariance of the samples about their own mean. NaN where $\Sigma$ is not
            positive definite.
    """
    num_coefficients = mean.size
    symmetric = 0.5 * (covariance + covariance.T)
    symmetric = symmetric + _JITTER * np.mean(np.diag(symmetric)) * np.eye(num_coefficients)
    try:
        factor = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.full(len(SPECTRUM_QUANTILES), np.nan)
    whitened = solve_triangular(factor, (samples - mean).T, lower=True)
    log_determinant = 2.0 * np.sum(np.log(np.diag(factor)))
    log_density = -0.5 * (
        num_coefficients * np.log(2.0 * np.pi)
        + log_determinant
        + np.mean(np.sum(whitened**2, axis=0))
    )
    sample_mean = samples.mean(axis=0)
    shift = solve_triangular(factor, sample_mean - mean, lower=True)
    sample_covariance = np.cov(samples, rowvar=False)
    half = solve_triangular(factor, sample_covariance, lower=True)
    spectrum = np.linalg.eigvalsh(solve_triangular(factor, half.T, lower=True))
    return float(log_density), float(shift @ shift), np.quantile(spectrum, SPECTRUM_QUANTILES)


# ==================================================================================================
def _candidates(config: CalibrationValidationRunConfig) -> list[tuple[CandidateKind, ...]]:
    """The candidates as (kind, k, shrinkage, inflation, minimum ratio, rank)."""
    candidates: list[tuple] = []
    for k in config.num_coefficients:
        candidates.append((CandidateKind.LAPLACE, k, 0.0, 1.0, 0.0, 0))
        candidates.append((CandidateKind.EMPIRICAL, k, 0.0, 1.0, 0.0, 0))
        for shrinkage, inflation, ratio in product(
            config.coupling_shrinkages, config.variance_inflations, config.minimum_variance_ratios
        ):
            candidates.append((CandidateKind.REGULARIZED, k, shrinkage, inflation, ratio, 0))
        for rank, inflation in product(config.ranks, config.variance_inflations):
            if rank <= k:
                candidates.append((CandidateKind.LOW_RANK, k, 0.0, inflation, 0.0, rank))
    return candidates


# ==================================================================================================
def _build_covariance(
    candidate: tuple, empirical: np.ndarray, laplace_variances: np.ndarray
) -> np.ndarray:
    kind, _, shrinkage, inflation, ratio, rank = candidate
    if kind is CandidateKind.LAPLACE:
        return np.diag(laplace_variances)
    if kind is CandidateKind.EMPIRICAL:
        return empirical
    if kind is CandidateKind.REGULARIZED:
        return regularize_coefficient_covariance(
            empirical, laplace_variances, shrinkage, inflation, ratio
        )
    return low_rank_update_covariance(empirical, laplace_variances, rank, inflation)


# ==================================================================================================
class CalibrationValidationRun(Run[CalibrationValidationRunConfig]):
    """Score covariance calibrations on chains that were not used to estimate them."""

    config_type = CalibrationValidationRunConfig
    name = "calibration-validation"
    outputs = {
        f"results/{_FILES['candidate_kinds']}": "Kind of each candidate covariance.",
        f"results/{_FILES['candidate_parameters']}": "Number of coefficients and settings of each "
        "candidate.",
        f"results/{_FILES['log_likelihood']}": "Mean log density of the held-out chain, per "
        "candidate and held-out chain.",
        f"results/{_FILES['mean_shift']}": "Mahalanobis distance of the held-out chain mean.",
        f"results/{_FILES['spectrum']}": "Quantiles of the whitened spectrum of the held-out "
        "covariance.",
        f"results/{_FILES['eigenvalues']}": "Eigenvalues of the Laplace approximation.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return [
            resolve_repository_path(path) / ChainProjectionResults.FILE_NAMES[name]
            for path in self.config.projection_paths
            for name in ("eigenvalues", "coefficient_chain")
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed in a notebook."""

    # ----------------------------------------------------------------------------------------------
    def _prepare(self, eigenvalues: np.ndarray) -> None:
        """Load what the candidates of a subclass need; `eigenvalues` are those of the chains."""

    # ----------------------------------------------------------------------------------------------
    def _candidate_list(self) -> list[tuple]:
        """The candidates as (kind, k, shrinkage, inflation, minimum ratio, rank)."""
        return _candidates(self.config)

    # ----------------------------------------------------------------------------------------------
    def _covariance(
        self, candidate: tuple, empirical: np.ndarray, laplace_variances: np.ndarray
    ) -> np.ndarray:
        """The covariance of a candidate, estimated from the training chains."""
        return _build_covariance(candidate, empirical, laplace_variances)

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        names = ChainProjectionResults.FILE_NAMES
        with steps.step(f"Loading the coefficients of {len(config.projection_paths)} chains"):
            directories = [resolve_repository_path(path) for path in config.projection_paths]
            eigenvalues = np.load(directories[0] / names["eigenvalues"])
            chains = []
            for directory in directories:
                if not np.array_equal(np.load(directory / names["eigenvalues"]), eigenvalues):
                    raise ValueError(
                        f"The eigenvalues in {directory} differ from those in {directories[0]}: "
                        "the chains are not projected onto the same eigenvectors."
                    )
                chains.append(np.load(directory / names["coefficient_chain"]).astype(np.float64))
        num_chains = len(chains)
        if max(config.num_coefficients) > eigenvalues.size:
            raise ValueError(
                f"num_coefficients up to {max(config.num_coefficients)} exceed the "
                f"{eigenvalues.size} eigenpairs."
            )
        self._prepare(eigenvalues)
        candidates = self._candidate_list()
        logger.info(f"      {len(candidates)} candidates, {num_chains} folds")

        log_likelihood = np.empty((len(candidates), num_chains))
        mean_shift = np.empty((len(candidates), num_chains))
        spectrum = np.empty((len(candidates), num_chains, len(SPECTRUM_QUANTILES)))
        with steps.step("Scoring the candidates on the held-out chains"):
            for held_out in range(num_chains):
                training = np.concatenate([c for i, c in enumerate(chains) if i != held_out])
                for k in config.num_coefficients:
                    laplace_variances = 1.0 / (1.0 + eigenvalues[:k])
                    training_mean = training[:, :k].mean(axis=0)
                    empirical = np.cov(training[:, :k], rowvar=False)
                    for index, candidate in enumerate(candidates):
                        if candidate[1] != k:
                            continue
                        covariance = self._covariance(candidate, empirical, laplace_variances)
                        (
                            log_likelihood[index, held_out],
                            mean_shift[index, held_out],
                            spectrum[index, held_out],
                        ) = score_covariance(chains[held_out][:, :k], training_mean, covariance)

        CalibrationValidationResults(
            candidate_kinds=np.array([int(c[0]) for c in candidates]),
            candidate_parameters=np.array([c[1:] for c in candidates], dtype=float),
            log_likelihood=log_likelihood,
            mean_shift=mean_shift,
            spectrum=spectrum,
            eigenvalues=eigenvalues,
        ).save(self.results_dir(run_dir))

        mean_score = np.nanmean(log_likelihood, axis=1)
        metrics: Metrics = {"num_candidates": len(candidates), "num_folds": num_chains}
        for k in config.num_coefficients:
            in_k = np.array([c[1] == k for c in candidates])
            laplace = next(
                i for i, c in enumerate(candidates) if c[0] is CandidateKind.LAPLACE and c[1] == k
            )
            best = int(np.nanargmax(np.where(in_k, mean_score, -np.inf)))
            reference = [
                i
                for i, c in enumerate(candidates)
                if c[0] is CandidateKind.REGULARIZED
                and c[1] == k
                and (c[2], c[3], c[4])
                == (
                    config.reference_shrinkage,
                    config.reference_inflation,
                    config.reference_minimum_ratio,
                )
            ]
            metrics[f"k{k}_laplace_log_likelihood"] = float(mean_score[laplace])
            metrics[f"k{k}_best_log_likelihood"] = float(mean_score[best])
            metrics[f"k{k}_best_kind"] = candidates[best][0].name
            metrics[f"k{k}_best_parameters"] = (
                f"shrinkage {candidates[best][2]}, inflation {candidates[best][3]}, "
                f"minimum ratio {candidates[best][4]}, rank {candidates[best][5]}"
            )
            if reference:
                metrics[f"k{k}_reference_log_likelihood"] = float(mean_score[reference[0]])
        return metrics


# ==================================================================================================
@dataclass(frozen=True)
class TargetCalibrationValidationRunConfig(CalibrationValidationRunConfig):
    r"""The validation with the empirical covariance shrunk towards a target.

    Attributes:
        target_paths (tuple[Path, ...]): The `results` directories of runs of
            [`HessianAverageRun`][single_runs.hessian_average.HessianAverageRun]; the target $T$ is
            the mean of their `averaged_covariance` (the average of the inverse precisions over
            states of a chain). It covers at least the largest number of coefficients.
        target_weights (tuple[float, ...]): Weights $w\in[0, 1]$ of the target in the candidates
            $(1 - w)S + wT$, with $S$ the empirical covariance of the training chains; the
            variance inflation and the lower bound of the grid are applied to the mixture.
    """

    target_paths: tuple[Path, ...]
    target_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.target_paths) == 0 or len(self.target_weights) == 0:
            raise ValueError("target_paths and target_weights must not be empty.")
        if min(self.target_weights) < 0.0 or max(self.target_weights) > 1.0:
            raise ValueError(f"target_weights must be in [0, 1], got {self.target_weights}.")


# ==================================================================================================
class TargetCalibrationValidationRun(CalibrationValidationRun):
    """The validation of the calibrations, with the Hessian average as a shrinkage target."""

    config_type = TargetCalibrationValidationRunConfig
    name = "calibration-validation-target"

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        config = cast(TargetCalibrationValidationRunConfig, self.config)
        return [
            *super().input_files(),
            *(resolve_repository_path(path) / _TARGET_FILE_NAME for path in config.target_paths),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def _prepare(self, eigenvalues: np.ndarray) -> None:
        config = cast(TargetCalibrationValidationRunConfig, self.config)
        targets = [
            np.load(resolve_repository_path(path) / _TARGET_FILE_NAME)
            for path in config.target_paths
        ]
        if min(target.shape[0] for target in targets) < max(config.num_coefficients):
            raise ValueError(
                f"The targets cover {min(target.shape[0] for target in targets)} coefficients, "
                f"fewer than num_coefficients = {max(config.num_coefficients)}."
            )
        size = min(target.shape[0] for target in targets)
        self._target = np.mean([target[:size, :size] for target in targets], axis=0)

    # ----------------------------------------------------------------------------------------------
    @override
    def _candidate_list(self) -> list[tuple]:
        config = cast(TargetCalibrationValidationRunConfig, self.config)
        candidates = super()._candidate_list()
        for k in config.num_coefficients:
            for weight, inflation, ratio in product(
                config.target_weights, config.variance_inflations, config.minimum_variance_ratios
            ):
                candidates.append((CandidateKind.TARGET, k, weight, inflation, ratio, 0))
        return candidates

    # ----------------------------------------------------------------------------------------------
    @override
    def _covariance(
        self, candidate: tuple, empirical: np.ndarray, laplace_variances: np.ndarray
    ) -> np.ndarray:
        kind, k, weight, inflation, ratio, _ = candidate
        if kind is not CandidateKind.TARGET:
            return super()._covariance(candidate, empirical, laplace_variances)
        mixture = (1.0 - weight) * empirical + weight * self._target[:k, :k]
        return regularize_coefficient_covariance(mixture, laplace_variances, 0.0, inflation, ratio)

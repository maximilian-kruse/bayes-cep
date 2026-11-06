r"""Gaussian from the Gauss-Newton Hessian averaged over chain states, to precondition samplers.

The calibration ([`CalibrationRun`][single_runs.calibrate.CalibrationRun]) estimates the covariance
of the leading Laplace coefficients from the samples of pilot chains, which are noisy where the
chains mix slowly. This run takes it from the Hessians instead: with $K_s = V^TH(u_s)V$ the
Gauss-Newton Hessian at a state $u_s$ in the coefficients of the Laplace eigenvectors $V$ (see
[`HessianAverageRun`][single_runs.hessian_average.HessianAverageRun]), the posterior is, to the
accuracy of the local Gaussian approximations, a mixture of $\mathcal N(\cdot, (I + K_s)^{-1})$
over the states, and its covariance in the leading $k$ coefficients is the leading block of the
average of $(I + K_s)^{-1}$ (the inverse of the average precision is much too narrow in the stiff
directions). The block $T$, mixed with the pilot covariance $S$ of the calibration as $(1 - w)S +
wT$, is inflated, bounded below by the Laplace covariance and cut off at the prior variance like the
calibrated covariance, and the Gaussian is built with the same function
(`build_gaussian_from_coefficient_covariance` of `bayes_cep.laplace.calibration`).

The results are those of a calibration run that the other studies read
(`CalibratedGaussianRecord`, `PilotStates`): the saved states are copied from the calibration run
of the same chains, so that studies that evaluate a preconditioner at the saved states
(`RejectionDiagnosticsRun`, `HmcTrajectoryRun`) compare the two on the same states.

Classes:
    AveragedGaussianRunConfig: The posterior, the chains and the regularization.
    AveragedGaussianResults: The averaged covariance, besides the Gaussian and the states.
    AveragedGaussianRun: The run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.laplace.builder import LaplaceBuilder, LaplaceSettings
from bayes_cep.laplace.calibration import (
    CalibratedGaussianRecord,
    build_gaussian_from_coefficient_covariance,
    regularize_coefficient_covariance,
)
from bayes_cep.laplace.reference_covariance import PriorReferenceCovariance
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.calibrate import CalibrationResults, PilotStates
from single_runs.hessian_average import project_gauss_newton_hessian
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components


# ==================================================================================================
@dataclass(frozen=True)
class AveragedGaussianRunConfig(RunConfig):
    r"""The posterior, the chains whose states carry the Hessians and the regularization.

    Attributes:
        problem (InferenceProblemConfig): The posterior the chains sample.
        map_estimate_path (Path): `.npy` file with the MAP estimate the approximation is centered
            at.
        laplace (LaplaceSettings): Eigensolver and rank of the eigenproblem at the MAP estimate.
        chain_paths (tuple[Path, ...]): The Zarr stores (`samples.zarr`) of finished MCMC runs of
            `problem`; directories, not hashed into the metadata of the run.
        burn_in (int): Number of samples discarded from the start of each chain.
        states_per_chain (int): Number of states of each chain, evenly spaced after the burn-in.
        num_coefficients (int): Number $k$ of leading coefficients whose covariance is replaced.
        variance_inflation (float): Factor multiplying the averaged covariance.
        minimum_variance_ratio (float): Lower bound in units of the Laplace covariance; 1 never
            lets the result be narrower than the Laplace approximation.
        calibration_path (Path): The `results` directory of the calibration run of the same chains:
            its saved states (`PilotStates`) are copied into the results of this run, and its
            `pilot_coefficient_covariance` is the empirical covariance that the average is mixed
            with. The eigenvalues must be those of this run.
        target_weight (float): Weight $w\in[0, 1]$ of the averaged covariance $T$ in $(1 - w)S +
            wT$, with $S$ the pilot covariance; 1 uses the average alone.
        sample_seed (int): Seed of the sample generator of the Gaussian.
    """

    problem: InferenceProblemConfig
    map_estimate_path: Path
    laplace: LaplaceSettings
    chain_paths: tuple[Path, ...]
    burn_in: int
    states_per_chain: int
    num_coefficients: int
    variance_inflation: float
    minimum_variance_ratio: float
    calibration_path: Path
    target_weight: float
    sample_seed: int

    def __post_init__(self) -> None:
        if self.burn_in < 0 or self.states_per_chain < 1 or self.num_coefficients < 1:
            raise ValueError(
                "burn_in must be non-negative, states_per_chain and num_coefficients positive, "
                f"got {self.burn_in}, {self.states_per_chain} and {self.num_coefficients}."
            )
        if len(self.chain_paths) == 0:
            raise ValueError("At least one chain is needed.")
        if not 0.0 <= self.target_weight <= 1.0:
            raise ValueError(f"target_weight must be in [0, 1], got {self.target_weight}.")
        if self.variance_inflation <= 0.0 or self.minimum_variance_ratio < 0.0:
            raise ValueError(
                "variance_inflation must be positive and minimum_variance_ratio non-negative, "
                f"got {self.variance_inflation} and {self.minimum_variance_ratio}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class AveragedGaussianResults(ArrayRecord):
    r"""What the run computed besides the Gaussian and the states; $k$ is the number of
    coefficients, $R$ the rank.

    Attributes:
        eigenvalues (np.ndarray): Eigenvalues of the Laplace approximation, shape `(R,)`.
        laplace_variances (np.ndarray): $1/(1+\lambda_i)$ of the first $k$ eigenvectors, shape
            `(k,)`.
        averaged_covariance (np.ndarray): The leading block of the average of the inverse
            precisions, before the regularization, shape `(k, k)`.
        calibrated_covariance (np.ndarray): The covariance of the Gaussian in the span of the
            first $k$ eigenvectors, after the regularization and the cut-off, shape `(k, k)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "eigenvalues": "hessian_eigenvalues.npy",
        "laplace_variances": "laplace_variances.npy",
        "averaged_covariance": "averaged_covariance.npy",
        "calibrated_covariance": "calibrated_covariance.npy",
    }

    eigenvalues: np.ndarray
    laplace_variances: np.ndarray
    averaged_covariance: np.ndarray
    calibrated_covariance: np.ndarray


_FILES = AveragedGaussianResults.FILE_NAMES  # the names of the result files
_GAUSSIAN_FILES = CalibratedGaussianRecord.FILE_NAMES
_STATE_FILES = PilotStates.FILE_NAMES


# ==================================================================================================
class AveragedGaussianRun(Run[AveragedGaussianRunConfig]):
    """The Gaussian preconditioner from the Hessians averaged over chain states."""

    config_type = AveragedGaussianRunConfig
    name = "averaged-gaussian"
    outputs = {
        f"results/{_FILES['eigenvalues']}": "Eigenvalues of the Laplace approximation.",
        f"results/{_FILES['laplace_variances']}": "Laplace variances of the first coefficients.",
        f"results/{_FILES['averaged_covariance']}": "Leading block of the average of the "
        "inverse precisions.",
        f"results/{_FILES['calibrated_covariance']}": "Covariance of the Gaussian in the span of "
        "the first eigenvectors.",
        f"results/{_GAUSSIAN_FILES['eigenvalues']}": "Eigenvalues of the Gaussian.",
        f"results/{_GAUSSIAN_FILES['eigenvectors']}": "Its eigenvectors, one column each.",
        f"results/{_GAUSSIAN_FILES['mean']}": "Its mean.",
        f"results/{_STATE_FILES['states']}": "The saved states of the calibration run.",
        f"results/{_STATE_FILES['indices']}": "Their pilot chain and sample index.",
        METRICS_RECORD: "Rank, extreme eigenvalues and the number of states of the average.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        # The chains are directories (Zarr stores) and not hashed.
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.map_estimate_path),
            *CalibrationResults.file_paths(resolve_repository_path(self.config.calibration_path)),
            *PilotStates.file_paths(resolve_repository_path(self.config.calibration_path)),
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
        assembled = assemble_posterior_components(config.problem, steps, logger)
        components = assembled.components
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))

        with steps.step(f"Eigenproblem of rank {config.laplace.num_eigenvalues} at the MAP"):
            approximation = LaplaceBuilder(config.laplace, components).build(map_estimate)
        eigenvalues = approximation.eigenpairs.eigenvalues
        eigenvectors = approximation.eigenpairs.eigenvectors
        rank = eigenvalues.size
        if config.num_coefficients > rank:
            raise ValueError(
                f"num_coefficients = {config.num_coefficients} exceeds the {rank} eigenpairs."
            )
        logger.info(f"      {describe_array('eigenvalues', eigenvalues)}")

        identity = np.eye(rank)
        covariance_sum = np.zeros((rank, rank))
        num_states = 0
        for position, chain_path in enumerate(config.chain_paths):
            chain = open_zarr_samples(resolve_repository_path(chain_path))
            indices = np.linspace(
                config.burn_in, chain.shape[0] - 1, config.states_per_chain
            ).astype(int)
            for index in indices:
                with steps.step(f"Chain {position}, sample {index}"):
                    precision = identity + project_gauss_newton_hessian(
                        components, np.asarray(chain[int(index)]), eigenvectors
                    )
                covariance_sum += np.linalg.inv(precision)
                num_states += 1
        k = config.num_coefficients
        averaged = covariance_sum[:k, :k] / num_states

        calibration_dir = resolve_repository_path(config.calibration_path)
        calibration = CalibrationResults.load(calibration_dir)
        if not np.array_equal(calibration.eigenvalues, eigenvalues):
            raise ValueError(
                f"The eigenvalues of the calibration in {calibration_dir} differ from those of "
                "this run: the coefficients refer to different eigenvectors."
            )
        pilot = calibration.pilot_coefficient_covariance
        if pilot.shape[0] < k:
            raise ValueError(
                f"The pilot covariance has {pilot.shape[0]} coefficients, fewer than {k}."
            )
        laplace_variances = 1.0 / (1.0 + eigenvalues[:k])
        mixture = (1.0 - config.target_weight) * pilot[:k, :k] + config.target_weight * averaged
        regularized = regularize_coefficient_covariance(
            0.5 * (mixture + mixture.T),
            laplace_variances,
            0.0,
            config.variance_inflation,
            config.minimum_variance_ratio,
        )
        reference = PriorReferenceCovariance(components.prior)
        covariance, gaussian, num_clipped = build_gaussian_from_coefficient_covariance(
            approximation, reference, regularized, approximation.gaussian.mean, config.sample_seed
        )

        results_dir = self.results_dir(run_dir)
        AveragedGaussianResults(
            eigenvalues=eigenvalues,
            laplace_variances=laplace_variances,
            averaged_covariance=averaged,
            calibrated_covariance=covariance,
        ).save(results_dir)
        CalibratedGaussianRecord.from_approximation(gaussian).save(results_dir)
        PilotStates.load(calibration_dir).save(results_dir)
        new_eigenvalues = gaussian.eigenpairs.eigenvalues
        return {
            "num_eigenpairs": int(new_eigenvalues.size),
            "num_states": int(num_states),
            "num_clipped": num_clipped,
            "eigenvalue_max": float(new_eigenvalues.max()),
            "eigenvalue_min": float(new_eigenvalues.min()),
            "variance_ratio_max": float(np.max(np.diag(covariance) / laplace_variances)),
        }

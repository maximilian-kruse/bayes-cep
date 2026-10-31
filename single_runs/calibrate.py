r"""Calibration of the Laplace approximation by pilot chains, without sampling.

The calibration of [`mcmc_calibrated`][single_runs.mcmc_calibrated] as a run of its own: it
computes the Laplace approximation at the MAP estimate, streams over the pilot chains for the
moments of the leading coefficients and calibrates the approximation (see
[`calibrate_approximation`][bayes_cep.laplace.calibration.calibrate_approximation]), and writes the
calibrated Gaussian with its eigenvectors and a few states of each pilot chain. The pilot chains
are large (a chain of 130000 states of the 15820 vertices is 16 GB), so the run is for the machine
they are on, and its result, about 100 MB, can be used elsewhere: to check step widths in a
notebook (the states are in the typical set, which is where a chain should start) or to run many
chains with one calibration.

Classes:
    CalibrationOutcome: The Laplace approximation, the moments and the calibrated approximation.
    CalibrationResults: The arrays of the Laplace approximation and its calibration.
    PilotStates: A few states of each pilot chain.
    CalibrationRunConfig: The posterior, the eigenproblem, the calibration and the pilot chains.
    CalibrationRun: The calibration run.

Functions:
    compute_calibration: The calibration of the Laplace approximation by the pilot chains.
    calibration_metrics: The scalar summary of a calibration.
    calibration_results: The arrays of a calibration.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.laplace.builder import LaplaceApproximation, LaplaceBuilder, LaplaceSettings
from bayes_cep.laplace.calibration import (
    CalibratedApproximation,
    CalibratedGaussianRecord,
    CalibrationSettings,
    SubspaceMoments,
    calibrate_approximation,
    estimate_subspace_moments,
)
from bayes_cep.laplace.reference_covariance import PriorReferenceCovariance
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.inference import (
    AssembledPosteriorComponents,
    InferenceProblemConfig,
    assemble_posterior_components,
)


# ==================================================================================================
@dataclass(frozen=True)
class CalibrationOutcome:
    """What the calibration computed.

    Attributes:
        assembled (AssembledPosteriorComponents): The posterior components and the data.
        laplace (LaplaceApproximation): The Laplace approximation at the MAP estimate.
        moments (SubspaceMoments): Moments of the pilot chains.
        calibrated (CalibratedApproximation): The calibrated approximation.
        seconds_approximation (float): Wall time of the Laplace approximation.
        seconds_calibration (float): Wall time of the moments and the calibration.
    """

    assembled: AssembledPosteriorComponents
    laplace: LaplaceApproximation
    moments: SubspaceMoments
    calibrated: CalibratedApproximation
    seconds_approximation: float
    seconds_calibration: float


# ==================================================================================================
@dataclass(frozen=True)
class CalibrationResults(ArrayRecord):
    r"""The Laplace approximation and its calibration; $k$ is the number of calibrated
    coefficients, $R$ the rank.

    Attributes:
        eigenvalues (np.ndarray): Eigenvalues of the Laplace approximation, descending, shape
            `(R,)`.
        calibrated_eigenvalues (np.ndarray): Eigenvalues of the calibrated approximation,
            descending, shape `(R,)`.
        laplace_variances (np.ndarray): $1/(1+\lambda_i)$ of the first $k$ eigenvectors, shape
            `(k,)`.
        pilot_coefficient_covariance (np.ndarray): Covariance of their coefficients in the pilot
            chains, shape `(k, k)`.
        calibrated_covariance (np.ndarray): The covariance after the regularization, the one of the
            preconditioner in the span of the first $k$ eigenvectors, shape `(k, k)`.
        pilot_coefficient_mean (np.ndarray): Mean of the coefficients, from the MAP estimate, shape
            `(k,)`.
        pilot_mean (np.ndarray): Mean of the pilot chains, shape `(num_vertices,)`.
        pilot_num_samples (np.ndarray): Number of samples used, shape `(1,)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "eigenvalues": "hessian_eigenvalues.npy",
        "calibrated_eigenvalues": "calibrated_eigenvalues.npy",
        "laplace_variances": "laplace_variances.npy",
        "pilot_coefficient_covariance": "pilot_coefficient_covariance.npy",
        "calibrated_covariance": "calibrated_covariance.npy",
        "pilot_coefficient_mean": "pilot_coefficient_mean.npy",
        "pilot_mean": "pilot_mean.npy",
        "pilot_num_samples": "pilot_num_samples.npy",
    }

    eigenvalues: np.ndarray
    calibrated_eigenvalues: np.ndarray
    laplace_variances: np.ndarray
    pilot_coefficient_covariance: np.ndarray
    calibrated_covariance: np.ndarray
    pilot_coefficient_mean: np.ndarray
    pilot_mean: np.ndarray
    pilot_num_samples: np.ndarray


# ==================================================================================================
@dataclass(frozen=True)
class PilotStates(ArrayRecord):
    """A few states of each pilot chain, evenly spaced after the burn-in.

    Attributes:
        states (np.ndarray): The states [rad], shape `(num_chains * num_saved_states,
            num_vertices)`, chain by chain.
        indices (np.ndarray): Position of the chain among the pilot chains and index of the sample
            in it, shape `(num_chains * num_saved_states, 2)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "states": "pilot_states.npy",
        "indices": "pilot_state_indices.npy",
    }

    states: np.ndarray
    indices: np.ndarray


_RESULT_FILES = CalibrationResults.FILE_NAMES
_STATE_FILES = PilotStates.FILE_NAMES
_GAUSSIAN_FILES = CalibratedGaussianRecord.FILE_NAMES


# ==================================================================================================
def compute_calibration(
    problem: InferenceProblemConfig,
    map_estimate_path: Path,
    laplace_settings: LaplaceSettings,
    calibration: CalibrationSettings,
    pilot_chain_paths: tuple[Path, ...],
    steps: StepReporter,
    logger: BaseLogger,
) -> CalibrationOutcome:
    """The Laplace approximation at the MAP estimate, calibrated by the pilot chains.

    Args:
        problem (InferenceProblemConfig): The posterior the pilot chains sample.
        map_estimate_path (Path): `.npy` file with the MAP estimate, one angle [rad] per vertex.
        laplace_settings (LaplaceSettings): Eigensolver and rank of the Laplace approximation.
        calibration (CalibrationSettings): Number of calibrated coefficients, burn-in and
            thinning of the pilot chains and the regularization of their covariance.
        pilot_chain_paths (tuple[Path, ...]): The Zarr stores of the pilot chains.
        steps (StepReporter): Reporter of the steps.
        logger (BaseLogger): Logger of the run.

    Returns:
        CalibrationOutcome: The approximations, the moments and the posterior components.
    """
    assembled = assemble_posterior_components(problem, steps, logger)
    prior = assembled.components.prior
    map_estimate = np.load(resolve_repository_path(map_estimate_path))
    logger.info(f"      {describe_array('MAP estimate [rad]', map_estimate)}")
    start = time.perf_counter()
    with steps.step(
        f"Computing {laplace_settings.num_eigenvalues} eigenpairs of the misfit Hessian"
    ):
        laplace = LaplaceBuilder(laplace_settings, assembled.components).build(map_estimate)
    seconds_approximation = time.perf_counter() - start
    eigenvalues = laplace.eigenpairs.eigenvalues
    num_coefficients = min(calibration.num_coefficients, eigenvalues.size)
    logger.info(
        f"      kept {eigenvalues.size} of {laplace_settings.num_eigenvalues} eigenpairs; "
        f"calibrating the leading {num_coefficients}"
    )

    start = time.perf_counter()
    with steps.step(f"Estimating the moments from {len(pilot_chain_paths)} pilot chain(s)"):
        weighted_eigenvectors = np.stack(
            [
                prior.apply_precision_operator(vector)
                for vector in laplace.eigenpairs.eigenvectors[:, :num_coefficients].T
            ],
            axis=1,
        )
        chains = [open_zarr_samples(resolve_repository_path(path)) for path in pilot_chain_paths]
        moments = estimate_subspace_moments(
            chains, weighted_eigenvectors, map_estimate, calibration
        )
    with steps.step("Calibrating the approximation"):
        calibrated = calibrate_approximation(
            laplace,
            PriorReferenceCovariance(prior),
            moments,
            calibration,
            seed=laplace_settings.sample_seed,
        )
    seconds_calibration = time.perf_counter() - start
    logger.info(
        "      "
        + describe_array("pilot variance / Laplace variance", calibrated.empirical_variance_ratios)
    )
    logger.info(
        "      "
        + describe_array(
            "calibrated variance / Laplace variance", calibrated.calibrated_variance_ratios
        )
    )
    if calibrated.num_clipped > 0:
        logger.info(
            f"      {calibrated.num_clipped} principal variances exceeded the prior variance "
            "and were cut off"
        )
    return CalibrationOutcome(
        assembled, laplace, moments, calibrated, seconds_approximation, seconds_calibration
    )


# ==================================================================================================
def calibration_metrics(outcome: CalibrationOutcome) -> Metrics:
    """The scalar summary of a calibration: the approximation, the pilot and the variance ratios."""
    eigenvalues = outcome.laplace.eigenpairs.eigenvalues
    calibrated = outcome.calibrated
    return {
        "num_eigenpairs": int(eigenvalues.size),
        "eigenvalue_max": float(eigenvalues[0]),
        "eigenvalue_min": float(eigenvalues[-1]),
        "seconds_approximation": outcome.seconds_approximation,
        "seconds_calibration": outcome.seconds_calibration,
        "pilot_num_samples": int(outcome.moments.num_samples[0]),
        "pilot_variance_ratio_median": float(np.median(calibrated.empirical_variance_ratios)),
        "pilot_variance_ratio_max": float(np.max(calibrated.empirical_variance_ratios)),
        "calibrated_variance_ratio_median": float(np.median(calibrated.calibrated_variance_ratios)),
        "calibrated_variance_ratio_max": float(np.max(calibrated.calibrated_variance_ratios)),
        "calibration_num_clipped": calibrated.num_clipped,
    }


# ==================================================================================================
def calibration_results(outcome: CalibrationOutcome) -> CalibrationResults:
    """The arrays of a calibration."""
    return CalibrationResults(
        eigenvalues=outcome.laplace.eigenpairs.eigenvalues,
        calibrated_eigenvalues=outcome.calibrated.approximation.eigenpairs.eigenvalues,
        laplace_variances=outcome.calibrated.laplace_variances,
        pilot_coefficient_covariance=outcome.moments.coefficient_covariance,
        calibrated_covariance=outcome.calibrated.calibrated_covariance,
        pilot_coefficient_mean=outcome.moments.coefficient_mean,
        pilot_mean=outcome.moments.mean,
        pilot_num_samples=outcome.moments.num_samples,
    )


# ==================================================================================================
@dataclass(frozen=True)
class CalibrationRunConfig(RunConfig):
    """The posterior, the eigenproblem, the calibration and the pilot chains.

    Attributes:
        problem (InferenceProblemConfig): The posterior the pilot chains sample.
        map_estimate_path (Path): `.npy` file with the MAP estimate, one angle [rad] per vertex;
            a relative path is relative to the repository. The approximation is centered there.
        laplace (LaplaceSettings): Eigensolver and rank of the Laplace approximation.
        calibration (CalibrationSettings): Number of calibrated coefficients, burn-in and
            thinning of the pilot chains, and the regularization of their covariance.
        pilot_chain_paths (tuple[Path, ...]): The Zarr stores (`samples.zarr`) of finished chains
            of this posterior, pooled; a relative path is relative to the repository. They are
            directories and not hashed into the metadata of the run.
        num_saved_states (int): Number of states of each pilot chain saved, evenly spaced from the
            end of the burn-in to the end of the chain.
    """

    problem: InferenceProblemConfig
    map_estimate_path: Path
    laplace: LaplaceSettings
    calibration: CalibrationSettings
    pilot_chain_paths: tuple[Path, ...]
    num_saved_states: int

    def __post_init__(self) -> None:
        if len(self.pilot_chain_paths) == 0:
            raise ValueError("At least one pilot chain is needed.")
        if self.num_saved_states < 1:
            raise ValueError(f"num_saved_states must be positive, got {self.num_saved_states}.")


# ==================================================================================================
class CalibrationRun(Run[CalibrationRunConfig]):
    """Calibrate the Laplace approximation of a posterior by pilot chains, and keep the result."""

    config_type = CalibrationRunConfig
    name = "calibrate"
    outputs = {
        f"results/{_RESULT_FILES['eigenvalues']}": "Eigenvalues of the Laplace approximation, "
        "descending.",
        f"results/{_RESULT_FILES['calibrated_eigenvalues']}": "Eigenvalues of the calibrated "
        "approximation, descending.",
        f"results/{_RESULT_FILES['laplace_variances']}": "Laplace variances of the calibrated "
        "coefficients.",
        f"results/{_RESULT_FILES['pilot_coefficient_covariance']}": "Covariance of the "
        "coefficients in the pilot chains.",
        f"results/{_RESULT_FILES['calibrated_covariance']}": "Their covariance after the "
        "regularization, the one of the preconditioner.",
        f"results/{_RESULT_FILES['pilot_coefficient_mean']}": "Mean of the coefficients in the "
        "pilot chains.",
        f"results/{_RESULT_FILES['pilot_mean']}": "Mean of the pilot chains.",
        f"results/{_RESULT_FILES['pilot_num_samples']}": "Number of pilot samples used.",
        f"results/{_GAUSSIAN_FILES['eigenvalues']}": "Eigenvalues of the calibrated Gaussian.",
        f"results/{_GAUSSIAN_FILES['eigenvectors']}": "Its eigenvectors, one column each.",
        f"results/{_GAUSSIAN_FILES['mean']}": "Its mean.",
        f"results/{_STATE_FILES['states']}": "A few states of each pilot chain.",
        f"results/{_STATE_FILES['indices']}": "Pilot chain and sample index of these states.",
        METRICS_RECORD: "Rank, extreme eigenvalues, the seconds of the approximation and the "
        "calibration, the number of pilot samples and the variance ratios of the pilot chains and "
        "the calibrated covariance over the Laplace variances.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.map_estimate_path),
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
        outcome = compute_calibration(
            config.problem,
            config.map_estimate_path,
            config.laplace,
            config.calibration,
            config.pilot_chain_paths,
            steps,
            logger,
        )
        results_dir = self.results_dir(run_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        calibration_results(outcome).save(results_dir)
        CalibratedGaussianRecord.from_approximation(outcome.calibrated.approximation).save(
            results_dir
        )
        with steps.step(f"Saving {config.num_saved_states} states of each pilot chain"):
            states, indices = [], []
            for position, path in enumerate(config.pilot_chain_paths):
                chain = open_zarr_samples(resolve_repository_path(path))
                sample_indices = np.linspace(
                    config.calibration.burn_in, chain.shape[0] - 1, config.num_saved_states
                ).astype(int)
                for index in sample_indices:
                    states.append(np.asarray(chain[int(index)], dtype=np.float64))
                    indices.append((position, int(index)))
            PilotStates(states=np.stack(states), indices=np.array(indices)).save(results_dir)
        return calibration_metrics(outcome)

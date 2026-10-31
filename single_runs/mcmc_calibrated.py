r"""MCMC run preconditioned by a Laplace approximation calibrated with pilot chains.

The run is the [`PreconditionedMcmcRun`][single_runs.mcmc_laplace.PreconditionedMcmcRun] with a
preconditioner that is not the Laplace approximation itself. The run computes the Laplace
approximation at the MAP estimate as there, streams over finished chains of the same posterior, the
pilot chains, for the empirical covariance of the coefficients of the leading eigenvectors, and
replaces the covariance of the approximation in their span by it (see
[`calibrate_approximation`][bayes_cep.laplace.calibration.calibrate_approximation]). The result is
held fixed for the chain: a frozen, sample-informed preconditioner, so the chain targets the
posterior whatever the pilot chains were.

The pilot chains can be those of a `PreconditionedMcmcRun` of the same posterior and rank, or of an
earlier calibrated run: the coefficients are computed here, with the eigenvectors of this run, so
nothing but the Zarr store of the chain is read. A second round, with the chain of the first as
pilot, improves the estimate in the directions the first chain explored slowly.

Classes:
    CalibratedMcmcRunConfig: The configuration of the preconditioned run and the calibration.
    CalibratedMcmcRun: The calibrated MCMC run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

from ls_bayesian.common.logging import BaseLogger

from bayes_cep.laplace.calibration import CalibrationSettings
from bayes_cep.mcmc.measures import LowRankGaussianMeasure
from bayes_cep.run.progress import StepReporter
from single_runs.calibrate import (
    CalibrationResults,
    calibration_metrics,
    calibration_results,
    compute_calibration,
)
from single_runs.mcmc import SamplingProblem
from single_runs.mcmc_laplace import PreconditionedMcmcRun, PreconditionedMcmcRunConfig

_FILES = CalibrationResults.FILE_NAMES  # the names of the result files, defined once there


# ==================================================================================================
@dataclass(frozen=True)
class CalibratedMcmcRunConfig(PreconditionedMcmcRunConfig):
    """The configuration of the preconditioned run and of the calibration of its preconditioner.

    Attributes:
        calibration (CalibrationSettings): Number of calibrated coefficients, burn-in and
            thinning of the pilot chains, and the regularization of their covariance.
        pilot_chain_paths (tuple[Path, ...]): The Zarr stores (`samples.zarr`) of finished chains
            of this posterior, pooled; a relative path is relative to the repository. The
            stores are directories and not hashed into the metadata of the run.
    """

    calibration: CalibrationSettings
    pilot_chain_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.pilot_chain_paths) == 0:
            raise ValueError("At least one pilot chain is needed.")


# ==================================================================================================
class CalibratedMcmcRun(PreconditionedMcmcRun):
    """Sample the posterior with PMALA or pCN preconditioned by a Laplace approximation calibrated
    with pilot chains."""

    config_type = CalibratedMcmcRunConfig
    name = "mcmc-calibrated"
    outputs = {
        **PreconditionedMcmcRun.outputs,
        f"results/{_FILES['calibrated_eigenvalues']}": "Eigenvalues of the calibrated "
        "approximation, descending.",
        f"results/{_FILES['laplace_variances']}": "Laplace variances of the calibrated "
        "coefficients.",
        f"results/{_FILES['pilot_coefficient_covariance']}": "Covariance of the coefficients in "
        "the pilot chains.",
        f"results/{_FILES['calibrated_covariance']}": "Their covariance after the regularization, "
        "the one of the preconditioner.",
        f"results/{_FILES['pilot_coefficient_mean']}": "Mean of the coefficients in the pilot "
        "chains.",
        f"results/{_FILES['pilot_mean']}": "Mean of the pilot chains.",
        f"results/{_FILES['pilot_num_samples']}": "Number of pilot samples used.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def _assemble_sampling_problem(
        self, steps: StepReporter, logger: BaseLogger
    ) -> SamplingProblem:
        config = cast(CalibratedMcmcRunConfig, self.config)
        outcome = compute_calibration(
            config.problem,
            config.map_estimate_path,
            config.laplace,
            config.calibration,
            config.pilot_chain_paths,
            steps,
            logger,
        )
        return SamplingProblem(
            log_posterior=outcome.assembled.components.to_log_posterior(),
            approximation=LowRankGaussianMeasure(outcome.calibrated.approximation.gaussian),
            metrics=calibration_metrics(outcome),
            results=calibration_results(outcome),
        )

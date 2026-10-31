r"""MCMC run preconditioned by the calibrated Gaussian of a finished calibration run.

The run is the [`McmcRun`][single_runs.mcmc.McmcRun] with the calibrated Gaussian of a
[`CalibrationRun`][single_runs.calibrate.CalibrationRun] as the `approximation` of the sampler, read
from the `results` directory of that run (its eigenvalues, eigenvectors and mean) instead of
computed: no Hessian eigenproblem and no pilot chains. Many chains, e.g. a sweep of step widths or
several starting states, then share one calibration. The Gaussian must belong to the posterior of
the run (same data, prior and rank); its eigenvectors are `Gamma_0^{-1}`-orthonormal for the prior
of that posterior, which the shape check below cannot verify.

Classes:
    StoredCalibrationMcmcRunConfig: The configuration of the plain run and the calibration to read.
    StoredCalibrationMcmcRun: The MCMC run with a stored calibration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.laplace.calibration import CalibratedGaussianRecord
from bayes_cep.laplace.reference_covariance import PriorReferenceCovariance
from bayes_cep.mcmc.measures import LowRankGaussianMeasure
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.run.progress import StepReporter
from single_runs.inference import assemble_posterior_components
from single_runs.mcmc import McmcRun, McmcRunConfig, SamplingProblem


# ==================================================================================================
@dataclass(frozen=True)
class StoredCalibrationMcmcRunConfig(McmcRunConfig):
    """The configuration of the plain run and of the calibration it is preconditioned by.

    Attributes:
        calibration_path (Path): The `results` directory of a finished calibration run of this
            posterior; a relative path is relative to the repository. The files of the calibrated
            Gaussian in it are hashed into the metadata of the run.
        sample_seed (int): Seed of the sample generator of the Gaussian, which only a start drawn
            from it would use.
    """

    calibration_path: Path
    sample_seed: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sampler.algorithm.requires_approximation:
            raise ValueError(
                f"The algorithm {self.sampler.algorithm.name} does not use an approximation; "
                "use the plain MCMC run."
            )


# ==================================================================================================
class StoredCalibrationMcmcRun(McmcRun):
    """Sample the posterior with PMALA or pCN preconditioned by a stored calibrated Gaussian."""

    config_type = StoredCalibrationMcmcRunConfig
    name = "mcmc-stored-calibration"

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        config = cast(StoredCalibrationMcmcRunConfig, self.config)
        calibration_dir = resolve_repository_path(config.calibration_path)
        return [
            *super().input_files(),
            *(calibration_dir / name for name in CalibratedGaussianRecord.FILE_NAMES.values()),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def _assemble_sampling_problem(
        self, steps: StepReporter, logger: BaseLogger
    ) -> SamplingProblem:
        config = cast(StoredCalibrationMcmcRunConfig, self.config)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        prior = assembled.components.prior
        with steps.step(f"Loading the calibrated Gaussian from {config.calibration_path}"):
            record = CalibratedGaussianRecord.load(resolve_repository_path(config.calibration_path))
        num_vertices = prior.mean_vector.shape[0]
        if record.eigenvectors.shape[0] != num_vertices or record.mean.shape != (num_vertices,):
            raise ValueError(
                f"The calibrated Gaussian has {record.eigenvectors.shape[0]} vertices, the "
                f"posterior {num_vertices}."
            )
        gaussian = record.build_gaussian(PriorReferenceCovariance(prior), config.sample_seed)
        eigenvalues = np.asarray(record.eigenvalues)
        logger.info(
            f"      {eigenvalues.size} eigenpairs, eigenvalues from {eigenvalues.max():.3e} down "
            f"to {eigenvalues.min():.3e}"
        )
        return SamplingProblem(
            log_posterior=assembled.components.to_log_posterior(),
            approximation=LowRankGaussianMeasure(gaussian),
            metrics={
                "num_eigenpairs": int(eigenvalues.size),
                "eigenvalue_max": float(eigenvalues.max()),
                "eigenvalue_min": float(eigenvalues.min()),
            },
        )

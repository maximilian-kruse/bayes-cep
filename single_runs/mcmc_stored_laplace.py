r"""MCMC run preconditioned by the Laplace approximation of a finished Laplace run.

The run is the [`McmcRun`][single_runs.mcmc.McmcRun] with the low-rank Laplace approximation of a
[`LaplaceRun`][single_runs.laplace.LaplaceRun] as the `approximation` of the sampler, built from the
eigenvalues and eigenvectors in the `results` directory of that run instead of computed: no Hessian
eigenproblem. Many chains, e.g. several starting states or step widths, then share one
approximation (the randomized eigensolver does not reproduce its eigenvectors across machines, so
chains that compute it themselves precondition with slightly different approximations). The
approximation must belong to the posterior of the run (same data, prior and rank) and be centered
at the MAP estimate it was computed at; its eigenvectors are `Gamma_0^{-1}`-orthonormal for the
prior of that posterior, which the shape check below cannot verify.

Classes:
    StoredLaplaceMcmcRunConfig: The configuration of the plain run and the Laplace run to read.
    StoredLaplaceMcmcRun: The MCMC run with a stored Laplace approximation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.lowrank.eigensolvers import GeneralizedEigenpairs
from ls_bayesian.lowrank.gaussian import LowRankGaussian

from bayes_cep.laplace.reference_covariance import PriorReferenceCovariance
from bayes_cep.mcmc.measures import LowRankGaussianMeasure
from bayes_cep.optimization.initial_guess import (
    ChainStateInitialGuess,
    FileInitialGuess,
    PriorMeanInitialGuess,
    PriorSampleInitialGuess,
)
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from single_runs.inference import assemble_posterior_components
from single_runs.laplace import LaplaceResults
from single_runs.mcmc import McmcRun, McmcRunConfig, SamplingProblem
from single_runs.mcmc_laplace import LaplaceSampleInitialState

_EIGENPAIR_FILES = (
    LaplaceResults.FILE_NAMES["eigenvalues"],
    LaplaceResults.FILE_NAMES["eigenvectors"],
)


# ==================================================================================================
@dataclass(frozen=True)
class StoredLaplaceMcmcRunConfig(McmcRunConfig):
    """The configuration of the plain run and of the Laplace approximation it is preconditioned by.

    Attributes:
        initial_state (PriorMeanInitialGuess | PriorSampleInitialGuess | FileInitialGuess |
            ChainStateInitialGuess | LaplaceSampleInitialState): As in the plain run, or a draw from
            the approximation.
        laplace_path (Path): The `results` directory of a finished Laplace run of this posterior; a
            relative path is relative to the repository. Its eigenvalues and eigenvectors are
            hashed into the metadata of the run.
        map_estimate_path (Path): `.npy` file with the MAP estimate the Laplace run was centered at
            (its `map_estimate_path`); a relative path is relative to the repository. The
            approximation is centered there.
        sample_seed (int): Seed of the sample generator of the approximation, which only a start
            drawn from it would use.
    """

    initial_state: (
        PriorMeanInitialGuess
        | PriorSampleInitialGuess
        | FileInitialGuess
        | ChainStateInitialGuess
        | LaplaceSampleInitialState
    )
    laplace_path: Path
    map_estimate_path: Path
    sample_seed: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sampler.algorithm.requires_approximation:
            raise ValueError(
                f"The algorithm {self.sampler.algorithm.name} does not use an approximation; "
                "use the plain MCMC run."
            )


# ==================================================================================================
class StoredLaplaceMcmcRun(McmcRun):
    """Sample the posterior with pCN or MALA preconditioned by a stored Laplace approximation."""

    config_type = StoredLaplaceMcmcRunConfig
    name = "mcmc-stored-laplace"

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        config = cast(StoredLaplaceMcmcRunConfig, self.config)
        laplace_dir = resolve_repository_path(config.laplace_path)
        return [
            *super().input_files(),
            *(laplace_dir / name for name in _EIGENPAIR_FILES),
            resolve_repository_path(config.map_estimate_path),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def _build_initial_state(
        self, sampling_problem: SamplingProblem
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        initial_state = self.config.initial_state
        if isinstance(initial_state, LaplaceSampleInitialState):
            return initial_state.build(cast(LowRankGaussianMeasure, sampling_problem.approximation))
        return super()._build_initial_state(sampling_problem)

    # ----------------------------------------------------------------------------------------------
    @override
    def _assemble_sampling_problem(
        self, steps: StepReporter, logger: BaseLogger
    ) -> SamplingProblem:
        config = cast(StoredLaplaceMcmcRunConfig, self.config)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        prior = assembled.components.prior
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))
        logger.info(f"      {describe_array('MAP estimate [rad]', map_estimate)}")
        laplace_dir = resolve_repository_path(config.laplace_path)
        with steps.step(f"Loading the Laplace approximation from {config.laplace_path}"):
            eigenvalues = np.load(laplace_dir / LaplaceResults.FILE_NAMES["eigenvalues"])
            eigenvectors = np.load(laplace_dir / LaplaceResults.FILE_NAMES["eigenvectors"])
        num_vertices = prior.mean_vector.shape[0]
        if eigenvectors.shape[0] != num_vertices or map_estimate.shape != (num_vertices,):
            raise ValueError(
                f"The Laplace approximation has {eigenvectors.shape[0]} vertices and the MAP "
                f"estimate {map_estimate.shape[0]}, the posterior {num_vertices}."
            )
        gaussian = LowRankGaussian(
            PriorReferenceCovariance(prior),
            GeneralizedEigenpairs(eigenvalues, eigenvectors),
            mean=map_estimate,
            seed=config.sample_seed,
        )
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

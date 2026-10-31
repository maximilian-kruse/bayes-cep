r"""MCMC run preconditioned by the Laplace approximation at the MAP estimate.

The run is the [`McmcRun`][single_runs.mcmc.McmcRun] with a Gaussian approximation of the
posterior: the low-rank Laplace approximation $\mathcal N(m_\mathrm{MAP}, \Gamma)$ at a stored MAP
estimate (see [`LaplaceBuilder`][bayes_cep.laplace.builder.LaplaceBuilder]; its Gauss-Newton
Hessian is evaluated inside this run, so it depends on the MAP estimate file only). The sampler is
either pCN with $\mathcal N(m_\mathrm{MAP}, \Gamma)$ as proposal (`PCN_PRECONDITIONED`) or MALA
with $\Gamma$ as preconditioner (`PMALA`). Posterior summary, chain diagnostics, plots and metrics
are those of the plain run, so the two are directly comparable; the metrics also hold the extent
and cost of the approximation.

Classes:
    LaplaceSampleInitialState: Start the chain at a draw from the Laplace approximation.
    PreconditionedMcmcRunConfig: The configuration of the plain run and the Laplace approximation.
    LaplaceSpectrum: The eigenvalues of the Laplace approximation.
    PreconditionedMcmcRun: The Laplace-preconditioned MCMC run.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.laplace.builder import LaplaceBuilder, LaplaceSettings
from bayes_cep.mcmc.measures import LowRankGaussianMeasure
from bayes_cep.optimization.initial_guess import (
    ChainStateInitialGuess,
    FileInitialGuess,
    PriorMeanInitialGuess,
    PriorSampleInitialGuess,
)
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from single_runs.inference import assemble_posterior_components
from single_runs.mcmc import McmcRun, McmcRunConfig, SamplingProblem


# ==================================================================================================
@dataclass(frozen=True)
class LaplaceSampleInitialState:
    r"""Start the chain at a draw $m_0 = m_\mathrm{MAP} + \widehat\Gamma\,\xi$ from the Laplace
    approximation.

    The standard normal vector $\xi$ is drawn from a generator seeded with `seed`, independent of
    the seed of the chain and of `LaplaceSettings.sample_seed`, so the start varies with `seed`
    alone. The approximation exists only inside the run, so unlike the strategies of
    [`InitialGuessStrategy`][bayes_cep.optimization.initial_guess.InitialGuessStrategy] this one is
    built from it.

    Attributes:
        seed (int): Seed of the draw.
    """

    seed: int

    # ----------------------------------------------------------------------------------------------
    def build(
        self, approximation: LowRankGaussianMeasure
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Draw the initial state, shape `(approximation.random_vector_size,)`."""
        random_vector = np.random.default_rng(self.seed).standard_normal(
            approximation.random_vector_size
        )
        return approximation.mean + approximation.apply_covariance_factorization(random_vector)


# ==================================================================================================
@dataclass(frozen=True)
class PreconditionedMcmcRunConfig(McmcRunConfig):
    """The configuration of the plain run and of the Laplace approximation it is preconditioned by.

    Attributes:
        initial_state (PriorMeanInitialGuess | PriorSampleInitialGuess | FileInitialGuess |
            ChainStateInitialGuess | LaplaceSampleInitialState): As in the plain run, or a draw from
            the approximation.
        laplace (LaplaceSettings): Eigensolver and rank of the Laplace approximation.
        map_estimate_path (Path): `.npy` file with the MAP estimate, one angle [rad] per vertex,
            e.g. the `map_estimate.npy` of a MAP run of the same problem; a relative path is
            relative to the repository. The Hessian is evaluated and the approximation centered
            there. The chain starts at `initial_state`, which is usually the same file.
    """

    initial_state: (
        PriorMeanInitialGuess
        | PriorSampleInitialGuess
        | FileInitialGuess
        | ChainStateInitialGuess
        | LaplaceSampleInitialState
    )
    laplace: LaplaceSettings
    map_estimate_path: Path

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sampler.algorithm.requires_approximation:
            raise ValueError(
                f"The algorithm {self.sampler.algorithm.name} does not use an approximation; "
                "use the plain MCMC run."
            )


# ==================================================================================================
@dataclass(frozen=True)
class LaplaceSpectrum(ArrayRecord):
    r"""The eigenvalues of the Laplace approximation.

    Attributes:
        eigenvalues (np.ndarray): Retained eigenvalues $\lambda_i$ of the misfit Hessian relative
            to the prior precision, in descending order, shape `(rank,)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {"eigenvalues": "hessian_eigenvalues.npy"}

    eigenvalues: np.ndarray


# ==================================================================================================
class PreconditionedMcmcRun(McmcRun):
    """Sample the posterior with pCN or MALA preconditioned by the Laplace approximation."""

    config_type = PreconditionedMcmcRunConfig
    name = "mcmc-laplace"
    outputs = {
        **McmcRun.outputs,
        f"results/{LaplaceSpectrum.FILE_NAMES['eigenvalues']}": "Retained eigenvalues of the "
        "misfit Hessian relative to the prior precision, descending.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        config = cast(PreconditionedMcmcRunConfig, self.config)
        return [*super().input_files(), resolve_repository_path(config.map_estimate_path)]

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
        config = cast(PreconditionedMcmcRunConfig, self.config)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        map_estimate = np.load(resolve_repository_path(config.map_estimate_path))
        logger.info(f"      {describe_array('MAP estimate [rad]', map_estimate)}")
        start = time.perf_counter()
        with steps.step(
            f"Computing {config.laplace.num_eigenvalues} eigenpairs of the misfit Hessian "
            f"({type(config.laplace.eigensolver).__name__})"
        ):
            approximation = LaplaceBuilder(config.laplace, assembled.components).build(map_estimate)
        seconds = time.perf_counter() - start
        eigenvalues = approximation.eigenpairs.eigenvalues
        logger.info(
            f"      kept {eigenvalues.size} of {config.laplace.num_eigenvalues} eigenpairs; "
            f"smallest kept eigenvalue {eigenvalues[-1]:.3e} "
            "(directions beyond it are not corrected)"
        )
        return SamplingProblem(
            log_posterior=assembled.components.to_log_posterior(),
            approximation=LowRankGaussianMeasure(approximation.gaussian),
            metrics={
                "num_eigenpairs": int(eigenvalues.size),
                "eigenvalue_max": float(eigenvalues[0]),
                "eigenvalue_min": float(eigenvalues[-1]),
                "seconds_approximation": seconds,
            },
            results=LaplaceSpectrum(eigenvalues=eigenvalues),
        )

"""Configurations of the two kinds of runs: prior investigation and MAP estimation.

A run is a pure function of its configuration: everything it depends on (parameters, seeds, data
source) is a field here, and nothing is hidden in module state. Configurations hold paths and
settings only, never arrays, so they can be serialized, hashed and compared. The ground truth of a
MAP run and the observation settings have no defaults on purpose, since studies vary them.

Classes:
    PriorParameters: Bilaplacian SPDE prior parameters.
    PriorRunConfig: Sample the prior and estimate its pointwise variance and correlation length.
    McmcRunSettings: Settings of the MCMC sampling stage.
    MapRunConfig: Generate synthetic observations from a ground truth and compute the MAP estimate.

Functions:
    reference_eikonal_settings: Eikonal forward-solver settings of the reference patient.
    reference_optimizer_strategy: Cameron-Martin L-BFGS with the loose reference tolerance.
    reference_config: The MAP run configuration reproducing the example data.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ls_bayesian.optimization.algorithms.custom_lbfgs import CustomLBFGSSettings

from bayes_cep.mcmc.builder import MCMCAlgorithmName
from bayes_cep.mesh.interpolation import (
    LinearInterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.optimization.strategies import CustomLBFGSStrategy, ScipyLBFGSBStrategy
from bayes_cep.posterior.eikonal_map import EikonalSolverSettings
from bayes_cep.preprocessing.ground_truth import (
    RealDataGroundTruthStrategy,
    SyntheticGroundTruthStrategy,
)
from bayes_cep.preprocessing.synthetic_observations import ObservationSamplingSettings
from bayes_cep.run.config import RunConfig
from bayes_cep.statistics.correlation_length import CorrelationLengthSettings

# Reference settings of the legacy single-patient MAP study. Used as defaults below; every one of
# them can be overridden in a study.
REFERENCE_RAW_DIR = Path("example_data/raw")
REFERENCE_INITIAL_SITE_IND = 12650
REFERENCE_LONGITUDINAL_VELOCITY = 1.5
REFERENCE_TRANSVERSAL_VELOCITY = 1.0
REFERENCE_PRIOR_KAPPA = 0.05
REFERENCE_PRIOR_TAU = 10.0
REFERENCE_NUM_OBSERVATIONS = 1000
REFERENCE_NOISE_VARIANCE = 1e-3
REFERENCE_GRADIENT_NORM_TOLERANCE = 100.0
REFERENCE_MAX_NUM_ITERATIONS = 1000
REFERENCE_MCMC_STEP_WIDTH = 1e-7
REFERENCE_MCMC_NUM_SAMPLES = 1000


# ==================================================================================================
def reference_eikonal_settings() -> EikonalSolverSettings:
    """Eikonal forward-solver settings of the reference patient."""
    return EikonalSolverSettings(
        initial_site_ind=REFERENCE_INITIAL_SITE_IND,
        longitudinal_velocity=REFERENCE_LONGITUDINAL_VELOCITY,
        transversal_velocity=REFERENCE_TRANSVERSAL_VELOCITY,
    )


# --------------------------------------------------------------------------------------------------
def reference_optimizer_strategy() -> CustomLBFGSStrategy:
    """Cameron-Martin L-BFGS with the loose reference gradient-norm tolerance."""
    return CustomLBFGSStrategy(
        lbfgs_settings=CustomLBFGSSettings(
            maximum_num_iterations=REFERENCE_MAX_NUM_ITERATIONS,
            gradient_norm_tolerance=REFERENCE_GRADIENT_NORM_TOLERANCE,
        )
    )


# ==================================================================================================
@dataclass(frozen=True)
class PriorParameters:
    r"""Bilaplacian SPDE prior parameters.

    Attributes:
        kappa (float): SPDE parameter $\kappa > 0$, controlling correlation length.
        tau (float): SPDE parameter $\tau > 0$, controlling marginal variance.
        seed (int): Random seed for prior sampling. Defaults to `0`.
    """

    kappa: float = REFERENCE_PRIOR_KAPPA
    tau: float = REFERENCE_PRIOR_TAU
    seed: int = 0


# ==================================================================================================
@dataclass(frozen=True)
class PriorRunConfig(RunConfig):
    """Sample a zero-mean prior and estimate its pointwise variance and correlation length.

    Needs only the mesh: no data, likelihood or optimization is involved.

    Attributes:
        raw_dir (Path): Directory containing `mesh.vtu`.
        correlation (CorrelationLengthSettings): Settings of the correlation length estimation.
        prior (PriorParameters): Prior parameters.
        num_samples (int): Number of samples for the variance and correlation length estimates.
        keep_samples (bool): Whether to store the samples themselves as a result. Off by default,
            since many runs of a study would otherwise add up to gigabytes.
    """

    raw_dir: Path
    correlation: CorrelationLengthSettings
    prior: PriorParameters = field(default_factory=PriorParameters)
    num_samples: int = 1000
    keep_samples: bool = False

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError(f"num_samples must be at least 2, got {self.num_samples}.")


# ==================================================================================================
@dataclass(frozen=True)
class McmcRunSettings:
    r"""Settings of the MCMC sampling stage, started from the MAP estimate.

    Attributes:
        algorithm (MCMCAlgorithmName): Sampling algorithm. Defaults to MALA.
        step_width (float): Proposal step width $\delta$.
        num_samples (int): Number of samples of the chain.
        seed (int): Random seed of the chain.
        chunk_size (int): Length of the on-disk chunk along the sample axis.
        log_interval (int): Number of samples between two rows of the progress table.
    """

    algorithm: MCMCAlgorithmName = MCMCAlgorithmName.MALA
    step_width: float = REFERENCE_MCMC_STEP_WIDTH
    num_samples: int = REFERENCE_MCMC_NUM_SAMPLES
    seed: int = 0
    chunk_size: int = 100
    log_interval: int = 1


# ==================================================================================================
@dataclass(frozen=True)
class MapRunConfig(RunConfig):
    """Generate synthetic observations from a ground truth, then compute the MAP estimate.

    The noise variance of `observations` is used both to generate the data and in the likelihood,
    and the eikonal forward model is shared by both, so that data and inference cannot disagree.

    Attributes:
        raw_dir (Path): Directory containing `mesh.vtu`, `basis_vecs.npy` and `fiber_field.npy`.
        ground_truth (RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy): Source of the
            ground-truth angle field. Required: it is a study dimension, not a default.
        observations (ObservationSamplingSettings): Number of observations, noise variance and
            seed of the synthetic data. Required.
        eikonal (EikonalSolverSettings): Forward-solver settings.
        interpolation (LinearInterpolationStrategy | NearestNeighborInterpolationStrategy):
            Interpolation of the vertex-based angle field onto simplices.
        prior (PriorParameters): Parameters of the inference prior. Its mean is the constant field
            built from the ground truth.
        optimizer (CustomLBFGSStrategy | ScipyLBFGSBStrategy): Optimizer backend with its model.
        mcmc (McmcRunSettings | None): Sampling of the posterior after the MAP estimate; `None`
            (the default) skips it.
    """

    raw_dir: Path
    ground_truth: RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy
    observations: ObservationSamplingSettings
    eikonal: EikonalSolverSettings = field(default_factory=reference_eikonal_settings)
    interpolation: LinearInterpolationStrategy | NearestNeighborInterpolationStrategy = field(
        default_factory=NearestNeighborInterpolationStrategy
    )
    prior: PriorParameters = field(default_factory=PriorParameters)
    optimizer: CustomLBFGSStrategy | ScipyLBFGSBStrategy = field(
        default_factory=reference_optimizer_strategy
    )
    mcmc: McmcRunSettings | None = None


# ==================================================================================================
def reference_config(
    ground_truth: RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy,
    with_mcmc: bool = False,
) -> MapRunConfig:
    """Build the MAP run configuration reproducing the example data.

    Args:
        ground_truth (RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy): Ground-truth
            source; there is deliberately no default.
        with_mcmc (bool): Whether to add the reference MCMC stage. Defaults to `False`.

    Returns:
        MapRunConfig: The reference configuration for the given ground truth.
    """
    return MapRunConfig(
        raw_dir=REFERENCE_RAW_DIR,
        ground_truth=ground_truth,
        observations=ObservationSamplingSettings(
            num_observations=REFERENCE_NUM_OBSERVATIONS,
            noise_variance=REFERENCE_NOISE_VARIANCE,
        ),
        mcmc=McmcRunSettings() if with_mcmc else None,
    )

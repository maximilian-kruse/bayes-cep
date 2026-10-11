"""The reference setup: every default parameter of the runs, in one place.

The configuration classes of the runs have no defaults for scientific parameters; presets
(`cli/run_single.py`), studies (`studies/`) and the example data all start from the configurations
built here and change what they investigate. To know which value a run uses unless stated
otherwise, read this module and nothing else.

The data locations (the raw data directory, the directory of the preprocessed data, the initial
state of a chain) are arguments, set by the study module or script that builds a config. The values
are those of the example data (`example_data/`): the single retained patient mesh, the standard
prior (kappa 0.05, tau 10), 1000 observations with noise variance 1e-3. Mesh-specific values (the
initial site of the eikonal solver, the velocities) belong to that patient.

Constants:
    NUM_OBSERVATIONS ... MCMC_NUM_SAMPLES: The reference values.

Functions:
    reference_prior_parameters: Parameters of the Bilaplacian SPDE prior.
    reference_synthetic_ground_truth: Synthetic ground truth sampled from the reference prior.
    reference_eikonal_settings: Eikonal forward-solver settings of the reference patient.
    reference_optimizer_strategy: Cameron-Martin L-BFGS with the loose reference tolerance.
    reference_correlation_length_settings: Estimation of the correlation length of the prior.
    reference_inference_problem: The posterior of the example data.
    reference_prior_config: The prior investigation run.
    reference_preprocessing_config: The preprocessing run, for a given ground truth.
    reference_synthetic_preprocessing_config: The preprocessing run with a synthetic ground truth.
    reference_patient_preprocessing_config: The preprocessing run with the real patient fibers.
    reference_map_config: The MAP run.
    reference_mcmc_config: The MCMC run, for a given initial state.
"""

from pathlib import Path

import numpy as np
from ls_bayesian.optimization.algorithms.custom_lbfgs import CustomLBFGSSettings

from bayes_cep.mcmc.builder import MCMCAlgorithmName, MCMCSettings, ZarrStorageSettings
from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.optimization.strategies import CustomLBFGSStrategy
from bayes_cep.posterior.eikonal_map import EikonalSolverSettings
from bayes_cep.posterior.prior import PriorParameters
from bayes_cep.preprocessing.ground_truth import (
    RealDataGroundTruthStrategy,
    SyntheticGroundTruthStrategy,
)
from bayes_cep.preprocessing.synthetic_observations import ObservationSamplingSettings
from bayes_cep.statistics.correlation_length import CorrelationLengthSettings
from single_runs.inference import InferenceProblemConfig
from single_runs.map import MapRunConfig
from single_runs.mcmc import McmcRunConfig
from single_runs.preprocessing import PreprocessingRunConfig
from single_runs.prior import PriorRunConfig

# Data
NUM_OBSERVATIONS = 1000
NOISE_VARIANCE = 1e-3
OBSERVATION_SEED = 0

# Eikonal forward model of the patient mesh
EIKONAL_INITIAL_SITE_INDEX = 12650
EIKONAL_LONGITUDINAL_VELOCITY = 1.5
EIKONAL_TRANSVERSAL_VELOCITY = 1.0
EIKONAL_SOLVER_TOLERANCE = 1e-6
EIKONAL_MAX_NUM_ITERATIONS = 1000
EIKONAL_MAX_VALUE = 1000.0
EIKONAL_USE_SOFT_UPDATE = True
EIKONAL_SOFTMINMAX_ORDER = 20
EIKONAL_SOFTMINMAX_CUTOFF = 0.01

# Bilaplacian SPDE prior
PRIOR_KAPPA = 0.05
PRIOR_TAU = 10.0
PRIOR_SEED = 0

# Prior investigation
PRIOR_RUN_NUM_SAMPLES = 1000
PRIOR_RUN_KEEP_SAMPLES = False
CORRELATION_MAX_DISTANCE = 40.0
CORRELATION_BOUNDARY_MARGIN = 10.0
CORRELATION_NUM_BASE_VERTICES = 200
CORRELATION_NUM_BINS = 40
CORRELATION_NOISE_FLOOR_SCALE = 2.0
CORRELATION_THRESHOLD = float(np.exp(-1.0))
CORRELATION_BURN_IN = 0
CORRELATION_SEED = 0

# MAP optimizer
MAP_MAX_NUM_ITERATIONS = 1000
MAP_GRADIENT_NORM_TOLERANCE = 100.0

# MCMC sampler. The data (many observations, tiny noise variance) make the likelihood potential
# very sharp relative to the broad prior, so large proposal steps are practically never accepted;
# the step width is a conservative starting point, not a tuned value. The acceptance rate falls
# further once the chain leaves the neighborhood of the MAP.
MCMC_ALGORITHM = MCMCAlgorithmName.MALA
MCMC_STEP_WIDTH = 1e-7
MCMC_NUM_SAMPLES = 1000
MCMC_SEED = 0
MCMC_CHUNK_SIZE = 100
MCMC_LOG_INTERVAL = 1


# ==================================================================================================
def reference_prior_parameters() -> PriorParameters:
    """Parameters of the Bilaplacian SPDE prior."""
    return PriorParameters(kappa=PRIOR_KAPPA, tau=PRIOR_TAU, seed=PRIOR_SEED)


# ==================================================================================================
def reference_eikonal_settings() -> EikonalSolverSettings:
    """Eikonal forward-solver settings of the reference patient."""
    return EikonalSolverSettings(
        initial_site_ind=EIKONAL_INITIAL_SITE_INDEX,
        longitudinal_velocity=EIKONAL_LONGITUDINAL_VELOCITY,
        transversal_velocity=EIKONAL_TRANSVERSAL_VELOCITY,
        solver_tolerance=EIKONAL_SOLVER_TOLERANCE,
        max_num_iterations=EIKONAL_MAX_NUM_ITERATIONS,
        max_value=EIKONAL_MAX_VALUE,
        use_soft_update=EIKONAL_USE_SOFT_UPDATE,
        softminmax_order=EIKONAL_SOFTMINMAX_ORDER,
        softminmax_cutoff=EIKONAL_SOFTMINMAX_CUTOFF,
    )


# ==================================================================================================
def reference_optimizer_strategy() -> CustomLBFGSStrategy:
    """Cameron-Martin L-BFGS with the loose reference gradient-norm tolerance."""
    return CustomLBFGSStrategy(
        lbfgs_settings=CustomLBFGSSettings(
            maximum_num_iterations=MAP_MAX_NUM_ITERATIONS,
            gradient_norm_tolerance=MAP_GRADIENT_NORM_TOLERANCE,
        )
    )


# ==================================================================================================
def reference_correlation_length_settings() -> CorrelationLengthSettings:
    """Estimation of the correlation length of the prior."""
    return CorrelationLengthSettings(
        max_distance=CORRELATION_MAX_DISTANCE,
        num_base_vertices=CORRELATION_NUM_BASE_VERTICES,
        num_bins=CORRELATION_NUM_BINS,
        boundary_margin=CORRELATION_BOUNDARY_MARGIN,
        noise_floor_scale=CORRELATION_NOISE_FLOOR_SCALE,
        correlation_threshold=CORRELATION_THRESHOLD,
        burn_in=CORRELATION_BURN_IN,
        seed=CORRELATION_SEED,
    )


# ==================================================================================================
def reference_inference_problem(raw_dir: Path, data_dir: Path) -> InferenceProblemConfig:
    """The posterior of the example data.

    Args:
        data_dir (Path): Directory with the data and the ground truth (the files of
            `PreprocessedData`, e.g. the `results` directory of a preprocessing run); there is
            deliberately no default.
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.

    Returns:
        InferenceProblemConfig: The reference problem for the given data.
    """
    return InferenceProblemConfig(
        raw_dir=raw_dir,
        data_dir=data_dir,
        prior=reference_prior_parameters(),
        eikonal=reference_eikonal_settings(),
        interpolation=NearestNeighborInterpolationStrategy(),
    )


# ==================================================================================================
def reference_synthetic_ground_truth() -> SyntheticGroundTruthStrategy:
    """Synthetic ground truth: a sample of the reference prior."""
    return SyntheticGroundTruthStrategy(prior=reference_prior_parameters())


# ==================================================================================================
def reference_prior_config(raw_dir: Path) -> PriorRunConfig:
    """The prior investigation run.

    Args:
        raw_dir (Path): Directory containing `mesh.vtu`; there is deliberately no default.
    """
    return PriorRunConfig(
        raw_dir=raw_dir,
        correlation=reference_correlation_length_settings(),
        prior=reference_prior_parameters(),
        num_samples=PRIOR_RUN_NUM_SAMPLES,
        keep_samples=PRIOR_RUN_KEEP_SAMPLES,
    )


# ==================================================================================================
def reference_preprocessing_config(
    raw_dir: Path,
    ground_truth: RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy,
) -> PreprocessingRunConfig:
    """The preprocessing run of the example data.

    Args:
        ground_truth (RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy): Ground-truth
            source; there is deliberately no default.
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.

    Returns:
        PreprocessingRunConfig: The reference configuration for the given ground truth.
    """
    return PreprocessingRunConfig(
        raw_dir=raw_dir,
        ground_truth=ground_truth,
        observations=ObservationSamplingSettings(
            num_observations=NUM_OBSERVATIONS,
            noise_variance=NOISE_VARIANCE,
            seed=OBSERVATION_SEED,
        ),
        eikonal=reference_eikonal_settings(),
        interpolation=NearestNeighborInterpolationStrategy(),
    )


# ==================================================================================================
def reference_synthetic_preprocessing_config(raw_dir: Path) -> PreprocessingRunConfig:
    """The preprocessing run with the reference synthetic ground truth.

    Args:
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.
    """
    return reference_preprocessing_config(raw_dir, reference_synthetic_ground_truth())


# ==================================================================================================
def reference_patient_preprocessing_config(raw_dir: Path) -> PreprocessingRunConfig:
    """The preprocessing run with the fiber field of the real patient as ground truth.

    Args:
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.
    """
    return reference_preprocessing_config(raw_dir, RealDataGroundTruthStrategy())


# ==================================================================================================
def reference_map_config(raw_dir: Path, data_dir: Path) -> MapRunConfig:
    """The MAP run of the example data.

    Args:
        data_dir (Path): Directory with the data and the ground truth (the files of
            `PreprocessedData`, e.g. the `results` directory of a preprocessing run); there is
            deliberately no default.
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.

    Returns:
        MapRunConfig: The reference configuration for the given data.
    """
    return MapRunConfig(
        problem=reference_inference_problem(raw_dir, data_dir),
        optimizer=reference_optimizer_strategy(),
    )


# ==================================================================================================
def reference_mcmc_config(
    raw_dir: Path,
    data_dir: Path,
    initial_state_path: Path,
) -> McmcRunConfig:
    """The MCMC run of the example data.

    Args:
        data_dir (Path): Directory with the data and the ground truth (the files of
            `PreprocessedData`, e.g. the `results` directory of a preprocessing run); there is
            deliberately no default.
        initial_state_path (Path): `.npy` file with the state the chain starts at; there is
            deliberately no default. The example data starts at the MAP estimate.
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.

    Returns:
        McmcRunConfig: The reference configuration for the given data.
    """
    return McmcRunConfig(
        problem=reference_inference_problem(raw_dir, data_dir),
        initial_state_path=initial_state_path,
        sampler=MCMCSettings(
            algorithm=MCMC_ALGORITHM,
            step_width=MCMC_STEP_WIDTH,
            storage=ZarrStorageSettings(chunk_size=MCMC_CHUNK_SIZE),
        ),
        num_samples=MCMC_NUM_SAMPLES,
        seed=MCMC_SEED,
        log_interval=MCMC_LOG_INTERVAL,
    )

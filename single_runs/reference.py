"""The reference setup: every default parameter of the runs, in one place.

The configuration classes of the runs have no defaults for scientific parameters; presets
(`bcep-single`), studies (`studies/`) and the example data all start from the configurations
built here and change what they investigate. To know which value a run uses unless stated
otherwise, read this module and nothing else.

The data locations (the raw data directory, the directory of the preprocessed data, the initial
state of a chain) are arguments, set by the study module or script that builds a config. The values
are those of the example data (`example_data/`): the single retained patient mesh, the standard
prior (kappa 0.1, tau 5), 1000 observations with noise variance 1e-3. Mesh-specific values (the
initial site of the eikonal solver, the velocities) belong to that patient.

Constants:
    The reference values for all data primitives.

Functions:
    reference_prior_parameters: Parameters of the Bilaplacian SPDE prior.
    reference_synthetic_ground_truth: Synthetic ground truth sampled from the reference prior.
    reference_eikonal_settings: Eikonal forward-solver settings of the reference patient.
    reference_optimizer_strategy: Cameron-Martin L-BFGS with the loose reference tolerance.
    reference_correlation_length_settings: Estimation of the correlation length of the prior.
    reference_moment_fit_settings: Estimation of kappa and tau of a patient fiber field.
    reference_prior_moment_fit_config: The prior parameter estimation run.
    reference_prior_recovery_config: The recovery test of the prior parameter estimation.
    reference_inference_problem: The posterior of the example data.
    reference_prior_config: The prior investigation run.
    reference_preprocessing_config: The preprocessing run, for a given ground truth.
    reference_synthetic_preprocessing_config: The preprocessing run with a synthetic ground truth.
    reference_patient_preprocessing_config: The preprocessing run with the real patient fibers.
    reference_smoothed_patient_preprocessing_config: The preprocessing run with a stored smoothed
        patient field.
    reference_map_config: The MAP run.
    reference_mcmc_config: The MCMC run, for a given initial state.
    reference_laplace_config: The Laplace run, for a given MAP estimate.
    reference_preconditioned_mcmc_config: The Laplace-preconditioned MCMC run.
"""

from pathlib import Path

import numpy as np
from ls_bayesian.lowrank.eigensolvers import RandomizedDoublePassSolver
from ls_bayesian.optimization.algorithms.custom_lbfgs import CustomLBFGSSettings

from bayes_cep.laplace.builder import LaplaceSettings
from bayes_cep.mcmc.builder import MCMCAlgorithmName, MCMCSettings, ZarrStorageSettings
from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.optimization.initial_guess import FileInitialGuess, PriorMeanInitialGuess
from bayes_cep.optimization.strategies import CustomLBFGSStrategy
from bayes_cep.posterior.eikonal_map import EikonalSolverSettings
from bayes_cep.posterior.prior import PriorParameters
from bayes_cep.preprocessing.ground_truth import (
    RealDataGroundTruthStrategy,
    SmoothedFieldGroundTruthStrategy,
    SyntheticGroundTruthStrategy,
)
from bayes_cep.preprocessing.synthetic_observations import ObservationSamplingSettings
from bayes_cep.statistics.correlation_length import CorrelationLengthSettings
from bayes_cep.statistics.prior_moment_fit import DistanceMetric, MomentFitSettings
from single_runs.inference import InferenceProblemConfig
from single_runs.laplace import LaplaceRunConfig
from single_runs.map import MapRunConfig
from single_runs.mcmc import ChainDiagnosticsSettings, McmcRunConfig
from single_runs.mcmc_laplace import PreconditionedMcmcRunConfig
from single_runs.preprocessing import PreprocessingRunConfig
from single_runs.prior import PriorRunConfig
from single_runs.prior_fit import PriorMomentFitRunConfig
from single_runs.prior_recovery import PriorRecoveryRunConfig

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
PRIOR_KAPPA = 0.1
PRIOR_TAU = 5.0
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

# Estimation of kappa and tau from a patient fiber field (simulated moments). The distances are in
# the units of the patient meshes; the outer two bins are not fitted.
MOMENT_FIT_DISTANCE_BIN_EDGES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1000.0)
MOMENT_FIT_NUM_FITTED_BINS = 6
MOMENT_FIT_NUM_PAIRS = 400_000
MOMENT_FIT_NUM_SOURCE_SIMPLICES = 400
MOMENT_FIT_CONCENTRATION_WEIGHT = 6.0
MOMENT_FIT_NUM_PRIOR_SAMPLES = 8
MOMENT_FIT_SEED = 0
MOMENT_FIT_KAPPAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8)
MOMENT_FIT_TAUS = (0.3, 0.45, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0)

# MAP optimizer
MAP_MAX_NUM_ITERATIONS = 1000
MAP_GRADIENT_NORM_TOLERANCE = 1

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
MCMC_BURN_IN = 100
MCMC_MAX_LAG = 200
MCMC_PLOTTED_COMPONENTS = (0, 42, 100, 1000)

# Laplace-preconditioned samplers. With the exact Gaussian posterior as approximation they would
# be rejection-free at a step width near 1, but the posterior is far from Gaussian over the
# 15820 dimensions: measured from the MAP estimate, the pCN log acceptance is about -65 at 0.5,
# -3 at 0.2 and 0 at 0.05. The values are conservative starting points, not tuned ones.
MCMC_PRECONDITIONED_STEP_WIDTHS = {
    MCMCAlgorithmName.PCN_PRECONDITIONED: 0.05,
    MCMCAlgorithmName.PMALA: 0.05,
}

# Laplace approximation. The misfit Hessian has at most one nonzero eigenvalue per observation, so
# the rank is at most the number of observations (1000); the log shows the smallest kept
# eigenvalue, and directions beyond it are not corrected. With 200 it is still above 1.
LAPLACE_NUM_EIGENVALUES = 500
LAPLACE_OVERSAMPLING = 20
LAPLACE_EIGENSOLVER_SEED = 0
LAPLACE_MINIMUM_EIGENVALUE = 1e-6
LAPLACE_NUM_SAMPLES = 1000
LAPLACE_SAMPLE_SEED = 0


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
def reference_moment_fit_settings(
    smoothing_scale: float, distance_metric: DistanceMetric
) -> MomentFitSettings:
    """Estimation of kappa and tau of a patient fiber field.

    Args:
        smoothing_scale (float): Scale of the smoothing of the prior samples; `0` for a raw field,
            the scale of the field for a smoothed one. There is deliberately no default.
        distance_metric (DistanceMetric): Distance that bins the simplex pairs; deliberately no
            default.
    """
    return MomentFitSettings(
        distance_bin_edges=MOMENT_FIT_DISTANCE_BIN_EDGES,
        num_fitted_bins=MOMENT_FIT_NUM_FITTED_BINS,
        num_pairs=MOMENT_FIT_NUM_PAIRS,
        distance_metric=distance_metric,
        num_source_simplices=MOMENT_FIT_NUM_SOURCE_SIMPLICES,
        concentration_weight=MOMENT_FIT_CONCENTRATION_WEIGHT,
        num_prior_samples=MOMENT_FIT_NUM_PRIOR_SAMPLES,
        seed=MOMENT_FIT_SEED,
        smoothing_scale=smoothing_scale,
    )


# ==================================================================================================
def reference_prior_moment_fit_config(
    raw_dir: Path, smoothing_scale: float, distance_metric: DistanceMetric
) -> PriorMomentFitRunConfig:
    """The prior parameter estimation run on the reference grid.

    Args:
        raw_dir (Path): Patient directory with the mesh and the fiber field; there is deliberately
            no default.
        smoothing_scale (float): `0` for the raw field, else the scale of the smoothed field.
        distance_metric (DistanceMetric): Distance that bins the simplex pairs.

    Returns:
        PriorMomentFitRunConfig: The reference configuration for the given field.
    """
    return PriorMomentFitRunConfig(
        raw_dir=raw_dir,
        fit=reference_moment_fit_settings(smoothing_scale, distance_metric),
        kappas=MOMENT_FIT_KAPPAS,
        taus=MOMENT_FIT_TAUS,
        interpolation=NearestNeighborInterpolationStrategy(),
    )


# ==================================================================================================
def reference_prior_recovery_config(
    raw_dir: Path,
    true_prior: PriorParameters,
    smoothing_scale: float,
    distance_metric: DistanceMetric,
) -> PriorRecoveryRunConfig:
    """The recovery test of the prior parameter estimation, on the reference grid.

    Args:
        raw_dir (Path): Patient directory with the mesh; there is deliberately no default.
        true_prior (PriorParameters): Parameters and seed of the fitted prior sample; the seed must
            differ from `MOMENT_FIT_SEED`.
        smoothing_scale (float): `0` for no smoothing, else the scale the sample is smoothed with.
        distance_metric (DistanceMetric): Distance that bins the simplex pairs.

    Returns:
        PriorRecoveryRunConfig: The reference configuration for the given truth.
    """
    return PriorRecoveryRunConfig(
        raw_dir=raw_dir,
        true_prior=true_prior,
        fit=reference_moment_fit_settings(smoothing_scale, distance_metric),
        kappas=MOMENT_FIT_KAPPAS,
        taus=MOMENT_FIT_TAUS,
        interpolation=NearestNeighborInterpolationStrategy(),
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
def reference_smoothed_patient_preprocessing_config(
    raw_dir: Path, angle_field_path: Path
) -> PreprocessingRunConfig:
    """The preprocessing run with a stored smoothed angle field of the patient as ground truth.

    Args:
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.
        angle_field_path (Path): `.npy` file with the smoothed angle [rad] per vertex (written by
            `scripts/smooth_patient_fibers.py`); there is deliberately no default.
    """
    return reference_preprocessing_config(
        raw_dir, SmoothedFieldGroundTruthStrategy(path=angle_field_path)
    )


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
        initial_guess=PriorMeanInitialGuess(),
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
        initial_state=FileInitialGuess(path=initial_state_path),
        sampler=MCMCSettings(
            algorithm=MCMC_ALGORITHM,
            step_width=MCMC_STEP_WIDTH,
            storage=ZarrStorageSettings(chunk_size=MCMC_CHUNK_SIZE),
        ),
        diagnostics=ChainDiagnosticsSettings(
            burn_in=MCMC_BURN_IN,
            max_lag=MCMC_MAX_LAG,
            plotted_components=MCMC_PLOTTED_COMPONENTS,
        ),
        num_samples=MCMC_NUM_SAMPLES,
        seed=MCMC_SEED,
        log_interval=MCMC_LOG_INTERVAL,
    )


# ==================================================================================================
def reference_laplace_config(
    raw_dir: Path,
    data_dir: Path,
    map_estimate_path: Path,
) -> LaplaceRunConfig:
    """The Laplace run of the example data.

    Args:
        data_dir (Path): Directory with the data and the ground truth (the files of
            `PreprocessedData`, e.g. the `results` directory of a preprocessing run); there is
            deliberately no default.
        map_estimate_path (Path): `.npy` file with the MAP estimate the approximation is centered
            at; there is deliberately no default.
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.

    Returns:
        LaplaceRunConfig: The reference configuration for the given data.
    """
    return LaplaceRunConfig(
        problem=reference_inference_problem(raw_dir, data_dir),
        map_estimate_path=map_estimate_path,
        laplace=LaplaceSettings(
            eigensolver=RandomizedDoublePassSolver(
                oversampling=LAPLACE_OVERSAMPLING, seed=LAPLACE_EIGENSOLVER_SEED
            ),
            num_eigenvalues=LAPLACE_NUM_EIGENVALUES,
            minimum_eigenvalue=LAPLACE_MINIMUM_EIGENVALUE,
            sample_seed=LAPLACE_SAMPLE_SEED,
        ),
        num_samples=LAPLACE_NUM_SAMPLES,
    )


# ==================================================================================================
def reference_preconditioned_mcmc_config(
    raw_dir: Path,
    data_dir: Path,
    map_estimate_path: Path,
    algorithm: MCMCAlgorithmName,
) -> PreconditionedMcmcRunConfig:
    """The Laplace-preconditioned MCMC run of the example data.

    The chain starts at the MAP estimate, which is also the center of the approximation.

    Args:
        data_dir (Path): Directory with the data and the ground truth (the files of
            `PreprocessedData`, e.g. the `results` directory of a preprocessing run); there is
            deliberately no default.
        map_estimate_path (Path): `.npy` file with the MAP estimate the approximation is centered
            at and the chain starts at; there is deliberately no default.
        raw_dir (Path): Directory containing the raw data; there is deliberately no default.
        algorithm (MCMCAlgorithmName): `PCN_PRECONDITIONED` or `PMALA`; there is deliberately no
            default. The step width is the reference value of that algorithm.

    Returns:
        PreconditionedMcmcRunConfig: The reference configuration for the given data.

    Raises:
        ValueError: If `algorithm` is not a preconditioned one.
    """
    if algorithm not in MCMC_PRECONDITIONED_STEP_WIDTHS:
        raise ValueError(
            f"algorithm must be one of {[name.name for name in MCMC_PRECONDITIONED_STEP_WIDTHS]}, "
            f"got {algorithm.name}."
        )
    plain = reference_mcmc_config(raw_dir, data_dir, map_estimate_path)
    return PreconditionedMcmcRunConfig(
        problem=plain.problem,
        initial_state=plain.initial_state,
        sampler=MCMCSettings(
            algorithm=algorithm,
            step_width=MCMC_PRECONDITIONED_STEP_WIDTHS[algorithm],
            storage=plain.sampler.storage,
        ),
        diagnostics=plain.diagnostics,
        num_samples=plain.num_samples,
        seed=plain.seed,
        log_interval=plain.log_interval,
        laplace=reference_laplace_config(raw_dir, data_dir, map_estimate_path).laplace,
        map_estimate_path=map_estimate_path,
    )

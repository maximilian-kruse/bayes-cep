"""The inference problem shared by the MAP and the MCMC run.

Both runs infer the same posterior from the same preprocessed data (see
[`PreprocessingRun`][single_runs.preprocessing.PreprocessingRun]), so the settings of that posterior
and its assembly live here once; the runs themselves only optimize or sample it. The configuration
is a pure description; `assemble_posterior` loads the files it names and builds the posterior.

Classes:
    InferenceProblemConfig: Which data and which model the posterior is built from.
    AssembledPosterior: The posterior and the data it was built from.

Functions:
    assemble_posterior: Load the mesh and the preprocessed data and build the posterior.
"""

from dataclasses import dataclass
from pathlib import Path

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.posterior.posterior import LogPosterior

from bayes_cep.mesh.interpolation import (
    LinearInterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.posterior.builder import PosteriorBuilder, PosteriorSettings
from bayes_cep.posterior.eikonal_map import EikonalSolverSettings
from bayes_cep.posterior.likelihood import LikelihoodSettings
from bayes_cep.posterior.prior import PriorSettings, SpdePriorParameters
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.run.progress import StepReporter
from single_runs.data import PreprocessedData, RawDataDirectory


# ==================================================================================================
@dataclass(frozen=True)
class InferenceProblemConfig:
    """Which data and which model the posterior of an inference run is built from.

    Attributes:
        raw_dir (Path): Directory containing `mesh.vtu` and `basis_vecs.npy`. It must be the one
            the preprocessing run used; `assemble_posterior` checks that the sizes agree.
        data_dir (Path): Directory with the data the posterior is conditioned on and the ground
            truth (the files of `PreprocessedData`). It can come from anywhere, e.g. the `results`
            directory of a preprocessing run.
        prior (SpdePriorParameters): Parameters of the inference prior. Its mean is the prior mean
            of the preprocessed data.
        eikonal (EikonalSolverSettings): Forward-solver settings of the inference.
        interpolation (LinearInterpolationStrategy | NearestNeighborInterpolationStrategy):
            Interpolation of the vertex-based angle field onto simplices.
    """

    raw_dir: Path
    data_dir: Path
    prior: SpdePriorParameters
    eikonal: EikonalSolverSettings
    interpolation: LinearInterpolationStrategy | NearestNeighborInterpolationStrategy

    # ----------------------------------------------------------------------------------------------
    def input_files(self) -> list[Path]:
        """The raw files and the preprocessed data the posterior is built from."""
        raw_data = RawDataDirectory.from_configured_path(self.raw_dir)
        return [
            raw_data.mesh_path,
            raw_data.basis_vectors_path,
            *PreprocessedData.file_paths(resolve_repository_path(self.data_dir)),
        ]


# ==================================================================================================
@dataclass(frozen=True)
class AssembledPosterior:
    """The posterior and the data it was built from.

    Attributes:
        log_posterior (LogPosterior): The log posterior of the preprocessed data; its `prior` is
            the prior.
        data (PreprocessedData): The data the posterior was built from.
    """

    log_posterior: LogPosterior
    data: PreprocessedData


# ==================================================================================================
def assemble_posterior(
    problem: InferenceProblemConfig, steps: StepReporter, logger: BaseLogger
) -> AssembledPosterior:
    """Load the mesh and the preprocessed data and build the posterior.

    Args:
        problem (InferenceProblemConfig): Which data and which model to use.
        steps (StepReporter): Reporter of the run, for the numbered progress steps.
        logger (BaseLogger): Logger of the run.

    Returns:
        AssembledPosterior: The posterior and the data.

    Raises:
        ValueError: If the preprocessed data does not belong to the mesh in `raw_dir`.
    """
    raw_data = RawDataDirectory.from_configured_path(problem.raw_dir)
    data_dir = resolve_repository_path(problem.data_dir)
    with steps.step(f"Loading the mesh and basis vectors from {raw_data.path}"):
        mesh = raw_data.load_mesh()
        basis_vectors = raw_data.load_basis_vectors()
    logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")
    with steps.step(f"Loading the preprocessed data from {data_dir}"):
        data = PreprocessedData.load(data_dir)
    if data.prior_mean.shape != (mesh.n_points,) or data.ground_truth.shape != (mesh.n_points,):
        raise ValueError(
            f"The data in {data_dir} has fields of shape {data.prior_mean.shape} and "
            f"{data.ground_truth.shape}, but the mesh in {raw_data.path} has {mesh.n_points} "
            "vertices: the raw data is not the one the preprocessing run used."
        )
    with steps.step("Building the posterior"):
        log_posterior = PosteriorBuilder(
            PosteriorSettings(
                mesh=mesh,
                basis_vectors=basis_vectors,
                prior_settings=PriorSettings(mean_vector=data.prior_mean, parameters=problem.prior),
                eikonal_settings=problem.eikonal,
                likelihood_settings=LikelihoodSettings(
                    num_vertices=mesh.n_points,
                    observed_vertex_indices=data.observed_vertex_indices,
                    data_vector=data.observed_activation_times,
                    noise_variance=data.noise_variance,
                ),
            ),
            interpolation_strategy=problem.interpolation,
        ).build()
    return AssembledPosterior(log_posterior, data)

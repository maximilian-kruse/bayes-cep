"""The inference problem shared by the MAP and the MCMC run.

Both runs infer the same posterior from the same preprocessed data (see
[`PreprocessingRun`][single_runs.preprocessing.PreprocessingRun]), so the settings of that posterior
and its assembly live here once; the runs themselves only optimize or sample it.

Classes:
    InferenceProblemConfig: Which data and which model the posterior is built from.
    AssembledPosterior: The posterior with the prior builder and the data it was built from.

Functions:
    reference_inference_problem: The inference problem of the example data.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.posterior.posterior import LogPosterior

from bayes_cep.mesh.interpolation import (
    LinearInterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.posterior.builder import PosteriorBuilder, PosteriorSettings
from bayes_cep.posterior.eikonal_map import EikonalSolverSettings
from bayes_cep.posterior.likelihood import LikelihoodSettings
from bayes_cep.posterior.prior import PriorSettings
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.run.progress import StepReporter
from single_runs.preprocessing import (
    REFERENCE_RAW_DIR,
    PreprocessedData,
    reference_eikonal_settings,
)
from single_runs.prior import PriorParameters


# ==================================================================================================
@dataclass(frozen=True)
class AssembledPosterior:
    """The posterior with the prior builder and the data it was built from.

    Attributes:
        posterior_builder (PosteriorBuilder): The builder; `posterior_builder.prior` is the prior.
        log_posterior (LogPosterior): The log posterior of the preprocessed data.
        data (PreprocessedData): The data the posterior was built from.
    """

    posterior_builder: PosteriorBuilder
    log_posterior: LogPosterior
    data: PreprocessedData


# ==================================================================================================
@dataclass(frozen=True)
class InferenceProblemConfig:
    """Which data and which model the posterior of an inference run is built from.

    Attributes:
        raw_dir (Path): Directory containing `mesh.vtu` and `basis_vecs.npy`. It must be the one
            the preprocessing run used.
        preprocessing_dir (Path): Run directory of the preprocessing run that produced the data.
        prior (PriorParameters): Parameters of the inference prior. Its mean is the prior mean of
            the preprocessed data.
        eikonal (EikonalSolverSettings): Forward-solver settings of the inference.
        interpolation (LinearInterpolationStrategy | NearestNeighborInterpolationStrategy):
            Interpolation of the vertex-based angle field onto simplices.
    """

    raw_dir: Path
    preprocessing_dir: Path
    prior: PriorParameters = field(default_factory=PriorParameters)
    eikonal: EikonalSolverSettings = field(default_factory=reference_eikonal_settings)
    interpolation: LinearInterpolationStrategy | NearestNeighborInterpolationStrategy = field(
        default_factory=NearestNeighborInterpolationStrategy
    )

    # ----------------------------------------------------------------------------------------------
    @property
    def preprocessed_results_dir(self) -> Path:
        """The `results/` folder of the preprocessing run."""
        return resolve_repository_path(self.preprocessing_dir) / "results"

    # ----------------------------------------------------------------------------------------------
    def input_files(self) -> list[Path]:
        """The raw files and the preprocessed data the posterior is built from."""
        raw_dir = resolve_repository_path(self.raw_dir)
        return [
            raw_dir / "mesh.vtu",
            raw_dir / "basis_vecs.npy",
            *PreprocessedData.file_paths(self.preprocessed_results_dir),
        ]

    # ----------------------------------------------------------------------------------------------
    def assemble_posterior(self, steps: StepReporter, logger: BaseLogger) -> AssembledPosterior:
        """Load the mesh and the preprocessed data and build the posterior.

        Args:
            steps (StepReporter): Reporter of the run, for the numbered progress steps.
            logger (BaseLogger): Logger of the run.

        Returns:
            AssembledPosterior: The posterior, its prior builder and the data.
        """
        raw_dir = resolve_repository_path(self.raw_dir)
        with steps.step(f"Loading the mesh and basis vectors from {raw_dir}"):
            mesh = load_pyvista_mesh(raw_dir / "mesh.vtu")
            basis_vectors = np.load(raw_dir / "basis_vecs.npy")
        logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")

        with steps.step(f"Loading the preprocessed data from {self.preprocessed_results_dir}"):
            data = PreprocessedData.load(self.preprocessed_results_dir)

        with steps.step("Building the posterior"):
            posterior_builder = PosteriorBuilder(
                PosteriorSettings(
                    mesh=mesh,
                    basis_vectors=basis_vectors,
                    prior_settings=PriorSettings(
                        mean_vector=data.prior_mean,
                        kappa=self.prior.kappa,
                        tau=self.prior.tau,
                        seed=self.prior.seed,
                    ),
                    eikonal_settings=self.eikonal,
                    likelihood_settings=LikelihoodSettings(
                        num_vertices=mesh.n_points,
                        observed_vertex_indices=data.observed_vertex_indices,
                        data_vector=data.observed_activation_times,
                        noise_variance=data.noise_variance,
                    ),
                ),
                interpolation_strategy=self.interpolation,
            )
            log_posterior = posterior_builder.build()
        return AssembledPosterior(posterior_builder, log_posterior, data)


# ==================================================================================================
def reference_inference_problem(preprocessing_dir: Path) -> InferenceProblemConfig:
    """The inference problem of the example data.

    Args:
        preprocessing_dir (Path): Run directory of the preprocessing run that produced the data;
            there is deliberately no default.

    Returns:
        InferenceProblemConfig: The reference problem for the given data.
    """
    return InferenceProblemConfig(raw_dir=REFERENCE_RAW_DIR, preprocessing_dir=preprocessing_dir)

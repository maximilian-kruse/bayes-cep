"""Preprocessing run: ground truth, prior mean and synthetic observations for the inference runs.

The preprocessing run is the only place where data is generated. The MAP and MCMC runs read its
run directory and nothing else of the data side, so several inference runs (and studies) share
exactly the same data.

Classes:
    PreprocessedData: The data the inference runs work on, with its files.
    PreprocessingRunConfig: Build a ground truth and generate synthetic observations from it.
    PreprocessingRun: The preprocessing run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.mesh.interpolation import (
    LinearInterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.mesh.plotting import render_cell_field
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap, EikonalSolverSettings
from bayes_cep.preprocessing.ground_truth import (
    RealDataGroundTruthStrategy,
    SyntheticGroundTruthStrategy,
)
from bayes_cep.preprocessing.prior_mean import build_constant_prior_mean
from bayes_cep.preprocessing.synthetic_observations import (
    ObservationSamplingSettings,
    generate_synthetic_observations,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import wrap_axial_angles


# ==================================================================================================
@dataclass(frozen=True)
class PreprocessedData:
    """The data the inference runs work on, as written by the preprocessing run.

    Attributes:
        ground_truth (np.ndarray): Ground-truth fiber angle [rad] per vertex.
        prior_mean (np.ndarray): Constant prior mean [rad] per vertex: the axial mean of the ground
            truth.
        observed_vertex_indices (np.ndarray): Indices of the observed vertices.
        observed_activation_times (np.ndarray): Noisy activation times at those vertices.
        noise_variance (float): Variance of the noise of the observations; the likelihood of the
            inference uses the same value, so that data and inference cannot disagree.
    """

    ground_truth: np.ndarray
    prior_mean: np.ndarray
    observed_vertex_indices: np.ndarray
    observed_activation_times: np.ndarray
    noise_variance: float

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def file_paths(data_dir: Path) -> list[Path]:
        """The files of the data in `data_dir`, the result directory of a preprocessing run."""
        names = (
            "ground_truth_angle_field.npy",
            "prior_mean_angle_field.npy",
            "observed_vertex_indices.npy",
            "observed_activation_times.npy",
            "observation_noise_variance.npy",
        )
        return [data_dir / name for name in names]

    # ----------------------------------------------------------------------------------------------
    def save(self, data_dir: Path) -> None:
        """Write the data into `data_dir`, which is created if missing."""
        data_dir.mkdir(parents=True, exist_ok=True)
        arrays = (
            self.ground_truth,
            self.prior_mean,
            self.observed_vertex_indices,
            self.observed_activation_times,
            np.array(self.noise_variance),
        )
        for path, array in zip(self.file_paths(data_dir), arrays, strict=True):
            np.save(path, array)

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, data_dir: Path) -> Self:
        """Read the data from `data_dir`, the result directory of a preprocessing run."""
        (
            ground_truth_path,
            prior_mean_path,
            vertex_indices_path,
            activation_times_path,
            noise_variance_path,
        ) = cls.file_paths(data_dir)
        return cls(
            ground_truth=np.load(ground_truth_path),
            prior_mean=np.load(prior_mean_path),
            observed_vertex_indices=np.load(vertex_indices_path),
            observed_activation_times=np.load(activation_times_path),
            noise_variance=float(np.load(noise_variance_path)),
        )


# ==================================================================================================
@dataclass(frozen=True)
class PreprocessingRunConfig(RunConfig):
    """Build a ground truth and generate synthetic observations from it.

    There are no defaults: the reference values are in `single_runs.reference`, and studies vary
    them from there.

    Attributes:
        raw_dir (Path): Directory containing `mesh.vtu`, `basis_vecs.npy` and `fiber_field.npy`.
        ground_truth (RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy): Source of the
            ground-truth angle field.
        observations (ObservationSamplingSettings): Number of observations, noise variance and
            seed of the synthetic data.
        eikonal (EikonalSolverSettings): Forward-solver settings that generate the data.
        interpolation (LinearInterpolationStrategy | NearestNeighborInterpolationStrategy):
            Interpolation of the vertex-based angle field onto simplices.
    """

    raw_dir: Path
    ground_truth: RealDataGroundTruthStrategy | SyntheticGroundTruthStrategy
    observations: ObservationSamplingSettings
    eikonal: EikonalSolverSettings
    interpolation: LinearInterpolationStrategy | NearestNeighborInterpolationStrategy


# ==================================================================================================
class PreprocessingRun(Run[PreprocessingRunConfig]):
    """Build the ground truth and the prior mean, and generate the synthetic observations."""

    outputs = {
        "results/ground_truth_angle_field.npy": "Ground-truth fiber angle [rad] per vertex.",
        "results/prior_mean_angle_field.npy": "Constant prior mean [rad] per vertex: the axial "
        "mean of the ground truth.",
        "results/observed_vertex_indices.npy": "Indices of the observed vertices.",
        "results/observed_activation_times.npy": "Noisy synthetic activation times at those "
        "vertices.",
        "results/observation_noise_variance.npy": "Variance of the observation noise (scalar).",
        "metrics.json": "num_observed_vertices and prior_mean_angle [rad].",
        "plots/": "ground_truth.png (study report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        raw_dir = resolve_repository_path(self.config.raw_dir)
        return [raw_dir / name for name in ("mesh.vtu", "basis_vecs.npy", "fiber_field.npy")]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        data = PreprocessedData.load(self.results_dir(run_dir))

        mesh = load_pyvista_mesh(resolve_repository_path(self.config.raw_dir) / "mesh.vtu")
        connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
        to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(
            mesh.points, connectivity
        )
        render_cell_field(
            mesh,
            wrap_axial_angles(to_simplices @ data.ground_truth),
            plots_dir / "ground_truth.png",
            "ground truth",
            "angle [rad]",
            "hsv",
            (-np.pi / 2, np.pi / 2),
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)

        raw_dir = resolve_repository_path(config.raw_dir)
        with steps.step(f"Loading the mesh and basis vectors from {raw_dir}"):
            mesh = load_pyvista_mesh(raw_dir / "mesh.vtu")
            basis_vectors = np.load(raw_dir / "basis_vecs.npy")
        logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")

        strategy_name = type(config.ground_truth).__name__
        with steps.step(f"Building ground truth and prior mean ({strategy_name})"):
            fiber_field = np.load(raw_dir / "fiber_field.npy")
            ground_truth = config.ground_truth.build(mesh, fiber_field, basis_vectors)
            prior_mean = build_constant_prior_mean(ground_truth)
        logger.info(f"      {describe_array('ground truth [rad]', ground_truth)}")
        logger.info(f"      prior mean angle: {prior_mean[0]:.4f} rad")

        with steps.step("Generating synthetic observations"):
            forward_map = EikonalParameterToSolutionMap(
                mesh, basis_vectors, config.eikonal, config.interpolation
            )
            observed_vertex_indices, observed_activation_times = generate_synthetic_observations(
                forward_map, ground_truth, config.observations
            )
        logger.info(
            f"      observed {observed_vertex_indices.shape[0]} of {mesh.n_points} vertices"
        )

        PreprocessedData(
            ground_truth=ground_truth,
            prior_mean=prior_mean,
            observed_vertex_indices=observed_vertex_indices,
            observed_activation_times=observed_activation_times,
            noise_variance=config.observations.noise_variance,
        ).save(self.results_dir(run_dir))
        return {
            "num_observed_vertices": int(observed_vertex_indices.shape[0]),
            "prior_mean_angle": float(prior_mean[0]),
        }

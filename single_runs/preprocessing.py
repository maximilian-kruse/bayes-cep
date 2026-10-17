"""Preprocessing run: ground truth, prior mean and synthetic observations for the inference runs.

The preprocessing run is the only place where data is generated. The MAP and MCMC runs read its
run directory and nothing else of the data side, so several inference runs (and studies) share
exactly the same data.

Classes:
    PreprocessingRunConfig: Build a ground truth and generate synthetic observations from it.
    PreprocessingRun: The preprocessing run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.mesh.interpolation import (
    LinearInterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap, EikonalSolverSettings
from bayes_cep.postprocessing.mesh_plots import render_axial_angle_field
from bayes_cep.preprocessing.ground_truth import (
    RealDataGroundTruthStrategy,
    SyntheticGroundTruthStrategy,
)
from bayes_cep.preprocessing.synthetic_observations import (
    ObservationSamplingSettings,
    generate_synthetic_observations,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, PLOTS_DIR_NAME
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.data import PreprocessedData, RawDataDirectory

GROUND_TRUTH_PLOT_FILE_NAME = "ground_truth.pdf"
_FILES = PreprocessedData.FILE_NAMES  # the names of the result files, defined once there


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

    config_type = PreprocessingRunConfig
    name = "preprocessing"
    outputs = {
        f"results/{_FILES['ground_truth']}": "Ground-truth fiber angle [rad] per vertex.",
        f"results/{_FILES['prior_mean']}": "Prior mean [rad] per vertex: zero.",
        f"results/{_FILES['observed_vertex_indices']}": "Indices of the observed vertices.",
        f"results/{_FILES['observed_activation_times']}": "Noisy synthetic activation times at "
        "those vertices.",
        f"results/{_FILES['noise_variance']}": "Variance of the observation noise (scalar).",
        METRICS_RECORD: "num_observed_vertices and prior_mean_angle [rad].",
        "plots/": f"{GROUND_TRUTH_PLOT_FILE_NAME} (study report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        raw_data = RawDataDirectory.from_configured_path(self.config.raw_dir)
        return [raw_data.mesh_path, raw_data.basis_vectors_path, raw_data.fiber_field_path]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        data = PreprocessedData.load(self.results_dir(run_dir))
        mesh = RawDataDirectory.from_configured_path(self.config.raw_dir).load_mesh()
        render_axial_angle_field(
            mesh, data.ground_truth, plots_dir / GROUND_TRUTH_PLOT_FILE_NAME, "ground truth"
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)

        raw_data = RawDataDirectory.from_configured_path(config.raw_dir)
        with steps.step(f"Loading the mesh and basis vectors from {raw_data.path}"):
            mesh = raw_data.load_mesh()
            basis_vectors = raw_data.load_basis_vectors()
        logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")

        strategy_name = type(config.ground_truth).__name__
        with steps.step(f"Building ground truth and prior mean ({strategy_name})"):
            fiber_field = raw_data.load_fiber_field()
            ground_truth = config.ground_truth.build(mesh, fiber_field, basis_vectors)
            # Always zero: in a realistic scenario the mean of the ground truth is not known.
            prior_mean = np.zeros_like(ground_truth)
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

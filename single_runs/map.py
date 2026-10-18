"""MAP estimation run: maximize the posterior of the preprocessed data.

The run reads the data of a preprocessing run (see
[`InferenceProblemConfig`][single_runs.inference.InferenceProblemConfig]) and does nothing but the
optimization, started at the prior mean, at a draw from the prior or at a stored state. The ground
truth in the data is used for the error metrics only.

Classes:
    MapRunConfig: Which posterior to maximize, and with which optimizer.
    MapResults: The arrays a MAP run writes.
    MapRun: The MAP estimation run.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from ls_bayesian.common.logging import BaseLogger, LoggerSettings

from bayes_cep.mesh.geodesic import assemble_edge_length_graph, compute_graph_distances
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.optimization.initial_guess import (
    FileInitialGuess,
    PriorMeanInitialGuess,
    PriorSampleInitialGuess,
)
from bayes_cep.optimization.strategies import CustomLBFGSStrategy, ScipyLBFGSBStrategy
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap
from bayes_cep.postprocessing.curve_plots import (
    plot_error_by_distance,
    plot_error_cdf,
    plot_optimization_history,
)
from bayes_cep.postprocessing.mesh_plots import (
    render_axial_angle_field_pair,
    render_cell_field,
    render_point_field,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import (
    METRICS_RECORD,
    PLOTS_DIR_NAME,
    ArrayRecord,
    RunDirectory,
    resolve_repository_path,
)
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_error_metrics import (
    ERROR_THRESHOLDS_DEGREES,
    compute_axial_error_metrics,
    compute_error_by_distance,
    compute_skill_score,
)
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff
from single_runs.data import PreprocessedData, RawDataDirectory
from single_runs.inference import AssembledPosterior, InferenceProblemConfig, assemble_posterior

TRUTH_VS_ESTIMATE_PLOT_FILE_NAME = "truth_vs_estimate.pdf"
DIFFERENCE_PLOT_FILE_NAME = "difference_to_truth.pdf"
ERROR_CDF_PLOT_FILE_NAME = "error_cdf.pdf"
ERROR_BY_DISTANCE_PLOT_FILE_NAME = "error_by_distance.pdf"
DISTANCE_PLOT_FILE_NAME = "distance_to_observations.pdf"
OPTIMIZATION_HISTORY_PLOT_FILE_NAME = "optimization_history.pdf"
OPTIMIZER_LOG_FILE_NAME = "optimizer.log"
NUM_DISTANCE_BINS = 10
DISTANCE_QUANTILES = (0.5, 0.9)


# ==================================================================================================
@dataclass(frozen=True)
class MapRunConfig(RunConfig):
    """Which posterior to maximize, and with which optimizer.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior.
        optimizer (CustomLBFGSStrategy | ScipyLBFGSBStrategy): Optimizer backend with its model.
        initial_guess (PriorMeanInitialGuess | PriorSampleInitialGuess | FileInitialGuess): Where
            the optimization starts.
    """

    problem: InferenceProblemConfig
    optimizer: CustomLBFGSStrategy | ScipyLBFGSBStrategy
    initial_guess: PriorMeanInitialGuess | PriorSampleInitialGuess | FileInitialGuess


# ==================================================================================================
@dataclass(frozen=True)
class MapResults(ArrayRecord):
    """The arrays a MAP run writes.

    Attributes:
        estimate (np.ndarray): MAP estimate of the fiber angle [rad] per vertex.
        loss_history (np.ndarray): Loss at every accepted optimizer iterate.
        gradient_norm_history (np.ndarray): Gradient norm at every accepted iterate.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "estimate": "map_estimate.npy",
        "loss_history": "map_loss_history.npy",
        "gradient_norm_history": "map_gradient_norm_history.npy",
    }

    estimate: np.ndarray
    loss_history: np.ndarray
    gradient_norm_history: np.ndarray


# ==================================================================================================
class MapRun(Run[MapRunConfig]):
    """Compute the MAP estimate of the posterior of the preprocessed data."""

    config_type = MapRunConfig
    name = "map"
    outputs = {
        f"results/{MapResults.FILE_NAMES['estimate']}": "MAP estimate of the fiber angle [rad] "
        "per vertex.",
        f"results/{MapResults.FILE_NAMES['loss_history']}": "Loss at every accepted optimizer "
        "iterate.",
        f"results/{MapResults.FILE_NAMES['gradient_norm_history']}": "Gradient norm at every "
        "accepted iterate.",
        METRICS_RECORD: "Optimization statistics (success, num_iterations, loss and gradient norm "
        "first/last, seconds_optimization); the final loss split into likelihood and prior part "
        "and the loss of the ground truth on the branch of the estimate (loss_truth*); the axial "
        "error per simplex in degrees "
        "(error_mean/median/p90/p95/p99/max_deg, fraction_below_<t>_deg, axial_concentration); "
        "the mean error of the prior mean and the skill_score against it; and the RMSE of the "
        "activation times of the estimate against those of the ground truth "
        "(activation_time_rmse_all/observed/unobserved).",
        OPTIMIZER_LOG_FILE_NAME: "Iteration table of the optimizer.",
        "plots/": f"{TRUTH_VS_ESTIMATE_PLOT_FILE_NAME}, {DIFFERENCE_PLOT_FILE_NAME}, "
        f"{ERROR_CDF_PLOT_FILE_NAME}, {ERROR_BY_DISTANCE_PLOT_FILE_NAME}, "
        f"{DISTANCE_PLOT_FILE_NAME}, {OPTIMIZATION_HISTORY_PLOT_FILE_NAME} (study report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        files = self.config.problem.input_files()
        if isinstance(self.config.initial_guess, FileInitialGuess):
            files.append(resolve_repository_path(self.config.initial_guess.path))
        return files

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        problem = self.config.problem
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        metrics = RunDirectory(run_dir).read_metrics()
        data = PreprocessedData.load(resolve_repository_path(problem.data_dir))
        results = MapResults.load(self.results_dir(run_dir))
        mesh = RawDataDirectory.from_configured_path(problem.raw_dir).load_mesh()
        connectivity = get_triangle_connectivity(mesh)
        to_simplices = problem.interpolation.assemble_matrix(mesh.points, connectivity)

        render_axial_angle_field_pair(
            mesh,
            data.ground_truth,
            results.estimate,
            plots_dir / TRUTH_VS_ESTIMATE_PLOT_FILE_NAME,
            "ground truth",
            "map estimate",
        )
        difference = compute_axial_data_diff(
            to_simplices @ results.estimate, to_simplices @ data.ground_truth
        )
        # The absolute error: the sign of an axial difference is not defined near 90 degrees
        error_degrees = np.degrees(np.abs(difference))
        render_cell_field(
            mesh,
            error_degrees,
            plots_dir / DIFFERENCE_PLOT_FILE_NAME,
            f"|MAP - ground truth|, mean {metrics['error_mean_deg']:.3g} deg, "
            f"median {metrics['error_median_deg']:.3g} deg",
            "absolute axial error [deg]",
            "viridis",
            (0.0, 90.0),
        )
        plot_error_cdf(
            [error_degrees],
            ["MAP"],
            plots_dir / ERROR_CDF_PLOT_FILE_NAME,
            ERROR_THRESHOLDS_DEGREES,
        )

        vertex_distances = compute_graph_distances(
            assemble_edge_length_graph(mesh.points, connectivity),
            data.observed_vertex_indices,
            min_over_sources=True,
        )
        render_point_field(
            mesh,
            vertex_distances,
            plots_dir / DISTANCE_PLOT_FILE_NAME,
            "distance to the nearest observation",
            "distance",
            "viridis",
        )
        bin_distances, bin_quantiles = compute_error_by_distance(
            to_simplices @ vertex_distances,
            error_degrees,
            NUM_DISTANCE_BINS,
            DISTANCE_QUANTILES,
        )
        plot_error_by_distance(
            bin_distances,
            bin_quantiles,
            [f"{int(100 * quantile)}th percentile" for quantile in DISTANCE_QUANTILES],
            plots_dir / ERROR_BY_DISTANCE_PLOT_FILE_NAME,
        )
        plot_optimization_history(
            results.loss_history,
            results.gradient_norm_history,
            plots_dir / OPTIMIZATION_HISTORY_PLOT_FILE_NAME,
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        problem = self.config.problem
        steps = StepReporter(logger)
        assembled = assemble_posterior(problem, steps, logger)
        raw_data = RawDataDirectory.from_configured_path(problem.raw_dir)
        optimizer_logger_settings = LoggerSettings(
            print_to_console=self.write_to_console, logfile_path=run_dir / OPTIMIZER_LOG_FILE_NAME
        )
        with BaseLogger(optimizer_logger_settings, prefix="map") as optimizer_logger:
            optimizer_name = type(self.config.optimizer).__name__
            with steps.step(f"Building the optimizer ({optimizer_name})"):
                model, optimizer = self.config.optimizer.build(
                    assembled.log_posterior, optimizer_logger
                )
            with steps.step(
                f"Building the initial guess ({type(self.config.initial_guess).__name__})"
            ):
                initial_guess = self.config.initial_guess.build(assembled.log_posterior.prior)
            logger.info(f"      {describe_array('initial guess [rad]', initial_guess)}")
            optimization_start = time.perf_counter()
            with steps.step("Running MAP estimation"):
                result = optimizer.run(initial_guess=initial_guess, model=model)
            optimization_seconds = time.perf_counter() - optimization_start
        outcome = "converged" if result.success else "did not converge"
        logger.info(f"      MAP estimation {outcome} after {result.num_iterations} iterations")
        logger.info(f"      status: {result.status_message}")
        logger.info(f"      loss: {result.loss_history[0]:.6e} -> {result.loss_history[-1]:.6e}")
        mesh = raw_data.load_mesh()
        to_simplices = problem.interpolation.assemble_matrix(
            mesh.points, get_triangle_connectivity(mesh)
        )
        with steps.step("Evaluating the error metrics"):
            error_metrics = self._compute_error_metrics(
                assembled, result.result, mesh, raw_data.load_basis_vectors(), to_simplices
            )
        logger.info(
            f"      axial error of the MAP: mean {error_metrics['error_mean_deg']:.3f} deg, "
            f"median {error_metrics['error_median_deg']:.3f} deg"
        )
        MapResults(
            estimate=result.result,
            loss_history=result.loss_history,
            gradient_norm_history=result.gradient_norm_history,
        ).save(self.results_dir(run_dir))
        return {
            "success": bool(result.success),
            "num_iterations": int(result.num_iterations),
            "status_message": str(result.status_message),
            "loss_initial": float(result.loss_history[0]),
            "loss_final": float(result.loss_history[-1]),
            "gradient_norm_initial": float(result.gradient_norm_history[0]),
            "gradient_norm_final": float(result.gradient_norm_history[-1]),
            "seconds_optimization": optimization_seconds,
            **error_metrics,
        }

    # ----------------------------------------------------------------------------------------------
    def _compute_error_metrics(
        self,
        assembled: AssembledPosterior,
        estimate: np.ndarray[tuple[int], np.dtype[np.float64]],
        mesh: pv.UnstructuredGrid,
        basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
        to_simplices: sp.coo_array,
    ) -> dict[str, float]:
        """Compare the estimate with the ground truth: axial error, losses and activation times.

        The forward model only sees the per-simplex values, so the angular error is measured
        there: vertices that no simplex takes its value from are constrained by the prior only.
        """
        data = assembled.data
        log_posterior = assembled.log_posterior
        ground_truth_simplices = to_simplices @ data.ground_truth
        difference = compute_axial_data_diff(to_simplices @ estimate, ground_truth_simplices)
        metrics = compute_axial_error_metrics(difference)

        prior_mean_error = np.degrees(
            np.abs(compute_axial_data_diff(to_simplices @ data.prior_mean, ground_truth_simplices))
        ).mean()
        metrics["prior_mean_error_mean_deg"] = float(prior_mean_error)
        metrics["skill_score"] = compute_skill_score(metrics["error_mean_deg"], prior_mean_error)

        # The final loss is the optimizer's own; the loss of the ground truth tells whether the
        # optimizer stopped above it (it is then in a local minimum) or below it. The prior is
        # real-valued, but the stored truth is only one representative of each axial angle (a
        # patient field has cuts where the representative jumps by pi, which the prior would
        # count as roughness). The same axial field on the branch of the estimate, within pi/2 of
        # it at every vertex, is the fair comparison.
        truth_on_estimate_branch = estimate + compute_axial_data_diff(data.ground_truth, estimate)
        final_likelihood = log_posterior.evaluate_likelihood_cost(estimate)
        final_prior = log_posterior.evaluate_prior_cost(estimate)
        truth_likelihood = log_posterior.evaluate_likelihood_cost(truth_on_estimate_branch)
        truth_prior = log_posterior.evaluate_prior_cost(truth_on_estimate_branch)
        metrics["loss_final_likelihood"] = float(final_likelihood)
        metrics["loss_final_prior"] = float(final_prior)
        metrics["loss_truth"] = float(truth_likelihood + truth_prior)
        metrics["loss_truth_likelihood"] = float(truth_likelihood)
        metrics["loss_truth_prior"] = float(truth_prior)

        forward_map = EikonalParameterToSolutionMap(
            mesh, basis_vectors, self.config.problem.eikonal, self.config.problem.interpolation
        )
        activation_time_error = forward_map.evaluate_forward(
            estimate
        ) - forward_map.evaluate_forward(data.ground_truth)
        is_observed = np.zeros(mesh.n_points, dtype=bool)
        is_observed[data.observed_vertex_indices] = True
        metrics["activation_time_rmse_all"] = float(np.sqrt(np.mean(activation_time_error**2)))
        metrics["activation_time_rmse_observed"] = float(
            np.sqrt(np.mean(activation_time_error[is_observed] ** 2))
        )
        metrics["activation_time_rmse_unobserved"] = float(
            np.sqrt(np.mean(activation_time_error[~is_observed] ** 2))
        )
        return metrics

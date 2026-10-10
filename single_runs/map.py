"""MAP estimation run: maximize the posterior of the preprocessed data.

The run reads the data of a preprocessing run (see
[`InferenceProblemConfig`][single_runs.inference.InferenceProblemConfig]) and does nothing but the
optimization, started at the prior mean. The ground truth in the data is used for the error
metrics only.

Constants:
    REFERENCE_*: Settings of the reference MAP run, used as defaults.

Classes:
    MapRunConfig: Which posterior to maximize, and with which optimizer.
    MapRun: The MAP estimation run.

Functions:
    reference_optimizer_strategy: Cameron-Martin L-BFGS with the loose reference tolerance.
    reference_map_config: The MAP run configuration of the example data.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import numpy as np
from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.optimization.algorithms.custom_lbfgs import CustomLBFGSSettings

from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.mesh.plotting import render_cell_field
from bayes_cep.optimization.strategies import CustomLBFGSStrategy, ScipyLBFGSBStrategy
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import RunDirectory, resolve_repository_path
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff, wrap_axial_angles
from single_runs.inference import InferenceProblemConfig, reference_inference_problem
from single_runs.preprocessing import PreprocessedData

REFERENCE_GRADIENT_NORM_TOLERANCE = 100.0
REFERENCE_MAX_NUM_ITERATIONS = 1000


# ==================================================================================================
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
class MapRunConfig(RunConfig):
    """Which posterior to maximize, and with which optimizer.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior.
        optimizer (CustomLBFGSStrategy | ScipyLBFGSBStrategy): Optimizer backend with its model.
    """

    problem: InferenceProblemConfig
    optimizer: CustomLBFGSStrategy | ScipyLBFGSBStrategy = field(
        default_factory=reference_optimizer_strategy
    )


# ==================================================================================================
def reference_map_config(preprocessing_dir: Path) -> MapRunConfig:
    """The MAP run configuration of the example data.

    Args:
        preprocessing_dir (Path): Run directory of the preprocessing run that produced the data;
            there is deliberately no default.

    Returns:
        MapRunConfig: The reference configuration for the given data.
    """
    return MapRunConfig(problem=reference_inference_problem(preprocessing_dir))


# ==================================================================================================
class MapRun(Run[MapRunConfig]):
    """Compute the MAP estimate of the posterior of the preprocessed data."""

    outputs = {
        "results/map_estimate.npy": "MAP estimate of the fiber angle [rad] per vertex.",
        "results/map_loss_history.npy": "Loss at every accepted optimizer iterate.",
        "results/map_gradient_norm_history.npy": "Gradient norm at every accepted iterate.",
        "metrics.json": "Optimization statistics (success, num_iterations, loss and gradient norm "
        "first/last, seconds_optimization) and axial error [rad] of the MAP estimate against the "
        "ground truth (error_mean_abs/median_abs/max_abs/rmse), plus the mean absolute error of "
        "the prior mean as a baseline (prior_mean_error_mean_abs).",
        "optimizer.log": "Iteration table of the optimizer.",
        "plots/": "map_estimate.png, difference_to_truth.png, optimization_history.png (study "
        "report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return self.config.problem.input_files()

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        problem = self.config.problem
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        results_dir = run_dir / "results"
        metrics = RunDirectory(run_dir).read_metrics()
        ground_truth = PreprocessedData.load(problem.preprocessed_results_dir).ground_truth
        map_estimate = np.load(results_dir / "map_estimate.npy")

        mesh = load_pyvista_mesh(resolve_repository_path(problem.raw_dir) / "mesh.vtu")
        connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
        to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(
            mesh.points, connectivity
        )
        render_cell_field(
            mesh,
            wrap_axial_angles(to_simplices @ map_estimate),
            plots_dir / "map_estimate.png",
            "map estimate",
            "angle [rad]",
            "hsv",
            (-np.pi / 2, np.pi / 2),
        )
        difference = to_simplices @ compute_axial_data_diff(map_estimate, ground_truth)
        limit = float(np.abs(difference).max())
        render_cell_field(
            mesh,
            difference,
            plots_dir / "difference_to_truth.png",
            f"MAP - ground truth, mean abs {metrics['error_mean_abs']:.3g} rad",
            "difference [rad]",
            "coolwarm",
            (-limit, limit),
        )

        figure, (loss_axis, gradient_axis) = plt.subplots(1, 2, figsize=(10, 4))
        loss_axis.plot(np.load(results_dir / "map_loss_history.npy"))
        loss_axis.set_xlabel("iteration")
        loss_axis.set_ylabel("loss")
        gradient_axis.semilogy(np.load(results_dir / "map_gradient_norm_history.npy"))
        gradient_axis.set_xlabel("iteration")
        gradient_axis.set_ylabel("gradient norm")
        figure.tight_layout()
        figure.savefig(plots_dir / "optimization_history.png", dpi=150)
        plt.close(figure)

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        steps = StepReporter(logger)
        assembled = self.config.problem.assemble_posterior(steps, logger)
        data = assembled.data

        optimizer_logger_settings = LoggerSettings(
            print_to_console=self.write_to_console, logfile_path=run_dir / "optimizer.log"
        )
        with BaseLogger(optimizer_logger_settings, prefix="map") as optimizer_logger:
            optimizer_name = type(self.config.optimizer).__name__
            with steps.step(f"Building the optimizer ({optimizer_name})"):
                model, optimizer = self.config.optimizer.build(
                    assembled.log_posterior, assembled.posterior_builder.prior, optimizer_logger
                )
            optimization_start = time.perf_counter()
            with steps.step("Running MAP estimation"):
                result = optimizer.run(initial_guess=data.prior_mean, model=model)
            optimization_seconds = time.perf_counter() - optimization_start

        outcome = "converged" if result.success else "did not converge"
        logger.info(f"      MAP estimation {outcome} after {result.num_iterations} iterations")
        logger.info(f"      status: {result.status_message}")
        logger.info(f"      loss: {result.loss_history[0]:.6e} -> {result.loss_history[-1]:.6e}")

        map_error = np.abs(compute_axial_data_diff(result.result, data.ground_truth))
        prior_mean_error = np.abs(compute_axial_data_diff(data.prior_mean, data.ground_truth))
        logger.info(f"      axial error of the MAP: mean {map_error.mean():.4f} rad")

        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        np.save(results_dir / "map_estimate.npy", result.result)
        np.save(results_dir / "map_loss_history.npy", result.loss_history)
        np.save(results_dir / "map_gradient_norm_history.npy", result.gradient_norm_history)

        return {
            "success": bool(result.success),
            "num_iterations": int(result.num_iterations),
            "status_message": str(result.status_message),
            "loss_initial": float(result.loss_history[0]),
            "loss_final": float(result.loss_history[-1]),
            "gradient_norm_initial": float(result.gradient_norm_history[0]),
            "gradient_norm_final": float(result.gradient_norm_history[-1]),
            "seconds_optimization": optimization_seconds,
            "error_mean_abs": float(map_error.mean()),
            "error_median_abs": float(np.median(map_error)),
            "error_max_abs": float(map_error.max()),
            "error_rmse": float(np.sqrt(np.mean(map_error**2))),
            "prior_mean_error_mean_abs": float(prior_mean_error.mean()),
        }

"""MAP estimation run: maximize the posterior of the preprocessed data.

The run reads the data of a preprocessing run (see
[`InferenceProblemConfig`][single_runs.inference.InferenceProblemConfig]) and does nothing but the
optimization, started at the prior mean. The ground truth in the data is used for the error
metrics only.

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
from ls_bayesian.common.logging import BaseLogger, LoggerSettings

from bayes_cep.optimization.strategies import CustomLBFGSStrategy, ScipyLBFGSBStrategy
from bayes_cep.postprocessing.curve_plots import plot_optimization_history
from bayes_cep.postprocessing.mesh_plots import (
    render_axial_angle_field,
    render_vertex_field_on_simplices,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import (
    METRICS_RECORD,
    PLOTS_DIR_NAME,
    ArrayRecord,
    RunDirectory,
    resolve_repository_path,
)
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff
from single_runs.data import PreprocessedData, RawDataDirectory
from single_runs.inference import InferenceProblemConfig, assemble_posterior

ESTIMATE_PLOT_FILE_NAME = "map_estimate.pdf"
DIFFERENCE_PLOT_FILE_NAME = "difference_to_truth.pdf"
OPTIMIZATION_HISTORY_PLOT_FILE_NAME = "optimization_history.pdf"
OPTIMIZER_LOG_FILE_NAME = "optimizer.log"


# ==================================================================================================
@dataclass(frozen=True)
class MapRunConfig(RunConfig):
    """Which posterior to maximize, and with which optimizer.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior.
        optimizer (CustomLBFGSStrategy | ScipyLBFGSBStrategy): Optimizer backend with its model.
    """

    problem: InferenceProblemConfig
    optimizer: CustomLBFGSStrategy | ScipyLBFGSBStrategy


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
        "first/last, seconds_optimization) and axial error [rad] of the MAP estimate against the "
        "ground truth (error_mean_abs/median_abs/max_abs/rmse), plus the mean absolute error of "
        "the prior mean as a baseline (prior_mean_error_mean_abs).",
        OPTIMIZER_LOG_FILE_NAME: "Iteration table of the optimizer.",
        "plots/": f"{ESTIMATE_PLOT_FILE_NAME}, {DIFFERENCE_PLOT_FILE_NAME}, "
        f"{OPTIMIZATION_HISTORY_PLOT_FILE_NAME} (study report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return self.config.problem.input_files()

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        problem = self.config.problem
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        metrics = RunDirectory(run_dir).read_metrics()
        ground_truth = PreprocessedData.load(resolve_repository_path(problem.data_dir)).ground_truth
        results = MapResults.load(self.results_dir(run_dir))
        mesh = RawDataDirectory.from_configured_path(problem.raw_dir).load_mesh()
        render_axial_angle_field(
            mesh, results.estimate, plots_dir / ESTIMATE_PLOT_FILE_NAME, "map estimate"
        )
        difference = compute_axial_data_diff(results.estimate, ground_truth)
        limit = float(np.abs(difference).max())
        render_vertex_field_on_simplices(
            mesh,
            difference,
            plots_dir / DIFFERENCE_PLOT_FILE_NAME,
            f"MAP - ground truth, mean abs {metrics['error_mean_abs']:.3g} rad",
            "difference [rad]",
            "coolwarm",
            (-limit, limit),
        )
        plot_optimization_history(
            results.loss_history,
            results.gradient_norm_history,
            plots_dir / OPTIMIZATION_HISTORY_PLOT_FILE_NAME,
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        steps = StepReporter(logger)
        assembled = assemble_posterior(self.config.problem, steps, logger)
        data = assembled.data
        optimizer_logger_settings = LoggerSettings(
            print_to_console=self.write_to_console, logfile_path=run_dir / OPTIMIZER_LOG_FILE_NAME
        )
        with BaseLogger(optimizer_logger_settings, prefix="map") as optimizer_logger:
            optimizer_name = type(self.config.optimizer).__name__
            with steps.step(f"Building the optimizer ({optimizer_name})"):
                model, optimizer = self.config.optimizer.build(
                    assembled.log_posterior, optimizer_logger
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
            "error_mean_abs": float(map_error.mean()),
            "error_median_abs": float(np.median(map_error)),
            "error_max_abs": float(map_error.max()),
            "error_rmse": float(np.sqrt(np.mean(map_error**2))),
            "prior_mean_error_mean_abs": float(prior_mean_error.mean()),
        }

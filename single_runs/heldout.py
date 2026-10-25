"""Held-out MAP run: the MAP estimate of a part of the observations, scored on the rest.

One fold of a cross-validation of the prior parameters. The run reads the data of a preprocessing
run, leaves out one fold of its observations, computes the MAP estimate of the posterior of the
others and predicts the activation times at the held-out vertices. The ground truth in the data is
not used, so the score is available for real data as well. A study sweeps the prior parameters and
the folds; `scripts/postprocess_prior_selection_study.py` averages the score over the folds.

Classes:
    HeldOutMapRunConfig: Which posterior, which optimizer and which fold to hold out.
    HeldOutResults: The residuals the run writes.
    HeldOutMapRun: The held-out MAP run.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger, LoggerSettings

from bayes_cep.optimization.initial_guess import (
    FileInitialGuess,
    PriorMeanInitialGuess,
    PriorSampleInitialGuess,
)
from bayes_cep.optimization.strategies import CustomLBFGSStrategy, ScipyLBFGSBStrategy
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap
from bayes_cep.postprocessing.curve_plots import plot_histogram
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import (
    METRICS_RECORD,
    PLOTS_DIR_NAME,
    ArrayRecord,
    resolve_repository_path,
)
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.cross_validation import (
    HeldOutSplitSettings,
    compute_gaussian_negative_log_predictive,
)
from single_runs.data import PreprocessedData, RawDataDirectory
from single_runs.inference import InferenceProblemConfig, assemble_posterior
from single_runs.map import OPTIMIZER_LOG_FILE_NAME

RESIDUAL_HISTOGRAM_PLOT_FILE_NAME = "heldout_residuals.pdf"


# ==================================================================================================
@dataclass(frozen=True)
class HeldOutMapRunConfig(RunConfig):
    """Which posterior to maximize, with which optimizer, and which fold of the data to hold out.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior; the likelihood
            uses only the training observations.
        optimizer (CustomLBFGSStrategy | ScipyLBFGSBStrategy): Optimizer backend with its model.
        initial_guess (PriorMeanInitialGuess | PriorSampleInitialGuess | FileInitialGuess): Where
            the optimization starts.
        split (HeldOutSplitSettings): The fold held out.
    """

    problem: InferenceProblemConfig
    optimizer: CustomLBFGSStrategy | ScipyLBFGSBStrategy
    initial_guess: PriorMeanInitialGuess | PriorSampleInitialGuess | FileInitialGuess
    split: HeldOutSplitSettings


# ==================================================================================================
@dataclass(frozen=True)
class HeldOutResults(ArrayRecord):
    """The arrays a held-out MAP run writes.

    Attributes:
        heldout_residuals (np.ndarray): Predicted minus observed activation time at the held-out
            observations, shape `(num_heldout,)`.
        training_residuals (np.ndarray): The same at the training observations, shape
            `(num_training,)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "heldout_residuals": "heldout_residuals.npy",
        "training_residuals": "training_residuals.npy",
    }

    heldout_residuals: np.ndarray
    training_residuals: np.ndarray


# ==================================================================================================
class HeldOutMapRun(Run[HeldOutMapRunConfig]):
    """Compute the MAP estimate of the training observations and predict the held-out ones."""

    config_type = HeldOutMapRunConfig
    name = "heldout-map"
    outputs = {
        f"results/{HeldOutResults.FILE_NAMES['heldout_residuals']}": "Predicted minus observed "
        "activation time at the held-out observations.",
        f"results/{HeldOutResults.FILE_NAMES['training_residuals']}": "The same at the training "
        "observations.",
        METRICS_RECORD: "Optimization statistics (success, num_iterations, loss and gradient norm "
        "first/last, seconds_optimization), the numbers of training and held-out observations, "
        "the RMSE of the activation times at both (training_rmse, heldout_rmse) and the mean "
        "negative log predictive density of the held-out observations "
        "(heldout_negative_log_predictive). The ground truth is not used.",
        OPTIMIZER_LOG_FILE_NAME: "Iteration table of the optimizer.",
        "plots/": f"{RESIDUAL_HISTOGRAM_PLOT_FILE_NAME} (study report).",
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
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        results = HeldOutResults.load(self.results_dir(run_dir))
        plot_histogram(
            results.heldout_residuals,
            plots_dir / RESIDUAL_HISTOGRAM_PLOT_FILE_NAME,
            "predicted minus observed activation time",
            "held-out observations",
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        problem = self.config.problem
        steps = StepReporter(logger)
        raw_data = RawDataDirectory.from_configured_path(problem.raw_dir)
        data = PreprocessedData.load(resolve_repository_path(problem.data_dir))
        training_positions, heldout_positions = self.config.split.split(
            data.observed_vertex_indices.shape[0]
        )
        logger.info(
            f"      fold {self.config.split.fold_index} of {self.config.split.num_folds}: "
            f"{training_positions.size} training and {heldout_positions.size} held-out observations"
        )
        assembled = assemble_posterior(problem, steps, logger, training_positions)

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
            optimization_start = time.perf_counter()
            with steps.step("Running MAP estimation"):
                result = optimizer.run(initial_guess=initial_guess, model=model)
            optimization_seconds = time.perf_counter() - optimization_start
        outcome = "converged" if result.success else "did not converge"
        logger.info(f"      MAP estimation {outcome} after {result.num_iterations} iterations")
        logger.info(f"      loss: {result.loss_history[0]:.6e} -> {result.loss_history[-1]:.6e}")

        with steps.step("Predicting the activation times"):
            forward_map = EikonalParameterToSolutionMap(
                raw_data.load_mesh(),
                raw_data.load_basis_vectors(),
                problem.eikonal,
                problem.interpolation,
            )
            predicted = forward_map.evaluate_forward(result.result)
        residuals = predicted[data.observed_vertex_indices] - data.observed_activation_times
        heldout_residuals = residuals[heldout_positions]
        training_residuals = residuals[training_positions]
        logger.info(f"      {describe_array('held-out residuals', heldout_residuals)}")

        HeldOutResults(
            heldout_residuals=heldout_residuals, training_residuals=training_residuals
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
            "num_training_observations": int(training_positions.size),
            "num_heldout_observations": int(heldout_positions.size),
            "training_rmse": float(np.sqrt(np.mean(training_residuals**2))),
            "heldout_rmse": float(np.sqrt(np.mean(heldout_residuals**2))),
            "heldout_negative_log_predictive": compute_gaussian_negative_log_predictive(
                heldout_residuals, data.noise_variance
            ),
        }

"""Recovery test of the prior parameter estimation: fit a field whose kappa and tau are known.

A zero-mean prior sample with the true parameters is drawn on the patient mesh, smoothed like a
smoothed patient field if `fit.smoothing_scale` is positive, and then fitted exactly as in
[`PriorMomentFitRun`][single_runs.prior_fit.PriorMomentFitRun]. The field is a draw from the model
the fit assumes, so this is the easy case: if the truth is not recovered here, it will not be for
the patient fields either. The run keeps the whole loss surface and records how far the estimate and
the width of the valley are from the truth.

Classes:
    PriorRecoveryRunConfig: Which true parameters to draw, and which grid and statistics to fit.
    PriorRecoveryRun: The recovery run.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.mesh.interpolation import (
    LinearInterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.posterior.prior import PriorParameters, build_fiber_angle_prior
from bayes_cep.postprocessing.curve_plots import plot_binned_correlation_comparison
from bayes_cep.postprocessing.summary_plots import plot_loss_maps
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, PLOTS_DIR_NAME
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_smoothing import smooth_axial_angle_field
from bayes_cep.statistics.prior_moment_fit import (
    MomentFitResult,
    MomentFitSettings,
    fit_prior_parameters,
)
from single_runs.data import RawDataDirectory
from single_runs.prior_fit import (
    CORRELATION_PLOT_FILE_NAME,
    LOSS_MAP_PLOT_FILE_NAME,
    MOMENT_FIT_FILE_NAME,
)

TRUE_FIELD_FILE_NAME = "true_vertex_angles.npy"
NEAR_MINIMUM_FACTOR = 1.1


# ==================================================================================================
@dataclass(frozen=True)
class PriorRecoveryRunConfig(RunConfig):
    """Fit a prior sample of known parameters on a grid of prior parameters.

    Attributes:
        raw_dir (Path): Patient directory; only `mesh.vtu` is read.
        true_prior (PriorParameters): Parameters and seed of the sample that is fitted. Its seed
            must differ from `fit.seed`, which seeds the prior samples of the fit, so that the
            truth is not one of them.
        fit (MomentFitSettings): Statistics and simulation; `fit.smoothing_scale` is also the
            scale the true field is smoothed with (`0`: not smoothed).
        kappas (tuple[float, ...]): Values of $\\kappa$ (rows of the loss), positive.
        taus (tuple[float, ...]): Values of $\\tau$ (columns of the loss), positive.
        interpolation (LinearInterpolationStrategy | NearestNeighborInterpolationStrategy):
            Vertex-to-simplex map of the true field and of the prior samples of the fit.
    """

    raw_dir: Path
    true_prior: PriorParameters
    fit: MomentFitSettings
    kappas: tuple[float, ...]
    taus: tuple[float, ...]
    interpolation: LinearInterpolationStrategy | NearestNeighborInterpolationStrategy

    def __post_init__(self) -> None:
        for name, values in (("kappas", self.kappas), ("taus", self.taus)):
            if len(values) == 0 or min(values) <= 0.0:
                raise ValueError(f"{name} must be non-empty and positive, got {values}.")
        if self.true_prior.seed == self.fit.seed:
            raise ValueError(
                f"The seed of the true field and fit.seed must differ, both are {self.fit.seed}."
            )


# ==================================================================================================
class PriorRecoveryRun(Run[PriorRecoveryRunConfig]):
    """Fit a prior sample of known parameters and compare the estimate with the truth."""

    config_type = PriorRecoveryRunConfig
    name = "prior-recovery"
    outputs = {
        f"results/{TRUE_FIELD_FILE_NAME}": "The fitted field [rad] per vertex: the prior sample, "
        "smoothed if fit.smoothing_scale is positive.",
        f"results/{MOMENT_FIT_FILE_NAME}": "A MomentFitResult, as of the prior moment fit run.",
        METRICS_RECORD: "true_kappa, true_tau, best_kappa, best_tau (grid point of the smallest "
        "loss), log_kappa_error, log_tau_error, log_kappa_tau_error (natural log of estimate over "
        "truth, of kappa, tau and kappa * tau), loss_min, loss_at_truth (the grid point nearest "
        "the truth in the logarithm), fraction_near_minimum (share of the grid with a loss within "
        "a factor 1.1 of the minimum: the width of the valley), observed_concentration, seconds.",
        "plots/": f"{LOSS_MAP_PLOT_FILE_NAME}, {CORRELATION_PLOT_FILE_NAME} (study report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return [RawDataDirectory.from_configured_path(self.config.raw_dir).mesh_path]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        config = self.config
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        result = MomentFitResult.load(self.results_dir(run_dir) / MOMENT_FIT_FILE_NAME)
        label = (
            f"true kappa {config.true_prior.kappa:g}, tau {config.true_prior.tau:g}, "
            f"smoothing scale {config.fit.smoothing_scale:g}"
        )
        plot_loss_maps(
            result.kappas,
            result.taus,
            [result.loss],
            [label],
            plots_dir / LOSS_MAP_PLOT_FILE_NAME,
            true_parameters=[(config.true_prior.kappa, config.true_prior.tau)],
        )
        best_row, best_column = np.unravel_index(np.argmin(result.loss), result.loss.shape)
        plot_binned_correlation_comparison(
            result.distance_bin_edges,
            result.observed_correlation,
            [result.correlation[best_row, best_column]],
            [f"prior, kappa {result.kappas[best_row]:.3g}, tau {result.taus[best_column]:.3g}"],
            config.fit.num_fitted_bins,
            plots_dir / CORRELATION_PLOT_FILE_NAME,
            label,
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        start_time = time.perf_counter()
        config = self.config
        steps = StepReporter(logger)
        results_dir = self.results_dir(run_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        raw_data = RawDataDirectory.from_configured_path(config.raw_dir)
        with steps.step(f"Loading the mesh from {raw_data.path}"):
            mesh = raw_data.load_mesh()
            connectivity = get_triangle_connectivity(mesh)
        logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")

        with steps.step(
            f"Drawing the true field (kappa {config.true_prior.kappa:g}, "
            f"tau {config.true_prior.tau:g}, smoothing scale {config.fit.smoothing_scale:g})"
        ):
            true_prior = build_fiber_angle_prior(
                create_dolfinx_mesh(mesh), np.zeros(mesh.n_points), config.true_prior
            )
            true_vertex_angles = true_prior.generate_sample()
            if config.fit.smoothing_scale > 0.0:
                true_vertex_angles, _ = smooth_axial_angle_field(
                    mesh, true_vertex_angles, config.fit.smoothing_scale
                )
            observed = (
                config.interpolation.assemble_matrix(mesh.points, connectivity) @ true_vertex_angles
            )
        logger.info(f"      {describe_array('true field [rad]', true_vertex_angles)}")
        np.save(results_dir / TRUE_FIELD_FILE_NAME, true_vertex_angles)

        with steps.step(
            f"Scoring {len(config.kappas)} x {len(config.taus)} prior parameter pairs "
            f"({config.fit.num_prior_samples} samples each)"
        ):
            result = fit_prior_parameters(
                mesh, observed, config.kappas, config.taus, config.fit, config.interpolation
            )
        result.save(results_dir / MOMENT_FIT_FILE_NAME)

        best_kappa, best_tau = result.best_parameters()
        truth_row = int(np.argmin(np.abs(np.log(result.kappas / config.true_prior.kappa))))
        truth_column = int(np.argmin(np.abs(np.log(result.taus / config.true_prior.tau))))
        loss_min = float(result.loss.min())
        logger.info(
            f"      best kappa {best_kappa:g}, tau {best_tau:g}, loss {loss_min:.4g} "
            f"(truth: kappa {config.true_prior.kappa:g}, tau {config.true_prior.tau:g}, "
            f"loss {result.loss[truth_row, truth_column]:.4g})"
        )
        return {
            "true_kappa": config.true_prior.kappa,
            "true_tau": config.true_prior.tau,
            "best_kappa": best_kappa,
            "best_tau": best_tau,
            "log_kappa_error": float(np.log(best_kappa / config.true_prior.kappa)),
            "log_tau_error": float(np.log(best_tau / config.true_prior.tau)),
            "log_kappa_tau_error": float(
                np.log(best_kappa * best_tau / (config.true_prior.kappa * config.true_prior.tau))
            ),
            "loss_min": loss_min,
            "loss_at_truth": float(result.loss[truth_row, truth_column]),
            "fraction_near_minimum": float(np.mean(result.loss <= NEAR_MINIMUM_FACTOR * loss_min)),
            "observed_concentration": result.observed_concentration,
            "seconds": time.perf_counter() - start_time,
        }

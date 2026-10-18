"""Prior parameter estimation run: the simulated-moment loss over a grid of kappa and tau.

Scores every pair of the grid against the fiber field of one patient (raw, or smoothed at a given
scale) with the statistics of `bayes_cep.statistics.prior_moment_fit`. The run keeps the whole loss
surface, not only its minimum, because the loss is flat along a valley of nearly constant
$\\kappa\\tau$. The losses of the fields of several patients, which have the same grid in a study,
are added afterwards (`pool_losses`).

Classes:
    PriorMomentFitRunConfig: Which field to fit, on which grid, with which statistics.
    PriorMomentFitRun: The prior parameter estimation run.
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
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.postprocessing.curve_plots import plot_binned_correlation_comparison
from bayes_cep.postprocessing.summary_plots import plot_loss_maps
from bayes_cep.preprocessing.angle_field import compute_angle_from_fiber_vectors
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, PLOTS_DIR_NAME
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.prior_moment_fit import (
    MomentFitResult,
    MomentFitSettings,
    fit_prior_parameters,
)
from single_runs.data import RawDataDirectory

MOMENT_FIT_FILE_NAME = "moment_fit.npz"
LOSS_MAP_PLOT_FILE_NAME = "loss_map.pdf"
CORRELATION_PLOT_FILE_NAME = "correlation.pdf"


# ==================================================================================================
@dataclass(frozen=True)
class PriorMomentFitRunConfig(RunConfig):
    """Score a grid of prior parameters against the fiber field of a patient.

    Attributes:
        raw_dir (Path): Patient directory (`mesh.vtu`, `basis_vecs.npy`, `fiber_field.npy`, and for
            a smoothed field `smoothed_scale_<scale>/angle_field.npy`).
        fit (MomentFitSettings): Statistics and simulation. `fit.smoothing_scale` selects the
            field: `0` the raw one, else the field smoothed at that scale, which is compared with
            prior samples smoothed in the same way.
        kappas (tuple[float, ...]): Values of $\\kappa$ (rows of the loss), positive.
        taus (tuple[float, ...]): Values of $\\tau$ (columns of the loss), positive.
        interpolation (LinearInterpolationStrategy | NearestNeighborInterpolationStrategy):
            Vertex-to-simplex map of the prior samples; use the one of the forward model.
    """

    raw_dir: Path
    fit: MomentFitSettings
    kappas: tuple[float, ...]
    taus: tuple[float, ...]
    interpolation: LinearInterpolationStrategy | NearestNeighborInterpolationStrategy

    def __post_init__(self) -> None:
        for name, values in (("kappas", self.kappas), ("taus", self.taus)):
            if len(values) == 0 or min(values) <= 0.0:
                raise ValueError(f"{name} must be non-empty and positive, got {values}.")


# ==================================================================================================
class PriorMomentFitRun(Run[PriorMomentFitRunConfig]):
    """Score the prior parameter grid against one fiber field."""

    config_type = PriorMomentFitRunConfig
    name = "prior-moment-fit"
    outputs = {
        f"results/{MOMENT_FIT_FILE_NAME}": "A MomentFitResult: kappas, taus, the loss "
        "(num_kappas, num_taus), the mean correlation per distance bin and the mean "
        "concentration of the prior samples on the grid, the same statistics of the observed "
        "field, and the bin edges.",
        METRICS_RECORD: "best_kappa and best_tau (grid point of the smallest loss), loss_min, "
        "observed_concentration, seconds.",
        "plots/": f"{LOSS_MAP_PLOT_FILE_NAME}, {CORRELATION_PLOT_FILE_NAME} (study report).",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        raw_data = RawDataDirectory.from_configured_path(self.config.raw_dir)
        if self.config.fit.smoothing_scale == 0.0:
            return [raw_data.mesh_path, raw_data.basis_vectors_path, raw_data.fiber_field_path]
        return [
            raw_data.mesh_path,
            raw_data.smoothed_angle_field_path(self.config.fit.smoothing_scale),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        result = MomentFitResult.load(self.results_dir(run_dir) / MOMENT_FIT_FILE_NAME)
        label = f"{self.config.raw_dir.name}, smoothing scale {self.config.fit.smoothing_scale:g}"
        plot_loss_maps(
            result.kappas, result.taus, [result.loss], [label], plots_dir / LOSS_MAP_PLOT_FILE_NAME
        )
        best_row, best_column = np.unravel_index(np.argmin(result.loss), result.loss.shape)
        plot_binned_correlation_comparison(
            result.distance_bin_edges,
            result.observed_correlation,
            [result.correlation[best_row, best_column]],
            [f"prior, kappa {result.kappas[best_row]:.3g}, tau {result.taus[best_column]:.3g}"],
            self.config.fit.num_fitted_bins,
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
        with steps.step(f"Loading the mesh and the fiber field from {raw_data.path}"):
            mesh = raw_data.load_mesh()
            if config.fit.smoothing_scale == 0.0:
                basis_vectors = raw_data.load_basis_vectors()
                observed = compute_angle_from_fiber_vectors(
                    raw_data.load_fiber_field(), basis_vectors[..., 0], basis_vectors[..., 1]
                )
            else:
                vertex_to_simplex = config.interpolation.assemble_matrix(
                    mesh.points, get_triangle_connectivity(mesh)
                )
                observed = vertex_to_simplex @ np.load(
                    raw_data.smoothed_angle_field_path(config.fit.smoothing_scale)
                )
        logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")
        logger.info(f"      {describe_array('observed angle [rad]', observed)}")

        with steps.step(
            f"Scoring {len(config.kappas)} x {len(config.taus)} prior parameter pairs "
            f"({config.fit.num_prior_samples} samples each)"
        ):
            result = fit_prior_parameters(
                mesh, observed, config.kappas, config.taus, config.fit, config.interpolation
            )
        result.save(results_dir / MOMENT_FIT_FILE_NAME)

        best_kappa, best_tau = result.best_parameters()
        logger.info(
            f"      best kappa {best_kappa:g}, tau {best_tau:g}, loss {result.loss.min():.4g}"
        )
        return {
            "best_kappa": best_kappa,
            "best_tau": best_tau,
            "loss_min": float(result.loss.min()),
            "observed_concentration": result.observed_concentration,
            "seconds": time.perf_counter() - start_time,
        }

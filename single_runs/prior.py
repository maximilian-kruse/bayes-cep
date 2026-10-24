"""Prior investigation run: sample the prior, estimate pointwise variance and correlation length.

A run is a pure function of its configuration: everything it depends on (parameters, seeds, data
source) is a field of its configuration, and nothing is hidden in module state. Configurations hold
paths and settings only, never arrays, so they can be serialized, hashed and compared.

Classes:
    PriorRunConfig: Sample the prior and estimate its pointwise variance and correlation length.
    PriorRun: The prior investigation run.
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.posterior.prior import PriorParameters, build_fiber_angle_prior
from bayes_cep.postprocessing.curve_plots import (
    plot_correlation_curve,
    plot_correlation_histogram,
)
from bayes_cep.postprocessing.mesh_plots import render_axial_angle_field, render_point_field
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, PLOTS_DIR_NAME, RunDirectory
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import (
    compute_axial_mean_and_variance,
    compute_standard_deviation_degrees,
    shift_angles_to_minimize_axial_variance,
)
from bayes_cep.statistics.correlation_length import (
    CorrelationLengthSettings,
    estimate_correlation_length,
)
from single_runs.data import RawDataDirectory

SAMPLE_FILE_NAME = "sample.npy"
POINTWISE_VARIANCE_FILE_NAME = "pointwise_variance.npy"
CORRELATION_CURVE_FILE_NAME = "correlation_curve.npz"
SAMPLES_FILE_NAME = "samples.npy"
SAMPLE_PLOT_FILE_NAME = "sample.pdf"
POINTWISE_STD_PLOT_FILE_NAME = "pointwise_std.pdf"
CORRELATION_CURVE_PLOT_FILE_NAME = "correlation_curve.pdf"
CORRELATION_HISTOGRAM_PLOT_FILE_NAME = "correlation_histogram.pdf"


# ==================================================================================================
@dataclass(frozen=True)
class PriorRunConfig(RunConfig):
    """Sample a zero-mean prior and estimate its pointwise variance and correlation length.

    Needs only the mesh: no data, likelihood or optimization is involved.

    Attributes:
        raw_dir (Path): Directory containing `mesh.vtu`.
        correlation (CorrelationLengthSettings): Settings of the correlation length estimation.
        prior (PriorParameters): Prior parameters.
        num_samples (int): Number of samples for the variance and correlation length estimates.
        keep_samples (bool): Whether to store the samples themselves as a result. Keep it off in
            studies, since many runs would otherwise add up to gigabytes.
    """

    raw_dir: Path
    correlation: CorrelationLengthSettings
    prior: PriorParameters
    num_samples: int
    keep_samples: bool

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError(f"num_samples must be at least 2, got {self.num_samples}.")


# ==================================================================================================
class PriorRun(Run[PriorRunConfig]):
    """Sample a zero-mean prior, then estimate its pointwise variance and correlation length."""

    config_type = PriorRunConfig
    name = "prior"
    outputs = {
        f"results/{SAMPLE_FILE_NAME}": "One prior sample [rad] per vertex, re-branched onto a "
        "single pi-periodic branch around its axial mean; drawn first, for visualization.",
        f"results/{POINTWISE_VARIANCE_FILE_NAME}": "Axial variance per vertex (-ln(R)/2, with R "
        "the length of the mean resultant of the doubled angles) over the samples.",
        f"results/{CORRELATION_CURVE_FILE_NAME}": "Binned correlation versus distance: "
        "bin_distances, bin_correlations, bin_counts, fitted_correlations (nan if the fit "
        "failed), fit_mask, base_vertices, and the histogram of the pooled pair correlations: "
        "histogram_counts, histogram_edges, noise_floor.",
        f"results/{SAMPLES_FILE_NAME}": "All samples, shape (num_samples, num_vertices); only if "
        "keep_samples.",
        METRICS_RECORD: "std_mean/min/max_deg: statistics of the pointwise axial standard "
        "deviation field in degrees; "
        "correlation_distance_threshold: distance where the binned correlation first drops below "
        "the threshold (model-free); correlation_length: fitted Matern length (nan if the fit "
        "failed); correlation_fit_succeeded and correlation_fit_failure_reason (empty if it "
        "succeeded): a failed fit does not fail the run; seconds.",
        "plots/": f"{SAMPLE_PLOT_FILE_NAME}, {POINTWISE_STD_PLOT_FILE_NAME}, "
        f"{CORRELATION_CURVE_PLOT_FILE_NAME}, {CORRELATION_HISTOGRAM_PLOT_FILE_NAME} "
        "(study report).",
    }

    optional_outputs = frozenset({f"results/{SAMPLES_FILE_NAME}"})

    @override
    def input_files(self) -> list[Path]:
        return [RawDataDirectory.from_configured_path(self.config.raw_dir).mesh_path]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        config = self.config
        plots_dir = run_dir / PLOTS_DIR_NAME
        plots_dir.mkdir(exist_ok=True)
        results_dir = self.results_dir(run_dir)
        metrics = RunDirectory(run_dir).read_metrics()
        label = f"kappa={config.prior.kappa}, tau={config.prior.tau}"
        mesh = RawDataDirectory.from_configured_path(config.raw_dir).load_mesh()
        render_axial_angle_field(
            mesh,
            np.load(results_dir / SAMPLE_FILE_NAME),
            plots_dir / SAMPLE_PLOT_FILE_NAME,
            f"prior sample ({label})",
        )
        render_point_field(
            mesh,
            compute_standard_deviation_degrees(np.load(results_dir / POINTWISE_VARIANCE_FILE_NAME)),
            plots_dir / POINTWISE_STD_PLOT_FILE_NAME,
            f"pointwise axial standard deviation ({label}), mean {metrics['std_mean_deg']:.3g} deg",
            "axial standard deviation [deg]",
            "viridis",
        )
        curve = np.load(results_dir / CORRELATION_CURVE_FILE_NAME)
        fit_failure_reason = str(metrics["correlation_fit_failure_reason"]) or None
        plot_correlation_curve(
            bin_distances=curve["bin_distances"],
            bin_correlations=curve["bin_correlations"],
            fit_mask=curve["fit_mask"],
            fitted_correlations=curve["fitted_correlations"],
            fitted_length=metrics["correlation_length"],
            correlation_threshold=config.correlation.correlation_threshold,
            distance_threshold=metrics["correlation_distance_threshold"],
            path=plots_dir / CORRELATION_CURVE_PLOT_FILE_NAME,
            title=label,
            fit_failure_reason=fit_failure_reason,
        )
        plot_correlation_histogram(
            histogram_counts=curve["histogram_counts"],
            histogram_edges=curve["histogram_edges"],
            noise_floor=float(curve["noise_floor"]),
            path=plots_dir / CORRELATION_HISTOGRAM_PLOT_FILE_NAME,
            title=label,
            fit_failure_reason=fit_failure_reason,
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

        with steps.step("Building the prior"):
            prior = build_fiber_angle_prior(
                create_dolfinx_mesh(mesh),
                np.zeros(mesh.n_points),
                config.prior,
            )

        with steps.step("Drawing the visualization sample"):
            sample = shift_angles_to_minimize_axial_variance(prior.generate_sample())
        logger.info(f"      {describe_array('sample [rad]', sample)}")

        with steps.step(f"Drawing {config.num_samples} samples"):
            samples = np.stack([prior.generate_sample() for _ in range(config.num_samples)])
            _, pointwise_variance = compute_axial_mean_and_variance(samples)
        pointwise_std_degrees = compute_standard_deviation_degrees(pointwise_variance)
        logger.info(f"      {describe_array('pointwise std [deg]', pointwise_std_degrees)}")

        # Saved before the correlation length, so that they exist whatever happens to it
        np.save(results_dir / SAMPLE_FILE_NAME, sample)
        np.save(results_dir / POINTWISE_VARIANCE_FILE_NAME, pointwise_variance)
        if config.keep_samples:
            np.save(results_dir / SAMPLES_FILE_NAME, samples)

        with steps.step("Estimating the correlation length"), warnings.catch_warnings():
            # A failed fit is reported once below, in the log of the run
            warnings.simplefilter("ignore", RuntimeWarning)
            correlation = estimate_correlation_length(
                mesh.points, connectivity, samples, config.correlation
            )
        logger.info(
            f"      correlation distance : {correlation.correlation_distance_threshold:.4g}"
        )
        if correlation.fit_succeeded:
            logger.info(f"      fitted Matern length : {correlation.length:.4g}")
        else:
            logger.warning(f"Correlation length fit failed: {correlation.fit_failure_reason}")
        np.savez(
            results_dir / CORRELATION_CURVE_FILE_NAME,
            bin_distances=correlation.bin_distances,
            bin_correlations=correlation.bin_correlations,
            bin_counts=correlation.bin_counts,
            fitted_correlations=correlation.fitted_correlations,
            fit_mask=correlation.fit_mask,
            base_vertices=correlation.base_vertices,
            histogram_counts=correlation.histogram_counts,
            histogram_edges=correlation.histogram_edges,
            noise_floor=correlation.noise_floor,
        )

        return {
            "std_mean_deg": float(pointwise_std_degrees.mean()),
            "std_min_deg": float(pointwise_std_degrees.min()),
            "std_max_deg": float(pointwise_std_degrees.max()),
            "correlation_distance_threshold": float(correlation.correlation_distance_threshold),
            "correlation_length": float(correlation.length),
            "correlation_fit_succeeded": correlation.fit_succeeded,
            "correlation_fit_failure_reason": correlation.fit_failure_reason or "",
            "seconds": time.perf_counter() - start_time,
        }

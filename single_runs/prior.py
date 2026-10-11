"""Prior investigation run: sample the prior, estimate pointwise variance and correlation length.

A run is a pure function of its configuration: everything it depends on (parameters, seeds, data
source) is a field of its configuration, and nothing is hidden in module state. Configurations hold
paths and settings only, never arrays, so they can be serialized, hashed and compared.

Classes:
    PriorRunConfig: Sample the prior and estimate its pointwise variance and correlation length.
    PriorRun: The prior investigation run.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.posterior.prior import PriorParameters, build_fiber_angle_prior
from bayes_cep.postprocessing.curve_plots import plot_correlation_curve
from bayes_cep.postprocessing.mesh_plots import render_axial_angle_field, render_point_field
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import RunDirectory
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import (
    compute_axial_mean_and_variance,
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
        "bin_distances, bin_correlations, bin_counts, fitted_correlations, fit_mask, "
        "base_vertices.",
        f"results/{SAMPLES_FILE_NAME}": "All samples, shape (num_samples, num_vertices); only if "
        "keep_samples.",
        "metrics.json": "variance_mean/min/max: statistics of the pointwise variance field; "
        "correlation_distance_threshold: distance where the binned correlation first drops below "
        "the threshold (model-free); correlation_length: fitted Matern length; seconds.",
        "plots/": "sample.png, pointwise_variance.png, correlation_curve.png (study report).",
    }

    @override
    def input_files(self) -> list[Path]:
        return [RawDataDirectory.from_configured_path(self.config.raw_dir).mesh_path]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        config = self.config
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        results_dir = self.results_dir(run_dir)
        metrics = RunDirectory(run_dir).read_metrics()
        label = f"kappa={config.prior.kappa}, tau={config.prior.tau}"
        mesh = RawDataDirectory.from_configured_path(config.raw_dir).load_mesh()
        render_axial_angle_field(
            mesh,
            np.load(results_dir / SAMPLE_FILE_NAME),
            plots_dir / "sample.png",
            f"prior sample ({label})",
        )
        render_point_field(
            mesh,
            np.load(results_dir / POINTWISE_VARIANCE_FILE_NAME),
            plots_dir / "pointwise_variance.png",
            f"pointwise axial variance ({label}), mean {metrics['variance_mean']:.3g}",
            "axial variance",
            "viridis",
        )
        curve = np.load(results_dir / CORRELATION_CURVE_FILE_NAME)
        plot_correlation_curve(
            bin_distances=curve["bin_distances"],
            bin_correlations=curve["bin_correlations"],
            fit_mask=curve["fit_mask"],
            fitted_correlations=curve["fitted_correlations"],
            fitted_length=metrics["correlation_length"],
            correlation_threshold=config.correlation.correlation_threshold,
            distance_threshold=metrics["correlation_distance_threshold"],
            path=plots_dir / "correlation_curve.png",
            title=label,
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
        logger.info(f"      {describe_array('pointwise variance', pointwise_variance)}")

        with steps.step("Estimating the correlation length"):
            correlation = estimate_correlation_length(
                mesh.points, connectivity, samples, config.correlation
            )
        logger.info(
            f"      correlation distance : {correlation.correlation_distance_threshold:.4g}"
        )
        logger.info(f"      fitted Matern length : {correlation.length:.4g}")

        np.save(results_dir / SAMPLE_FILE_NAME, sample)
        np.save(results_dir / POINTWISE_VARIANCE_FILE_NAME, pointwise_variance)
        np.savez(
            results_dir / CORRELATION_CURVE_FILE_NAME,
            bin_distances=correlation.bin_distances,
            bin_correlations=correlation.bin_correlations,
            bin_counts=correlation.bin_counts,
            fitted_correlations=correlation.fitted_correlations,
            fit_mask=correlation.fit_mask,
            base_vertices=correlation.base_vertices,
        )
        if config.keep_samples:
            np.save(results_dir / SAMPLES_FILE_NAME, samples)

        return {
            "variance_mean": float(pointwise_variance.mean()),
            "variance_min": float(pointwise_variance.min()),
            "variance_max": float(pointwise_variance.max()),
            "correlation_distance_threshold": float(correlation.correlation_distance_threshold),
            "correlation_length": float(correlation.length),
            "seconds": time.perf_counter() - start_time,
        }

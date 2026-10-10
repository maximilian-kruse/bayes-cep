"""Prior investigation run: sample the prior, estimate pointwise variance and correlation length.

A run is a pure function of its configuration: everything it depends on (parameters, seeds, data
source) is a field of its configuration, and nothing is hidden in module state. Configurations hold
paths and settings only, never arrays, so they can be serialized, hashed and compared.

Classes:
    PriorParameters: Bilaplacian SPDE prior parameters; shared with the MAP run.
    PriorRunConfig: Sample the prior and estimate its pointwise variance and correlation length.
    PriorRun: The prior investigation run.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.mesh.io import create_dolfinx_mesh, load_pyvista_mesh
from bayes_cep.mesh.plotting import render_cell_field, render_point_field
from bayes_cep.posterior.prior import PriorSettings, build_fiber_angle_prior
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import RunDirectory, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import (
    compute_axial_mean_and_variance,
    shift_angles_to_minimize_axial_variance,
    wrap_axial_angles,
)
from bayes_cep.statistics.correlation_length import (
    CorrelationLengthSettings,
    estimate_correlation_length,
)

REFERENCE_PRIOR_KAPPA = 0.05
REFERENCE_PRIOR_TAU = 10.0


# ==================================================================================================
@dataclass(frozen=True)
class PriorParameters:
    r"""Bilaplacian SPDE prior parameters.

    Attributes:
        kappa (float): SPDE parameter $\kappa > 0$, controlling correlation length.
        tau (float): SPDE parameter $\tau > 0$, controlling marginal variance.
        seed (int): Random seed for prior sampling. Defaults to `0`.
    """

    kappa: float = REFERENCE_PRIOR_KAPPA
    tau: float = REFERENCE_PRIOR_TAU
    seed: int = 0


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
        keep_samples (bool): Whether to store the samples themselves as a result. Off by default,
            since many runs of a study would otherwise add up to gigabytes.
    """

    raw_dir: Path
    correlation: CorrelationLengthSettings
    prior: PriorParameters = field(default_factory=PriorParameters)
    num_samples: int = 1000
    keep_samples: bool = False

    def __post_init__(self) -> None:
        if self.num_samples < 2:
            raise ValueError(f"num_samples must be at least 2, got {self.num_samples}.")


# ==================================================================================================
class PriorRun(Run[PriorRunConfig]):
    """Sample a zero-mean prior, then estimate its pointwise variance and correlation length."""

    outputs = {
        "results/sample.npy": "One prior sample [rad] per vertex, re-branched onto a single "
        "pi-periodic branch around its axial mean; drawn first, for visualization.",
        "results/pointwise_variance.npy": "Axial variance per vertex (-ln(R)/2, with R the length "
        "of the mean resultant of the doubled angles) over the samples.",
        "results/correlation_curve.npz": "Binned correlation versus distance: bin_distances, "
        "bin_correlations, bin_counts, fitted_correlations, fit_mask, base_vertices.",
        "results/samples.npy": "All samples, shape (num_samples, num_vertices); only if "
        "keep_samples.",
        "metrics.json": "variance_mean/min/max: statistics of the pointwise variance field; "
        "correlation_distance_threshold: distance where the binned correlation first drops below "
        "the threshold (model-free); correlation_length: fitted Matern length; seconds.",
        "plots/": "sample.png, pointwise_variance.png, correlation_curve.png (study report).",
    }

    @override
    def input_files(self) -> list[Path]:
        return [resolve_repository_path(self.config.raw_dir) / "mesh.vtu"]

    @override
    def report(self, run_dir: Path) -> None:
        config = self.config
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        results_dir = run_dir / "results"
        metrics = RunDirectory(run_dir).read_metrics()
        label = f"kappa={config.prior.kappa}, tau={config.prior.tau}"

        mesh = load_pyvista_mesh(resolve_repository_path(config.raw_dir) / "mesh.vtu")
        connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
        to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(
            mesh.points, connectivity
        )
        render_cell_field(
            mesh,
            wrap_axial_angles(to_simplices @ np.load(results_dir / "sample.npy")),
            plots_dir / "sample.png",
            f"prior sample ({label})",
            "angle [rad]",
            "hsv",
            (-np.pi / 2, np.pi / 2),
        )
        render_point_field(
            mesh,
            np.load(results_dir / "pointwise_variance.npy"),
            plots_dir / "pointwise_variance.png",
            f"pointwise axial variance ({label}), mean {metrics['variance_mean']:.3g}",
            "axial variance",
            "viridis",
        )

        curve = np.load(results_dir / "correlation_curve.npz")
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(
            curve["bin_distances"], curve["bin_correlations"], "o", label="binned correlation"
        )
        fit_mask = curve["fit_mask"]
        axis.plot(
            curve["bin_distances"][fit_mask],
            curve["fitted_correlations"][fit_mask],
            "-",
            label=f"Matern fit, length {metrics['correlation_length']:.3g}",
        )
        axis.axhline(config.correlation.correlation_threshold, color="gray", linestyle=":")
        distance = metrics["correlation_distance_threshold"]
        if np.isfinite(distance):
            axis.axvline(
                distance, color="gray", linestyle="--", label=f"threshold at {distance:.3g}"
            )
        axis.set_xlabel("distance")
        axis.set_ylabel("axial correlation")
        axis.set_title(label)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plots_dir / "correlation_curve.png", dpi=150)
        plt.close(figure)

    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        start_time = time.perf_counter()
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        config = self.config
        steps = StepReporter(logger)

        raw_dir = resolve_repository_path(config.raw_dir)
        with steps.step(f"Loading the mesh from {raw_dir}"):
            mesh = load_pyvista_mesh(raw_dir / "mesh.vtu")
            connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
        logger.info(f"      mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles")

        with steps.step("Building the prior"):
            prior = build_fiber_angle_prior(
                create_dolfinx_mesh(mesh),
                PriorSettings(
                    mean_vector=np.zeros(mesh.n_points),
                    kappa=config.prior.kappa,
                    tau=config.prior.tau,
                    seed=config.prior.seed,
                ),
            )

        with steps.step("Drawing the visualization sample"):
            sample = shift_angles_to_minimize_axial_variance(prior.generate_sample()).flatten()
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

        np.save(results_dir / "sample.npy", sample)
        np.save(results_dir / "pointwise_variance.npy", pointwise_variance)
        np.savez(
            results_dir / "correlation_curve.npz",
            bin_distances=correlation.bin_distances,
            bin_correlations=correlation.bin_correlations,
            bin_counts=correlation.bin_counts,
            fitted_correlations=correlation.fitted_correlations,
            fit_mask=correlation.fit_mask,
            base_vertices=correlation.base_vertices,
        )
        if config.keep_samples:
            np.save(results_dir / "samples.npy", samples)

        return {
            "variance_mean": float(pointwise_variance.mean()),
            "variance_min": float(pointwise_variance.min()),
            "variance_max": float(pointwise_variance.max()),
            "correlation_distance_threshold": float(correlation.correlation_distance_threshold),
            "correlation_length": float(correlation.length),
            "seconds": time.perf_counter() - start_time,
        }

"""Prior investigation run: sample the prior, estimate pointwise variance and correlation length.

Classes:
    PriorRun: The prior investigation run.
"""

import time
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.mesh.io import create_dolfinx_mesh, load_pyvista_mesh
from bayes_cep.posterior.prior import PriorSettings, build_fiber_angle_prior
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.run.template import Metrics, Run
from bayes_cep.statistics.axial_statistics import (
    compute_axial_mean_and_variance,
    shift_angles_to_minimize_axial_variance,
)
from bayes_cep.statistics.correlation_length import estimate_correlation_length
from single_runs.config import PriorRunConfig
from single_runs.progress import StepReporter, describe_array


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
        # Imported here: plotting needs pyvista and matplotlib, which are not needed to run.
        from single_runs.plots import report_prior_run

        report_prior_run(self.config, run_dir)

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

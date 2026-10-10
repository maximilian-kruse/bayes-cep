"""Plots of a finished run, rendered from its results and written to `<run_dir>/plots`.

Rendering needs pyvista's off-screen mode and matplotlib, so it is a
separate step from the run itself: runs on a cluster only write data, plots are made where a
renderer is available. This module decides what to plot for which kind of run; the generic
plotting utilities are in `bayes_cep.postprocessing`.

Functions:
    report_prior_run: Write the plots of a finished prior run.
    report_map_run: Write the plots of a finished MAP run.
"""

from pathlib import Path

import numpy as np

from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.postprocessing.curve_plots import plot_optimization_history
from bayes_cep.postprocessing.mesh_plots import render_cell_field, render_point_field
from bayes_cep.run.directories import RunDirectory, resolve_repository_path
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff, wrap_axial_angles
from single_runs.config import MapRunConfig, PriorRunConfig


# ==================================================================================================
def report_prior_run(config: PriorRunConfig, run_dir: Path) -> None:
    """Write the plots of a finished prior run to `<run_dir>/plots`."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    mesh = load_pyvista_mesh(resolve_repository_path(config.raw_dir) / "mesh.vtu")
    connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
    to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(mesh.points, connectivity)
    results_dir = run_dir / "results"
    metrics = RunDirectory(run_dir).read_metrics()
    label = f"kappa={config.prior.kappa}, tau={config.prior.tau}"

    sample = np.load(results_dir / "sample.npy")
    render_cell_field(
        mesh,
        wrap_axial_angles(to_simplices @ sample),
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
    axis.plot(curve["bin_distances"], curve["bin_correlations"], "o", label="binned correlation")
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
        axis.axvline(distance, color="gray", linestyle="--", label=f"threshold at {distance:.3g}")
    axis.set_xlabel("distance")
    axis.set_ylabel("axial correlation")
    axis.set_title(label)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "correlation_curve.png", dpi=150)
    plt.close(figure)


# --------------------------------------------------------------------------------------------------
def report_map_run(config: MapRunConfig, run_dir: Path) -> None:
    """Write the plots of a finished MAP run to `<run_dir>/plots`."""
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    mesh = load_pyvista_mesh(resolve_repository_path(config.raw_dir) / "mesh.vtu")
    connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
    to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(mesh.points, connectivity)
    results_dir = run_dir / "results"
    metrics = RunDirectory(run_dir).read_metrics()

    ground_truth = np.load(results_dir / "ground_truth_angle_field.npy")
    map_estimate = np.load(results_dir / "map_estimate.npy")
    angle_clim = (-np.pi / 2, np.pi / 2)
    for name, field in (("ground_truth", ground_truth), ("map_estimate", map_estimate)):
        render_cell_field(
            mesh,
            wrap_axial_angles(to_simplices @ field),
            plots_dir / f"{name}.png",
            name.replace("_", " "),
            "angle [rad]",
            "hsv",
            angle_clim,
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

    plot_optimization_history(
        np.load(results_dir / "map_loss_history.npy"),
        np.load(results_dir / "map_gradient_norm_history.npy"),
        plots_dir / "optimization_history.png",
    )

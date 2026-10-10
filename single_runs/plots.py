"""Plots of a finished run, rendered from its results and written to `<run_dir>/plots`.

Rendering needs pyvista's off-screen mode and matplotlib, so it is a
separate step from the run itself: runs on a cluster only write data, plots are made where a
renderer is available.

Functions:
    report_prior_run: Write the plots of a finished prior run.
    report_map_run: Write the plots of a finished MAP run.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.run.directories import RunDirectory, resolve_repository_path
from bayes_cep.statistics.axial_statistics import compute_axial_data_diff
from single_runs.config import MapRunConfig, PriorRunConfig

WINDOW_SIZE = (1000, 800)


# ==================================================================================================
def _wrap_axial(angles: np.ndarray) -> np.ndarray:
    r"""Wrap angles onto $(-\pi/2, \pi/2]$, the principal branch of axial data."""
    return np.angle(np.exp(2j * angles)) / 2


# --------------------------------------------------------------------------------------------------
def _render(
    mesh: pv.UnstructuredGrid,
    path: Path,
    title: str,
    cmap: str,
    clim: tuple[float, float] | None = None,
) -> None:
    """Render the active scalars of `mesh` off-screen into a PNG."""
    plotter = pv.Plotter(off_screen=True, window_size=list(WINDOW_SIZE))
    plotter.add_mesh(mesh, cmap=cmap, clim=clim)
    plotter.add_text(title, font_size=10)
    plotter.screenshot(path)
    plotter.close()


# --------------------------------------------------------------------------------------------------
def _render_cell_field(
    mesh: pv.UnstructuredGrid,
    simplex_values: np.ndarray,
    path: Path,
    title: str,
    name: str,
    cmap: str,
    clim: tuple[float, float] | None = None,
) -> None:
    plot_mesh = mesh.copy()
    plot_mesh.cell_data[name] = simplex_values
    plot_mesh.set_active_scalars(name)
    _render(plot_mesh, path, title, cmap, clim)


# --------------------------------------------------------------------------------------------------
def _render_point_field(
    mesh: pv.UnstructuredGrid,
    vertex_values: np.ndarray,
    path: Path,
    title: str,
    name: str,
    cmap: str,
) -> None:
    plot_mesh = mesh.copy()
    plot_mesh.point_data[name] = vertex_values
    plot_mesh.set_active_scalars(name)
    _render(plot_mesh, path, title, cmap)


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
    _render_cell_field(
        mesh,
        _wrap_axial(to_simplices @ sample),
        plots_dir / "sample.png",
        f"prior sample ({label})",
        "angle [rad]",
        "hsv",
        (-np.pi / 2, np.pi / 2),
    )
    _render_point_field(
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
    import matplotlib.pyplot as plt

    mesh = load_pyvista_mesh(resolve_repository_path(config.raw_dir) / "mesh.vtu")
    connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
    to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(mesh.points, connectivity)
    results_dir = run_dir / "results"
    metrics = RunDirectory(run_dir).read_metrics()

    ground_truth = np.load(results_dir / "ground_truth_angle_field.npy")
    map_estimate = np.load(results_dir / "map_estimate.npy")
    angle_clim = (-np.pi / 2, np.pi / 2)
    for name, field in (("ground_truth", ground_truth), ("map_estimate", map_estimate)):
        _render_cell_field(
            mesh,
            _wrap_axial(to_simplices @ field),
            plots_dir / f"{name}.png",
            name.replace("_", " "),
            "angle [rad]",
            "hsv",
            angle_clim,
        )
    difference = to_simplices @ compute_axial_data_diff(map_estimate, ground_truth)
    limit = float(np.abs(difference).max())
    _render_cell_field(
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

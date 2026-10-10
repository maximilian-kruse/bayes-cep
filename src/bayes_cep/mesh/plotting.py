"""Off-screen rendering of scalar fields on a triangle mesh into PNG files.

Rendering needs pyvista's off-screen mode, so plotting is a separate step from computing: runs on
a cluster only write data, plots are made where a renderer is available.

Functions:
    render_cell_field: Render one value per simplex into a PNG.
    render_point_field: Render one value per vertex into a PNG.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

WINDOW_SIZE = (1000, 800)


# ==================================================================================================
def render_cell_field(
    mesh: pv.UnstructuredGrid,
    simplex_values: np.ndarray,
    path: Path,
    title: str,
    name: str,
    cmap: str,
    clim: tuple[float, float] | None = None,
) -> None:
    """Render one value per simplex of `mesh` into the PNG `path`.

    Args:
        mesh (pv.UnstructuredGrid): The mesh; it is not modified.
        simplex_values (np.ndarray): One value per simplex, shape `(num_simplices,)`.
        path (Path): Target PNG file.
        title (str): Text drawn into the image.
        name (str): Name of the field (label of the color bar).
        cmap (str): Name of the matplotlib colormap.
        clim (tuple[float, float] | None): Color limits; the data range if `None`.
    """
    plot_mesh = mesh.copy()
    plot_mesh.cell_data[name] = simplex_values
    plot_mesh.set_active_scalars(name)
    _render_active_scalars(plot_mesh, path, title, cmap, clim)


# --------------------------------------------------------------------------------------------------
def render_point_field(
    mesh: pv.UnstructuredGrid,
    vertex_values: np.ndarray,
    path: Path,
    title: str,
    name: str,
    cmap: str,
) -> None:
    """Render one value per vertex of `mesh` into the PNG `path`.

    Args:
        mesh (pv.UnstructuredGrid): The mesh; it is not modified.
        vertex_values (np.ndarray): One value per vertex, shape `(num_vertices,)`.
        path (Path): Target PNG file.
        title (str): Text drawn into the image.
        name (str): Name of the field (label of the color bar).
        cmap (str): Name of the matplotlib colormap.
    """
    plot_mesh = mesh.copy()
    plot_mesh.point_data[name] = vertex_values
    plot_mesh.set_active_scalars(name)
    _render_active_scalars(plot_mesh, path, title, cmap)


# --------------------------------------------------------------------------------------------------
def _render_active_scalars(
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

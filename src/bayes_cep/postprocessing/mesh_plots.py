"""Scalar fields on a triangular surface mesh, rendered off-screen into PNG files.

Functions:
    render_cell_field: Render a field with one value per simplex.
    render_point_field: Render a field with one value per vertex.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

WINDOW_SIZE = (1000, 800)
TITLE_FONT_SIZE = 10


# ==================================================================================================
def render_cell_field(
    mesh: pv.UnstructuredGrid,
    simplex_values: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
    title: str,
    name: str,
    cmap: str,
    clim: tuple[float, float] | None = None,
) -> None:
    """Render a field with one value per simplex of `mesh` into a PNG.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        simplex_values (np.ndarray): Values, shape `(num_simplices,)`.
        path (Path): Output PNG file.
        title (str): Text in the corner of the image.
        name (str): Name of the field, shown at the color bar.
        cmap (str): Matplotlib color map name.
        clim (tuple[float, float] | None): Color range. Defaults to the range of the values.
    """
    plot_mesh = mesh.copy()
    plot_mesh.cell_data[name] = simplex_values
    plot_mesh.set_active_scalars(name)
    _render(plot_mesh, path, title, cmap, clim)


# --------------------------------------------------------------------------------------------------
def render_point_field(
    mesh: pv.UnstructuredGrid,
    vertex_values: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
    title: str,
    name: str,
    cmap: str,
    clim: tuple[float, float] | None = None,
) -> None:
    """Render a field with one value per vertex of `mesh` into a PNG.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        vertex_values (np.ndarray): Values, shape `(num_vertices,)`.
        path (Path): Output PNG file.
        title (str): Text in the corner of the image.
        name (str): Name of the field, shown at the color bar.
        cmap (str): Matplotlib color map name.
        clim (tuple[float, float] | None): Color range. Defaults to the range of the values.
    """
    plot_mesh = mesh.copy()
    plot_mesh.point_data[name] = vertex_values
    plot_mesh.set_active_scalars(name)
    _render(plot_mesh, path, title, cmap, clim)


# ==================================================================================================
def _render(
    mesh: pv.UnstructuredGrid,
    path: Path,
    title: str,
    cmap: str,
    clim: tuple[float, float] | None,
) -> None:
    """Render the active scalars of `mesh` off-screen into a PNG."""
    plotter = pv.Plotter(off_screen=True, window_size=list(WINDOW_SIZE))
    plotter.add_mesh(mesh, cmap=cmap, clim=clim)
    plotter.add_text(title, font_size=TITLE_FONT_SIZE)
    plotter.screenshot(path)
    plotter.close()

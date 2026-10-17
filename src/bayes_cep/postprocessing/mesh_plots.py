"""Scalar fields on a triangular surface mesh, rendered off-screen into image files.

The format follows the suffix of the output path: `.pdf`, `.svg`, `.eps` and `.ps` are written as
vector graphics (`VECTOR_SUFFIXES`, for publications), any other suffix (`.png`) as a raster image.

Functions:
    render_cell_field: Render a field with one value per simplex.
    render_point_field: Render a field with one value per vertex.
    render_vertex_field_on_simplices: Render a per-vertex field with one value per simplex.
    render_axial_angle_field: Render a per-vertex fiber-angle field.
    render_axial_angle_field_pair: Render two per-vertex fiber-angle fields side by side.
"""

from pathlib import Path

import numpy as np
import pyvista as pv

from bayes_cep.mesh.interpolation import NearestNeighborInterpolationStrategy
from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.statistics.axial_statistics import wrap_axial_angles

WINDOW_SIZE = (1000, 800)
VECTOR_SUFFIXES = (".pdf", ".svg", ".eps", ".ps")
TITLE_FONT_SIZE = 10
ANGLE_NAME = "angle [rad]"


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
    """Render a field with one value per simplex of `mesh` into an image file.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        simplex_values (np.ndarray): Values, shape `(num_simplices,)`.
        path (Path): Output file; see the module docstring for the formats.
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
    """Render a field with one value per vertex of `mesh` into an image file.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        vertex_values (np.ndarray): Values, shape `(num_vertices,)`.
        path (Path): Output file; see the module docstring for the formats.
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
def render_vertex_field_on_simplices(
    mesh: pv.UnstructuredGrid,
    vertex_values: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
    title: str,
    name: str,
    cmap: str,
    clim: tuple[float, float] | None = None,
) -> None:
    """Render a field with one value per vertex of `mesh` as one value per simplex into a file.

    Each simplex shows the value of the vertex nearest to its centroid, so that the plot has the
    resolution of the piecewise-constant fields the forward model works with.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        vertex_values (np.ndarray): Values, shape `(num_vertices,)`.
        path (Path): Output file; see the module docstring for the formats.
        title (str): Text in the corner of the image.
        name (str): Name of the field, shown at the color bar.
        cmap (str): Matplotlib color map name.
        clim (tuple[float, float] | None): Color range. Defaults to the range of the values.
    """
    to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(
        mesh.points, get_triangle_connectivity(mesh)
    )
    render_cell_field(mesh, to_simplices @ vertex_values, path, title, name, cmap, clim)


# --------------------------------------------------------------------------------------------------
def render_axial_angle_field(
    mesh: pv.UnstructuredGrid,
    vertex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
    title: str,
) -> None:
    r"""Render a per-vertex fiber-angle field, wrapped onto $(-\pi/2, \pi/2]$, into an image file.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        vertex_angles (np.ndarray): Angles in radians, shape `(num_vertices,)`.
        path (Path): Output file; see the module docstring for the formats.
        title (str): Text in the corner of the image.
    """
    render_vertex_field_on_simplices(
        mesh,
        wrap_axial_angles(vertex_angles),
        path,
        title,
        ANGLE_NAME,
        "hsv",
        (-np.pi / 2, np.pi / 2),
    )


# --------------------------------------------------------------------------------------------------
def render_axial_angle_field_pair(
    mesh: pv.UnstructuredGrid,
    left_vertex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    right_vertex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    path: Path,
    left_title: str,
    right_title: str,
) -> None:
    r"""Render two per-vertex fiber-angle fields side by side, e.g. the truth and an estimate.

    Both fields are wrapped onto $(-\pi/2, \pi/2]$ and share the cyclic color map and range, so
    that the same color means the same orientation in both.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        left_vertex_angles (np.ndarray): Angles in radians, shape `(num_vertices,)`.
        right_vertex_angles (np.ndarray): Angles in radians, shape `(num_vertices,)`.
        path (Path): Output file; see the module docstring for the formats.
        left_title (str): Text in the corner of the left panel.
        right_title (str): Text in the corner of the right panel.
    """
    to_simplices = NearestNeighborInterpolationStrategy().assemble_matrix(
        mesh.points, get_triangle_connectivity(mesh)
    )
    plotter = pv.Plotter(
        off_screen=True, shape=(1, 2), window_size=[2 * WINDOW_SIZE[0], WINDOW_SIZE[1]]
    )
    panels = ((left_vertex_angles, left_title), (right_vertex_angles, right_title))
    for column, (vertex_angles, title) in enumerate(panels):
        plot_mesh = mesh.copy()
        plot_mesh.cell_data[ANGLE_NAME] = to_simplices @ wrap_axial_angles(vertex_angles)
        plot_mesh.set_active_scalars(ANGLE_NAME)
        plotter.subplot(0, column)
        plotter.add_mesh(plot_mesh, cmap="hsv", clim=(-np.pi / 2, np.pi / 2))
        plotter.add_text(title, font_size=TITLE_FONT_SIZE)
    _save_plotter(plotter, path)


# ==================================================================================================
def _render(
    mesh: pv.UnstructuredGrid,
    path: Path,
    title: str,
    cmap: str,
    clim: tuple[float, float] | None,
) -> None:
    """Render the active scalars of `mesh` off-screen into a vector or raster image, by the
    suffix of `path`."""
    plotter = pv.Plotter(off_screen=True, window_size=list(WINDOW_SIZE))
    plotter.add_mesh(mesh, cmap=cmap, clim=clim)
    plotter.add_text(title, font_size=TITLE_FONT_SIZE)
    _save_plotter(plotter, path)


# --------------------------------------------------------------------------------------------------
def _save_plotter(plotter: pv.Plotter, path: Path) -> None:
    """Write the scene of `plotter` into a vector or raster image, by the suffix of `path`, and
    close it."""
    if path.suffix.lower() in VECTOR_SUFFIXES:
        plotter.save_graphic(path)
    else:
        plotter.screenshot(path)
    plotter.close()

"""Topology of triangular surface meshes.

Functions:
    get_triangle_connectivity: Vertex indices of the triangles of a triangular surface mesh.
    find_unique_edges: Unique undirected edges of a triangular mesh and their multiplicity.
    find_boundary_vertices: Vertices on the mesh boundary.
"""

import numpy as np
import pyvista as pv


# ==================================================================================================
def get_triangle_connectivity(
    pv_mesh: pv.UnstructuredGrid,
) -> np.ndarray[tuple[int, int], np.dtype[np.integer]]:
    """Vertex indices of the triangles of a triangular surface mesh.

    Assumes `pv_mesh` has already been validated as purely triangular, e.g. via
    [`load_pyvista_mesh`][bayes_cep.mesh.io.load_pyvista_mesh].

    Args:
        pv_mesh (pv.UnstructuredGrid): Triangular surface mesh.

    Returns:
        np.ndarray: Connectivity, shape `(num_simplices, 3)`.
    """
    return pv_mesh.cells.reshape(-1, 4)[:, 1:]


# --------------------------------------------------------------------------------------------------
def find_unique_edges(
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.integer]], np.ndarray[tuple[int], np.dtype[np.integer]]
]:
    """Find the unique undirected edges of a triangular mesh and how often each occurs.

    Args:
        connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.

    Returns:
        tuple[np.ndarray, np.ndarray]: The edges as sorted vertex pairs, shape `(num_edges, 2)`,
            and the number of triangles sharing each edge, shape `(num_edges,)`.
    """
    edges = np.concatenate(
        (connectivity[:, [0, 1]], connectivity[:, [1, 2]], connectivity[:, [2, 0]]), axis=0
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return unique_edges, counts


# --------------------------------------------------------------------------------------------------
def find_boundary_vertices(
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
) -> np.ndarray[tuple[int], np.dtype[np.integer]]:
    """Find the vertices on the mesh boundary, i.e. on an edge shared by only one triangle.

    Args:
        connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.

    Returns:
        np.ndarray: Sorted indices of boundary vertices; empty for a closed surface.
    """
    edges, counts = find_unique_edges(connectivity)
    return np.unique(edges[counts == 1])

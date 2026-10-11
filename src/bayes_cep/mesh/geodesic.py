"""Geodesic (mesh-graph) distances on triangular surface meshes.

Distances are shortest paths along mesh edges, weighted by edge length. They overestimate the true
geodesic distance by a few percent (a path is restricted to edges) but, unlike Euclidean distances,
follow the curvature of the surface.

Functions:
    assemble_edge_length_graph: Sparse symmetric graph of mesh edges, weighted by edge length.
    compute_graph_distances: Shortest-path distances from source vertices to all vertices.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from bayes_cep.mesh.topology import find_unique_edges


# ==================================================================================================
def assemble_edge_length_graph(
    vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
) -> sp.csr_array:
    """Assemble the sparse symmetric graph of mesh edges, weighted by edge length.

    Args:
        vertex_coordinates (np.ndarray): Mesh vertex coordinates, shape `(num_vertices, dim)`.
        connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.

    Returns:
        sp.csr_array: Symmetric edge-length matrix, shape `(num_vertices, num_vertices)`.
    """
    num_vertices = vertex_coordinates.shape[0]
    edges, _ = find_unique_edges(connectivity)
    lengths = np.linalg.norm(
        vertex_coordinates[edges[:, 0]] - vertex_coordinates[edges[:, 1]], axis=1
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    return sp.csr_array(
        (np.concatenate((lengths, lengths)), (rows, columns)), shape=(num_vertices, num_vertices)
    )


# --------------------------------------------------------------------------------------------------
def compute_graph_distances(
    graph: sp.csr_array,
    source_vertices: np.ndarray[tuple[int], np.dtype[np.integer]],
    max_distance: float = np.inf,
    min_over_sources: bool = False,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Compute shortest-path distances from source vertices to all vertices.

    Args:
        graph (sp.csr_array): Edge-length graph, see
            [`assemble_edge_length_graph`][bayes_cep.mesh.geodesic.assemble_edge_length_graph].
        source_vertices (np.ndarray): Source vertex indices, shape `(num_sources,)`.
        max_distance (float): Distances beyond this are not computed and set to `inf`. Defaults to
            `inf`.
        min_over_sources (bool): If `True`, return only the distance to the nearest source.
            Defaults to `False`.

    Returns:
        np.ndarray: Distances, shape `(num_sources, num_vertices)`, or `(num_vertices,)` if
            `min_over_sources`. Unreached vertices are `inf`.
    """
    return dijkstra(
        graph,
        directed=False,
        indices=source_vertices,
        limit=max_distance,
        min_only=min_over_sources,
    )

"""Geodesic (mesh-graph) distances on triangular surface meshes.

Distances are shortest paths along mesh edges, weighted by edge length. They overestimate the true
geodesic distance by a few percent (a path is restricted to edges) but, unlike Euclidean distances,
follow the curvature of the surface.

Functions:
    assemble_edge_length_graph: Sparse symmetric graph of mesh edges, weighted by edge length.
    assemble_centroid_edge_length_graph: The same graph extended by one node per triangle centroid.
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
def assemble_centroid_edge_length_graph(
    vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
) -> sp.csr_array:
    """Assemble the edge-length graph extended by one node per triangle centroid.

    The nodes `0, ..., num_vertices - 1` are the vertices and the nodes `num_vertices + i` the
    centroid of triangle `i`. Besides the mesh edges, a centroid is connected to its three vertices
    and to the centroids of the triangles sharing an edge with it, each weighted by the Euclidean
    distance. Shortest paths between centroids then do not have to detour over vertices, which
    matters for distances of the order of the edge length.

    Args:
        vertex_coordinates (np.ndarray): Mesh vertex coordinates, shape `(num_vertices, dim)`.
        connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.

    Returns:
        sp.csr_array: Symmetric matrix, shape
            `(num_vertices + num_simplices, num_vertices + num_simplices)`.
    """
    num_vertices = vertex_coordinates.shape[0]
    num_simplices = connectivity.shape[0]
    centroids = vertex_coordinates[connectivity].mean(axis=1)
    centroid_nodes = num_vertices + np.arange(num_simplices)

    centroid_vertex_rows = np.repeat(centroid_nodes, 3)
    centroid_vertex_columns = connectivity.reshape(-1)
    centroid_vertex_lengths = np.linalg.norm(
        centroids[np.repeat(np.arange(num_simplices), 3)]
        - vertex_coordinates[centroid_vertex_columns],
        axis=1,
    )

    # Two triangles share an edge exactly when their sorted half-edges coincide.
    half_edges = np.sort(
        np.concatenate(
            (connectivity[:, [0, 1]], connectivity[:, [1, 2]], connectivity[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    owners = np.tile(np.arange(num_simplices), 3)
    order = np.lexsort((half_edges[:, 1], half_edges[:, 0]))
    sorted_edges = half_edges[order]
    sorted_owners = owners[order]
    shared = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    first_owner = sorted_owners[:-1][shared]
    second_owner = sorted_owners[1:][shared]
    neighbor_lengths = np.linalg.norm(centroids[first_owner] - centroids[second_owner], axis=1)

    num_nodes = num_vertices + num_simplices
    vertex_graph = assemble_edge_length_graph(vertex_coordinates, connectivity)
    extension = sp.csr_array(
        (
            np.concatenate((centroid_vertex_lengths, neighbor_lengths)),
            (
                np.concatenate((centroid_vertex_rows, num_vertices + first_owner)),
                np.concatenate((centroid_vertex_columns, num_vertices + second_owner)),
            ),
        ),
        shape=(num_nodes, num_nodes),
    )
    centroid_block = sp.csr_array((num_simplices, num_simplices))
    padded_vertex_graph = sp.block_diag((vertex_graph, centroid_block))
    return sp.csr_array(padded_vertex_graph + extension + extension.T)


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

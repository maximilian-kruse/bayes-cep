"""Strategies for interpolating a vertex-based field onto piecewise-constant per-simplex values.

Classes:
    InterpolationStrategy: ABC interface for vertex-to-simplex interpolation.
    LinearInterpolationStrategy: Average a triangle's three vertex values.
    NearestNeighborInterpolationStrategy: Take the value of the vertex closest to each triangle's
        centroid.

Note:
    The concrete strategies are (stateless) dataclasses on purpose: `tyro` only exposes a class as
    a selectable subcommand (e.g. `--interpolation:linear-interpolation-strategy`) in the scripts'
    CLIs if it is a dataclass; otherwise it freezes the argument to the default instance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


# ==================================================================================================
class InterpolationStrategy(ABC):
    """ABC interface for vertex-to-simplex interpolation.

    Implementations assemble a sparse matrix $M \\in \\mathbb{R}^{N_S \\times N_V}$, mapping a
    vertex-based field $x$ to per-simplex values $y = Mx$, e.g. to move a vertex-based fiber-angle
    parameter onto piecewise-constant cells

    Methods:
        assemble_matrix: Assemble the vertex-to-simplex interpolation matrix.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def assemble_matrix(
        self,
        vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
    ) -> sp.coo_array:
        """Assemble the vertex-to-simplex interpolation matrix.

        Args:
            vertex_coordinates (np.ndarray): Mesh vertex coordinates, shape
                `(num_vertices, dim)`.
            connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.

        Returns:
            sp.coo_array: Interpolation matrix, shape `(num_simplices, num_vertices)`.
        """


# ==================================================================================================
@dataclass(frozen=True)
class LinearInterpolationStrategy(InterpolationStrategy):
    """Average a triangle's three vertex values.

    This is the exact value of the linear (P1) interpolant of the vertex field at the triangle's
    centroid. Does not use `vertex_coordinates`; only the connectivity determines the weights.
    """

    # ----------------------------------------------------------------------------------------------
    def assemble_matrix(
        self,
        vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
    ) -> sp.coo_array:
        """Assemble the interpolation matrix, with weight `1/3` per triangle vertex."""
        num_vertices = int(np.max(connectivity)) + 1
        num_simplices = connectivity.shape[0]
        row_inds = np.repeat(np.arange(num_simplices), 3)
        col_inds = connectivity.flatten()
        data = np.full(num_simplices * 3, 1 / 3)
        return sp.coo_array((data, (row_inds, col_inds)), shape=(num_simplices, num_vertices))


# ==================================================================================================
@dataclass(frozen=True)
class NearestNeighborInterpolationStrategy(InterpolationStrategy):
    """Take the value of the vertex closest to each triangle's centroid."""

    # ----------------------------------------------------------------------------------------------
    def assemble_matrix(
        self,
        vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
    ) -> sp.coo_array:
        """Assemble the interpolation matrix, with a single unit weight per triangle."""
        num_vertices = vertex_coordinates.shape[0]
        num_simplices = connectivity.shape[0]
        triangle_vertex_coords = vertex_coordinates[connectivity]
        centroids = triangle_vertex_coords.mean(axis=1, keepdims=True)
        distances_to_centroid = np.linalg.norm(triangle_vertex_coords - centroids, axis=-1)
        nearest_local_ind = np.argmin(distances_to_centroid, axis=1)
        row_inds = np.arange(num_simplices)
        col_inds = connectivity[row_inds, nearest_local_ind]
        data = np.ones(num_simplices)
        return sp.coo_array((data, (row_inds, col_inds)), shape=(num_simplices, num_vertices))

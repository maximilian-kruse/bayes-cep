r"""Fiber-vector-to-angle conversion and the per-vertex angle field it induces.

Functions:
    compute_angle_from_fiber_vectors: Per-simplex fiber angle from 3D fiber vectors.
    interpolate_simplex_field_to_vertices: Area-weighted axial simplex-to-vertex angle
        interpolation.
    build_angle_field_from_fiber_field: Per-vertex angle field from raw patient fiber data.
"""

import numpy as np
import pyvista as pv
import scipy.sparse as sp

from bayes_cep.statistics.axial_statistics import shift_angles_to_minimize_axial_variance


# ==================================================================================================
def compute_angle_from_fiber_vectors(
    fiber_vectors: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    basis_vector_one: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    basis_vector_two: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Compute the per-simplex fiber angle $\theta$ from 3D fiber vectors.

    Inverse of the convention [`FiberTensor`][bayes_cep.posterior.fiber_tensor.FiberTensor] uses to
    reconstruct the fiber direction from an angle, $f = \cos(\theta) e_1 + \sin(\theta) e_2$:
    $\theta = \operatorname{atan2}(f \cdot e_2,\, f \cdot e_1)$.

    Args:
        fiber_vectors (np.ndarray): Per-simplex 3D fiber vectors, shape `(num_simplices, 3)`.
        basis_vector_one (np.ndarray): First local tangent-plane basis vector per simplex, shape
            `(num_simplices, 3)`.
        basis_vector_two (np.ndarray): Second local tangent-plane basis vector per simplex, shape
            `(num_simplices, 3)`.

    Returns:
        np.ndarray: Per-simplex fiber angle in radians, shape `(num_simplices,)`.
    """
    angle = np.arctan2(
        np.einsum("ij,ij->i", fiber_vectors, basis_vector_two),
        np.einsum("ij,ij->i", fiber_vectors, basis_vector_one),
    )
    return angle


# --------------------------------------------------------------------------------------------------
def interpolate_simplex_field_to_vertices(
    mesh: pv.UnstructuredGrid,
    simplex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Area-weight-interpolate a per-simplex axial angle field onto mesh vertices.

    Each vertex value is the area-weighted axial (mod-$\pi$) mean of its incident triangles'
    angles, computed via the mean resultant vector of the doubled angles (see
    `axial_statistics`'s module docstring) rather than a plain arithmetic mean: neighboring
    simplices can report angles that differ by up to $\pi$ for the same physical fiber orientation
    (e.g. $10°$ vs. $-170°$), and a plain average across such a branch cut is arbitrarily wrong (up
    to $\pi/2$ off), not something a later global re-branching step can undo per vertex.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh.
        simplex_angles (np.ndarray): Per-simplex angle field in radians, shape `(num_simplices,)`.

    Returns:
        np.ndarray: Per-vertex axial angle field, shape `(num_vertices,)`, in $(-\pi/2, \pi/2]$.
    """
    connectivity = mesh.cells.reshape(-1, 4)[:, 1:]
    num_vertices = mesh.points.shape[0]
    num_simplices = connectivity.shape[0]
    simplex_areas = np.asarray(mesh.compute_cell_sizes().cell_data["Area"], dtype=np.float64)

    row_inds = connectivity.flatten()
    col_inds = np.repeat(np.arange(num_simplices), 3)
    data = np.repeat(simplex_areas, 3)
    weighted_adjacency = sp.coo_array(
        (data, (row_inds, col_inds)), shape=(num_vertices, num_simplices)
    )
    area_sum_per_vertex = np.asarray(weighted_adjacency.sum(axis=1)).ravel()
    simplex_to_vertex_matrix = sp.diags_array(1 / area_sum_per_vertex) @ weighted_adjacency

    doubled_angle_unit_vectors = np.exp(2j * simplex_angles)
    vertex_mean_resultant = simplex_to_vertex_matrix @ doubled_angle_unit_vectors
    return np.angle(vertex_mean_resultant) / 2


# --------------------------------------------------------------------------------------------------
def build_angle_field_from_fiber_field(
    mesh: pv.UnstructuredGrid,
    fiber_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Build the per-vertex fiber-angle field from one patient's raw data.

    Converts per-simplex fiber vectors to angles, area-weight-interpolates them to vertices, then
    resolves the mod-$\pi$ branch ambiguity globally via `shift_angles_to_minimize_axial_variance`
    (a single axial mean over the whole field).

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh.
        fiber_field (np.ndarray): Per-simplex 3D fiber vectors, shape `(num_simplices, 3)`.
        basis_vectors (np.ndarray): Local tangent-plane basis vectors per simplex, shape
            `(num_simplices, 3, 2)`; `basis_vectors[..., 0]`/`basis_vectors[..., 1]` are the two
            basis vectors the fiber angle is measured against.

    Returns:
        np.ndarray: Per-vertex ground-truth fiber angle field, shape `(num_vertices,)`.
    """
    simplex_angles = compute_angle_from_fiber_vectors(
        fiber_field, basis_vectors[..., 0], basis_vectors[..., 1]
    )
    vertex_angles = interpolate_simplex_field_to_vertices(mesh, simplex_angles)
    return shift_angles_to_minimize_axial_variance(vertex_angles, axis=1).flatten()

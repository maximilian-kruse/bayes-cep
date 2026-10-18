r"""Smoothing of an axial fiber-angle field on a triangular surface mesh.

A fiber direction is axial, so angles and vectors cannot be averaged directly: a vector and its
negative are the same fiber, and their average is zero. The field is therefore smoothed as the
doubled-angle unit vector $z = e^{2i\theta}$, which is the same for $\theta$ and $\theta + \pi$, and
the smoothed angle is half the angle of the smoothed $z$.

The smoothing is the heat equation on the surface, $\partial_t z = \Delta z$, for the time
$t = s^2 / 2$ (a Gaussian of standard deviation $s$ in the surface distance, in the flat limit). It
follows the surface, so a neighborhood does not reach across a thin wall or between two nearby
veins, and the boundary of the mesh has no flux. The Laplacian is the finite element one of the
mesh (cotangent weights) with a lumped mass matrix, and the time is split into several implicit
steps, which approximates the Gaussian kernel much better than one step.

The linear smoothing blurs the broad structure together with narrow artifacts. The adaptive
variant makes the diffusivity grow with the gradient of $z$, $g = 1 + (|\nabla z| / g_0)^2$, so that
narrow strips with a steep rotation of the fiber are removed quickly and smooth regions hardly
change. Optionally a confidence weight per vertex is smoothed along with the field (normalized
convolution), so that vertices of low confidence do not pull the average.

The vector median filter is a different kind of smoothing: every vertex takes the doubled-angle
vector of the member of its neighborhood (a few rings of the mesh) that is closest to all others,
which removes structures narrower than about half the neighborhood completely, while broad
transitions stay. It is robust against outliers and keeps unit vectors, but leaves plateaus, so a
light linear smoothing can follow.

Functions:
    smooth_axial_angle_field: Smooth a per-vertex angle field; also returns the local coherence.
    smooth_axial_angle_field_median: Vector median filter on the doubled-angle vectors, optionally
        followed by a linear smoothing.
    smooth_axial_angle_field_adaptive: The same with a gradient-dependent diffusivity and optional
        confidence weights.
    estimate_gradient_scale: A quantile of the gradient of the doubled-angle vector over the edges,
        as the gradient scale of the adaptive smoothing.
    smooth_fiber_field: Smooth a per-simplex fiber vector field; also returns the vertex angles and
        the coherence.
"""

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from bayes_cep.mesh.topology import get_triangle_connectivity
from bayes_cep.preprocessing.angle_field import (
    compute_angle_from_fiber_vectors,
    interpolate_simplex_field_to_vertices,
)

COTANGENT_DEGENERACY_TOLERANCE = 1e-12
WEIGHT_FLOOR = 1e-12
MEDIAN_CHUNK_SIZE = 2000


# ==================================================================================================
def smooth_axial_angle_field(
    mesh: pv.UnstructuredGrid,
    vertex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    scale: float,
    num_steps: int = 8,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
    r"""Smooth a per-vertex axial angle field on a surface mesh.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        vertex_angles (np.ndarray): Angles in radians, shape `(num_vertices,)`. Only the angle
            modulo $\pi$ matters.
        scale (float): Standard deviation $s$ of the Gaussian smoothing, in the units of the mesh
            coordinates. `0` returns the field wrapped onto $(-\pi/2, \pi/2]$, with full coherence.
        num_steps (int): Number of implicit time steps the smoothing time is split into. More steps
            approximate the Gaussian kernel better, one step gives a heavier-tailed kernel.
            Defaults to `8`.

    Returns:
        tuple[np.ndarray, np.ndarray]: The smoothed angles in $(-\pi/2, \pi/2]$ and the coherence
            $|z_s| \in [0, 1]$, both of shape `(num_vertices,)`. The coherence is one where the
            neighboring directions agree and near zero where they cancel, and the angle is not
            meaningful there.

    Raises:
        ValueError: If `vertex_angles` does not have one entry per vertex, `scale` is negative or
            `num_steps` is not positive.
    """
    if vertex_angles.shape != (mesh.n_points,):
        raise ValueError(
            f"Got angles of shape {vertex_angles.shape} for a mesh with {mesh.n_points} vertices."
        )
    if scale < 0.0:
        raise ValueError(f"The smoothing scale must not be negative, got {scale}.")
    if num_steps <= 0:
        raise ValueError(f"The number of steps must be positive, got {num_steps}.")

    doubled_angle_vectors = np.stack(
        [np.cos(2.0 * vertex_angles), np.sin(2.0 * vertex_angles)], axis=1
    )
    if scale > 0.0:
        stiffness_matrix, lumped_mass = _assemble_stiffness_and_lumped_mass(
            mesh.points, get_triangle_connectivity(mesh)
        )
        step_time = scale**2 / 2.0 / num_steps
        step_matrix = (sp.diags_array(lumped_mass) + step_time * stiffness_matrix).tocsc()
        factorization = splu(step_matrix)
        for _ in range(num_steps):
            doubled_angle_vectors = factorization.solve(
                lumped_mass[:, None] * doubled_angle_vectors
            )
    coherence = np.linalg.norm(doubled_angle_vectors, axis=1)
    smoothed_angles = 0.5 * np.arctan2(doubled_angle_vectors[:, 1], doubled_angle_vectors[:, 0])
    return smoothed_angles, coherence


# ==================================================================================================
def estimate_gradient_scale(
    mesh: pv.UnstructuredGrid,
    vertex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    quantile: float,
) -> float:
    r"""A quantile of the gradient of the doubled-angle vector over the edges of the mesh.

    The gradient of an edge is $|z_i - z_j| / \ell_{ij}$, in units of one per mesh unit. Gradients
    above the quantile are treated as steep by `smooth_axial_angle_field_adaptive`.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh.
        vertex_angles (np.ndarray): Angles in radians, shape `(num_vertices,)`.
        quantile (float): Quantile in $(0, 1)$, e.g. `0.9`.

    Returns:
        float: The gradient scale.
    """
    first, second, _, _ = _assemble_corner_stencils(mesh.points, get_triangle_connectivity(mesh))
    doubled_angle_vectors = np.stack(
        [np.cos(2.0 * vertex_angles), np.sin(2.0 * vertex_angles)], axis=1
    )
    return float(
        np.quantile(
            _compute_edge_gradients(mesh.points, doubled_angle_vectors, first, second), quantile
        )
    )


# ==================================================================================================
def smooth_axial_angle_field_adaptive(
    mesh: pv.UnstructuredGrid,
    vertex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    scale: float,
    gradient_scale: float,
    num_steps: int = 8,
    confidence: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
    r"""Smooth a per-vertex axial angle field with a diffusivity that grows with the gradient.

    As `smooth_axial_angle_field`, but each implicit step uses the cotangent weight of every edge
    times $g = 1 + (|\nabla z| / g_0)^2$, with the gradient of the current doubled-angle vector $z$
    on the edge and $g_0$ = `gradient_scale`. Steep strips diffuse much faster than smooth regions,
    which stay close to the linear smoothing with the same `scale`.

    With `confidence` $w$, the numerator $w z$ and the denominator $w$ are smoothed by the same
    operator and the field is their ratio, so a vertex with a small weight hardly contributes.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        vertex_angles (np.ndarray): Angles in radians, shape `(num_vertices,)`.
        scale (float): Standard deviation of the smoothing in the smooth regions, in mesh units.
            `0` returns the field wrapped onto $(-\pi/2, \pi/2]$.
        gradient_scale (float): The gradient $g_0$ where the diffusivity is doubled, in one per mesh
            unit; see `estimate_gradient_scale`. A very large value gives the linear smoothing.
        num_steps (int): Number of implicit time steps. Defaults to `8`.
        confidence (np.ndarray | None): Non-negative weight per vertex, shape `(num_vertices,)`;
            `None` weighs all vertices equally.

    Returns:
        tuple[np.ndarray, np.ndarray]: The smoothed angles in $(-\pi/2, \pi/2]$ and the
            coherence $|z_s|$, both of shape `(num_vertices,)`.

    Raises:
        ValueError: If a shape does not have one entry per vertex, `scale` is negative,
            `gradient_scale` is not positive, `num_steps` is not positive or `confidence` is
            negative.
    """
    if vertex_angles.shape != (mesh.n_points,):
        raise ValueError(
            f"Got angles of shape {vertex_angles.shape} for a mesh with {mesh.n_points} vertices."
        )
    if scale < 0.0:
        raise ValueError(f"The smoothing scale must not be negative, got {scale}.")
    if gradient_scale <= 0.0:
        raise ValueError(f"The gradient scale must be positive, got {gradient_scale}.")
    if num_steps <= 0:
        raise ValueError(f"The number of steps must be positive, got {num_steps}.")
    weights = np.ones(mesh.n_points) if confidence is None else confidence
    if weights.shape != (mesh.n_points,) or np.any(weights < 0.0):
        raise ValueError("The confidence needs one non-negative entry per vertex.")

    doubled_angle_vectors = np.stack(
        [np.cos(2.0 * vertex_angles), np.sin(2.0 * vertex_angles)], axis=1
    )
    if scale > 0.0:
        first, second, half_cotangent, lumped_mass = _assemble_corner_stencils(
            mesh.points, get_triangle_connectivity(mesh)
        )
        step_time = scale**2 / 2.0 / num_steps
        # Numerator (two components) and denominator of the normalized convolution
        state = np.column_stack([weights[:, None] * doubled_angle_vectors, weights])
        for _ in range(num_steps):
            current = state[:, :2] / np.maximum(state[:, 2:], WEIGHT_FLOOR)
            gradients = _compute_edge_gradients(mesh.points, current, first, second)
            diffusivity = 1.0 + (gradients / gradient_scale) ** 2
            stiffness_matrix = _assemble_stiffness_matrix(
                first, second, diffusivity * half_cotangent, mesh.n_points
            )
            step_matrix = (sp.diags_array(lumped_mass) + step_time * stiffness_matrix).tocsc()
            state = splu(step_matrix).solve(lumped_mass[:, None] * state)
        doubled_angle_vectors = state[:, :2] / np.maximum(state[:, 2:], WEIGHT_FLOOR)
    coherence = np.linalg.norm(doubled_angle_vectors, axis=1)
    smoothed_angles = 0.5 * np.arctan2(doubled_angle_vectors[:, 1], doubled_angle_vectors[:, 0])
    return smoothed_angles, coherence


# ==================================================================================================
def smooth_axial_angle_field_median(
    mesh: pv.UnstructuredGrid,
    vertex_angles: np.ndarray[tuple[int], np.dtype[np.float64]],
    num_rings: int,
    num_iterations: int,
    linear_scale: float = 0.0,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]], np.ndarray[tuple[int], np.dtype[np.float64]]
]:
    r"""Smooth a per-vertex axial angle field with a vector median filter.

    Each iteration replaces the doubled-angle vector of every vertex by the member of its
    neighborhood (the vertex and its neighbors up to `num_rings` edges away) with the smallest sum
    of distances to the other members (the vector median of Astola et al.). Structures narrower
    than about half the neighborhood are removed, and broad transitions are kept. The neighborhood
    counts edges, not lengths, so its size follows the mesh resolution. A linear smoothing of
    `linear_scale` can follow, to remove the plateaus the median leaves.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        vertex_angles (np.ndarray): Angles in radians, shape `(num_vertices,)`.
        num_rings (int): Size of the neighborhood in edges; `1` takes the direct neighbors.
        num_iterations (int): Number of median filter passes.
        linear_scale (float): Scale of the linear smoothing after the filter, in the units of the
            mesh coordinates; `0` for none. Defaults to `0`.

    Returns:
        tuple[np.ndarray, np.ndarray]: The smoothed angles in $(-\pi/2, \pi/2]$ and the coherence
            $|z_s|$, both of shape `(num_vertices,)`. Without the linear smoothing the coherence
            is one everywhere.

    Raises:
        ValueError: If `vertex_angles` does not have one entry per vertex, `num_rings` or
            `num_iterations` is not positive, or `linear_scale` is negative.
    """
    if vertex_angles.shape != (mesh.n_points,):
        raise ValueError(
            f"Got angles of shape {vertex_angles.shape} for a mesh with {mesh.n_points} vertices."
        )
    if num_rings <= 0 or num_iterations <= 0:
        raise ValueError("The numbers of rings and iterations must be positive.")
    if linear_scale < 0.0:
        raise ValueError(f"The linear scale must not be negative, got {linear_scale}.")

    neighbors, is_member = _collect_ring_neighborhoods(mesh, num_rings)
    doubled_angle_vectors = np.stack(
        [np.cos(2.0 * vertex_angles), np.sin(2.0 * vertex_angles)], axis=1
    )
    for _ in range(num_iterations):
        doubled_angle_vectors = _apply_vector_median(doubled_angle_vectors, neighbors, is_member)
    angles = 0.5 * np.arctan2(doubled_angle_vectors[:, 1], doubled_angle_vectors[:, 0])
    if linear_scale > 0.0:
        return smooth_axial_angle_field(mesh, angles, linear_scale)
    return angles, np.ones(mesh.n_points)


# ==================================================================================================
def smooth_fiber_field(
    mesh: pv.UnstructuredGrid,
    fiber_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    scale: float,
    num_steps: int = 8,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    r"""Smooth a per-simplex fiber vector field of a surface mesh.

    The vectors are converted to angles against the basis vectors, moved to the vertices (axial
    area-weighted mean), smoothed by `smooth_axial_angle_field`, and moved back to the simplices as
    the axial mean of the three vertex angles. The result is a drop-in replacement of `fiber_field`
    for the preprocessing; the vertex angles are the smoothed field before that last averaging.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh; not modified.
        fiber_field (np.ndarray): Per-simplex 3D fiber vectors, shape `(num_simplices, 3)`.
        basis_vectors (np.ndarray): Local tangent-plane basis vectors per simplex, shape
            `(num_simplices, 3, 2)`.
        scale (float): Standard deviation of the Gaussian smoothing, in the units of the mesh
            coordinates.
        num_steps (int): Number of implicit time steps. Defaults to `8`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The smoothed unit fiber vectors, shape
            `(num_simplices, 3)`, the smoothed vertex angles in $(-\pi/2, \pi/2]$ and the vertex
            coherence, both of shape `(num_vertices,)`.
    """
    simplex_angles = compute_angle_from_fiber_vectors(
        fiber_field, basis_vectors[..., 0], basis_vectors[..., 1]
    )
    vertex_angles, coherence = smooth_axial_angle_field(
        mesh, interpolate_simplex_field_to_vertices(mesh, simplex_angles), scale, num_steps
    )
    connectivity = get_triangle_connectivity(mesh)
    smoothed_simplex_angles = 0.5 * np.angle(
        np.mean(np.exp(2j * vertex_angles[connectivity]), axis=1)
    )
    smoothed_fiber_field = (
        np.cos(smoothed_simplex_angles)[:, None] * basis_vectors[..., 0]
        + np.sin(smoothed_simplex_angles)[:, None] * basis_vectors[..., 1]
    )
    return smoothed_fiber_field, vertex_angles, coherence


# ==================================================================================================
def _assemble_corner_stencils(
    vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.integer]],
    np.ndarray[tuple[int], np.dtype[np.integer]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Per triangle corner, the opposite edge and its cotangent weight, and the lumped mass.

    The weight of an edge is half the sum of the cotangents of the two angles opposite to it, so
    the finite element Laplacian is assembled from the half cotangents of both triangles of an edge.
    A triangle with an obtuse angle gives a negative weight, which is correct for the finite
    element operator.

    Args:
        vertex_coordinates (np.ndarray): Mesh vertex coordinates, shape `(num_vertices, 3)`.
        connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The two end vertices of the edge
            opposite to every corner and half the cotangent of the corner angle, each of shape
            `(3 * num_simplices,)`, and a third of the area of the adjacent triangles per vertex,
            shape `(num_vertices,)`.
    """
    lumped_mass = np.zeros(vertex_coordinates.shape[0])
    firsts, seconds, half_cotangents = [], [], []
    for corner in range(3):
        apex = connectivity[:, corner]
        first = connectivity[:, (corner + 1) % 3]
        second = connectivity[:, (corner + 2) % 3]
        edge_to_first = vertex_coordinates[first] - vertex_coordinates[apex]
        edge_to_second = vertex_coordinates[second] - vertex_coordinates[apex]
        double_area = np.linalg.norm(np.cross(edge_to_first, edge_to_second), axis=1)
        cotangent = np.einsum("ij,ij->i", edge_to_first, edge_to_second) / np.maximum(
            double_area, COTANGENT_DEGENERACY_TOLERANCE
        )
        # The edge opposite to the apex connects `first` and `second`
        firsts.append(first)
        seconds.append(second)
        half_cotangents.append(0.5 * cotangent)
        np.add.at(lumped_mass, apex, double_area / 6.0)
    return (
        np.concatenate(firsts),
        np.concatenate(seconds),
        np.concatenate(half_cotangents),
        lumped_mass,
    )


# --------------------------------------------------------------------------------------------------
def _assemble_stiffness_matrix(
    first: np.ndarray[tuple[int], np.dtype[np.integer]],
    second: np.ndarray[tuple[int], np.dtype[np.integer]],
    edge_weights: np.ndarray[tuple[int], np.dtype[np.float64]],
    num_vertices: int,
) -> sp.csr_array:
    """Assemble the stiffness matrix (positive semidefinite for positive weights) from the weights
    of the corner edges, so that `K @ u` is the weighted Laplacian of the vertex field `u`."""
    return sp.coo_array(
        (
            np.concatenate([-edge_weights, -edge_weights, edge_weights, edge_weights]),
            (
                np.concatenate([first, second, first, second]),
                np.concatenate([second, first, first, second]),
            ),
        ),
        shape=(num_vertices, num_vertices),
    ).tocsr()


# --------------------------------------------------------------------------------------------------
def _compute_edge_gradients(
    vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    vertex_vectors: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    first: np.ndarray[tuple[int], np.dtype[np.integer]],
    second: np.ndarray[tuple[int], np.dtype[np.integer]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Gradient magnitude of a vector field along each corner edge: |difference| / length."""
    edge_length = np.linalg.norm(vertex_coordinates[first] - vertex_coordinates[second], axis=1)
    difference = np.linalg.norm(vertex_vectors[first] - vertex_vectors[second], axis=1)
    return difference / np.maximum(edge_length, COTANGENT_DEGENERACY_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def _assemble_stiffness_and_lumped_mass(
    vertex_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    connectivity: np.ndarray[tuple[int, int], np.dtype[np.integer]],
) -> tuple[sp.csr_array, np.ndarray[tuple[int], np.dtype[np.float64]]]:
    """Assemble the cotangent stiffness matrix and the lumped (vertex area) mass vector.

    Args:
        vertex_coordinates (np.ndarray): Mesh vertex coordinates, shape `(num_vertices, 3)`.
        connectivity (np.ndarray): Triangle vertex indices, shape `(num_simplices, 3)`.

    Returns:
        tuple[sp.csr_array, np.ndarray]: The stiffness matrix, shape `(num_vertices,
            num_vertices)`, and a third of the area of the adjacent triangles per vertex, shape
            `(num_vertices,)`.
    """
    first, second, half_cotangent, lumped_mass = _assemble_corner_stencils(
        vertex_coordinates, connectivity
    )
    return (
        _assemble_stiffness_matrix(first, second, half_cotangent, vertex_coordinates.shape[0]),
        lumped_mass,
    )


# --------------------------------------------------------------------------------------------------
def _collect_ring_neighborhoods(
    mesh: pv.UnstructuredGrid, num_rings: int
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.integer]],
    np.ndarray[tuple[int, int], np.dtype[np.bool_]],
]:
    """The vertices within `num_rings` edges of every vertex (itself included), padded.

    Returns:
        tuple[np.ndarray, np.ndarray]: The neighbor indices, shape `(num_vertices, max_size)`
            (padded with the vertex itself) and the mask of the true members, same shape.
    """
    connectivity = get_triangle_connectivity(mesh)
    edges = np.concatenate(
        [connectivity[:, [0, 1]], connectivity[:, [1, 2]], connectivity[:, [2, 0]]]
    )
    num_vertices = mesh.n_points
    adjacency = sp.coo_array(
        (
            np.ones(2 * edges.shape[0]),
            (
                np.concatenate([edges[:, 0], edges[:, 1]]),
                np.concatenate([edges[:, 1], edges[:, 0]]),
            ),
        ),
        shape=(num_vertices, num_vertices),
    ).tocsr()
    adjacency.data[:] = 1.0
    step = (adjacency + sp.identity(num_vertices, format="csr")).tocsr()
    reach = step.copy()
    for _ in range(num_rings - 1):
        reach = (reach @ step).tocsr()
        reach.data[:] = 1.0
    sizes = np.diff(reach.indptr)
    max_size = int(sizes.max())
    neighbors = np.repeat(np.arange(num_vertices)[:, None], max_size, axis=1)
    is_member = np.zeros((num_vertices, max_size), dtype=bool)
    for vertex in range(num_vertices):
        members = reach.indices[reach.indptr[vertex] : reach.indptr[vertex + 1]]
        neighbors[vertex, : members.size] = members
        is_member[vertex, : members.size] = True
    return neighbors, is_member


# --------------------------------------------------------------------------------------------------
def _apply_vector_median(
    vectors: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    neighbors: np.ndarray[tuple[int, int], np.dtype[np.integer]],
    is_member: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Replace each vector by the member of its neighborhood with the smallest sum of distances."""
    result = np.empty_like(vectors)
    for start in range(0, vectors.shape[0], MEDIAN_CHUNK_SIZE):
        chunk = slice(start, start + MEDIAN_CHUNK_SIZE)
        members = vectors[neighbors[chunk]]  # (chunk, size, 2)
        mask = is_member[chunk]
        distances = np.linalg.norm(members[:, :, None, :] - members[:, None, :, :], axis=3)
        distances = distances * mask[:, None, :]
        total = distances.sum(axis=2)
        total[~mask] = np.inf
        best = np.argmin(total, axis=1)
        result[chunk] = members[np.arange(members.shape[0]), best]
    return result

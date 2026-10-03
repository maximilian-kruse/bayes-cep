r"""Eikonal activation-time forward map for fiber-orientation inference.

Classes:
    EikonalSolverSettings: Settings for the eikonax forward solver, derivator, and tensor field.
    EikonalParameterToSolutionMap: `ls_bayesian` `ParameterToSolutionMap` wrapping eikonax.
"""

from dataclasses import dataclass
from typing import override

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from eikonax import derivator, linalg, preprocessing, solver, tensorfield
from ls_bayesian.posterior import interfaces

from bayes_cep.mesh.interpolation import (
    InterpolationStrategy,
    NearestNeighborInterpolationStrategy,
)
from bayes_cep.posterior.fiber_tensor import FiberTensor, FiberTensorSettings


# ==================================================================================================
@dataclass(frozen=True)
class EikonalSolverSettings:
    """Settings for the eikonax forward solver, derivator, and anisotropic tensor field.

    Attributes:
        initial_site_ind (int): Vertex index of the (single) pacing/activation site.
        longitudinal_velocity (float): Conduction velocity along the fiber direction.
        transversal_velocity (float): Conduction velocity transverse to the fiber direction.
        solver_tolerance (float): Convergence tolerance for the forward solve.
        max_num_iterations (int): Maximum number of solver iterations.
        max_value (float): Initial/ceiling value for the solution vector.
        use_soft_update (bool): Whether to use a soft minmax approximation for differentiability
            of the solver/derivator updates.
        softminmax_order (int): Order of the soft minmax approximation used for differentiability.
        softminmax_cutoff (float): Cutoff distance for the soft minmax approximation.
    """

    initial_site_ind: int
    longitudinal_velocity: float
    transversal_velocity: float
    solver_tolerance: float = 1e-6
    max_num_iterations: int = 1000
    max_value: float = 1000.0
    use_soft_update: bool = True
    softminmax_order: int = 20
    softminmax_cutoff: float = 0.01


# ==================================================================================================
class _DerivativeCache:
    """Caches the eikonax adjoint solver for the most recently seen parameter vector.

    `LogPosterior` evaluates the forward map and then its gradient at the same parameter vector in
    consecutive calls; this avoids repeating the adjoint system assembly in that case.
    """

    def __init__(self) -> None:
        self._parameter_vector: np.ndarray | None = None
        self._derivative_solver: derivator.DerivativeSolver | None = None
        self._sparse_partial_parameter: sp.coo_array | None = None

    def get(
        self, parameter_vector: np.ndarray
    ) -> tuple[derivator.DerivativeSolver, sp.coo_array] | None:
        """Return the cached derivative structures if they match `parameter_vector`, else `None`."""
        if self._parameter_vector is None or not np.allclose(
            parameter_vector, self._parameter_vector
        ):
            return None
        return self._derivative_solver, self._sparse_partial_parameter

    def set(
        self,
        parameter_vector: np.ndarray,
        derivative_solver: derivator.DerivativeSolver,
        sparse_partial_parameter: sp.coo_array,
    ) -> None:
        """Cache the derivative structures for `parameter_vector`."""
        self._parameter_vector = parameter_vector
        self._derivative_solver = derivative_solver
        self._sparse_partial_parameter = sparse_partial_parameter


# ==================================================================================================
class EikonalParameterToSolutionMap(interfaces.ParameterToSolutionMap):
    r"""Eikonal activation-time forward map $u = F(m)$ for a per-vertex fiber-angle parameter.

    Wraps `eikonax`'s differentiable eikonal solver as an `ls_bayesian` `ParameterToSolutionMap`:
    the parameter $m$ is a fiber-orientation angle per mesh vertex, interpolated to a per-simplex
    anisotropic conductivity tensor via `FiberTensor` before each forward solve. Gradients are
    computed via eikonax's discrete adjoint.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        pv_mesh: pv.UnstructuredGrid,
        basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
        settings: EikonalSolverSettings,
        interpolation_strategy: InterpolationStrategy | None = None,
    ) -> None:
        r"""Build the eikonax solver/derivator and the vertex-to-simplex fiber tensor field.

        Args:
            pv_mesh (pv.UnstructuredGrid): Triangular surface mesh, vertex order matching the
                parameter vectors this map is called with.
            basis_vectors (np.ndarray): Local tangent-plane basis vectors per simplex, shape
                `(num_simplices, 3, 2)`; `basis_vectors[..., 0]`/`basis_vectors[..., 1]` are the
                two basis vectors a fiber angle is measured against.
            settings (EikonalSolverSettings): Forward solver and tensor field settings.
            interpolation_strategy (InterpolationStrategy | None, optional): Strategy for
                interpolating the vertex-based parameter onto simplices. Defaults to
                `NearestNeighborInterpolationStrategy` if `None`.
        """
        vertices = pv_mesh.points
        simplices = pv_mesh.cells.reshape(-1, 4)[:, 1:]
        interpolation_strategy = interpolation_strategy or NearestNeighborInterpolationStrategy()
        self._vertex_to_simplex_matrix = interpolation_strategy.assemble_matrix(vertices, simplices)

        num_simplices = simplices.shape[0]
        fiber_tensor_settings = FiberTensorSettings(
            dimension=3,
            basis_vectors_one=basis_vectors[..., 0],
            basis_vectors_two=basis_vectors[..., 1],
            longitudinal_velocities=settings.longitudinal_velocity * np.ones(num_simplices),
            transversal_velocities=settings.transversal_velocity * np.ones(num_simplices),
        )
        self._tensor_field = tensorfield.TensorField(
            num_simplices=num_simplices,
            vector_to_simplices_map=tensorfield.LinearScalarMap(),
            simplex_tensor=FiberTensor(fiber_tensor_settings),
        )

        mesh_data = preprocessing.MeshData(vertices, simplices)
        initial_sites = preprocessing.InitialSites(inds=(settings.initial_site_ind,), values=(0,))
        solver_data = solver.SolverData(
            loop_type="jitted_while",
            max_value=settings.max_value,
            use_soft_update=settings.use_soft_update,
            softminmax_order=settings.softminmax_order,
            softminmax_cutoff=settings.softminmax_cutoff,
            max_num_iterations=settings.max_num_iterations,
            tolerance=settings.solver_tolerance,
        )
        derivator_data = derivator.PartialDerivatorData(
            use_soft_update=settings.use_soft_update,
            softminmax_order=settings.softminmax_order,
            softminmax_cutoff=settings.softminmax_cutoff,
        )
        self._solver = solver.Solver(mesh_data, solver_data, initial_sites)
        self._derivator = derivator.PartialDerivator(mesh_data, derivator_data, initial_sites)
        self._cache = _DerivativeCache()

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_forward(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Solve the eikonal equation for the activation-time field $u = F(m)$."""
        tensor_field_instance = self._assemble_tensor_field(parameter_vector)
        solution = self._solver.run(tensor_field_instance)
        return np.asarray(solution.values, dtype=np.float64)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the transposed Jacobian $(\nabla_m F(m))^T v$ via eikonax's discrete adjoint."""
        derivative_solver, sparse_partial_parameter = self._get_derivative_structures(
            solution_vector, parameter_vector
        )
        adjoint_solution = derivative_solver.solve(adjoint_vector)
        return adjoint_solution.T @ sparse_partial_parameter @ self._vertex_to_simplex_matrix

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_jacobian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Not implemented.

        eikonax's `DerivativeSolver` only exposes the adjoint (transposed) triangular solve needed
        for `evaluate_gradient`, not the forward triangular solve a Jacobian-vector product would
        need. Not required for MAP estimation, which only uses `evaluate_forward` and
        `evaluate_gradient`.
        """
        raise NotImplementedError(
            "EikonalParameterToSolutionMap does not support Jacobian-vector products."
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        adjoint_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        gradient_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Not implemented: second-order forward-map contributions are not required for MAP."""
        raise NotImplementedError(
            "EikonalParameterToSolutionMap does not support Hessian-vector products."
        )

    # ----------------------------------------------------------------------------------------------
    def _assemble_tensor_field(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray:
        """Interpolate the vertex-based parameter to cells and assemble the conductivity tensor."""
        parameter_vector_on_cells = self._vertex_to_simplex_matrix @ parameter_vector
        return self._tensor_field.assemble_field(parameter_vector_on_cells)

    # ----------------------------------------------------------------------------------------------
    def _get_derivative_structures(
        self,
        solution_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> tuple[derivator.DerivativeSolver, sp.coo_array]:
        """Build (or fetch cached) adjoint solver and parameter-Jacobian for `parameter_vector`."""
        cached = self._cache.get(parameter_vector)
        if cached is not None:
            return cached
        parameter_vector_on_cells = self._vertex_to_simplex_matrix @ parameter_vector
        tensor_field_instance = self._tensor_field.assemble_field(parameter_vector_on_cells)
        output_partial_solution, output_partial_tensor = (
            self._derivator.compute_partial_derivatives(solution_vector, tensor_field_instance)
        )
        tensor_partial_parameter = self._tensor_field.assemble_jacobian(parameter_vector_on_cells)
        output_partial_parameter = linalg.contract_derivative_tensors(
            output_partial_tensor, tensor_partial_parameter
        )
        sparse_partial_solution = linalg.convert_to_scipy_sparse(output_partial_solution)
        sparse_partial_parameter = linalg.convert_to_scipy_sparse(output_partial_parameter)
        derivative_solver = derivator.DerivativeSolver(solution_vector, sparse_partial_solution)
        self._cache.set(parameter_vector, derivative_solver, sparse_partial_parameter)
        return derivative_solver, sparse_partial_parameter

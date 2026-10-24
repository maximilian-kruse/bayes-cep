r"""Gauss-Newton Hessian of the data misfit at a fixed parameter.

Classes:
    GaussNewtonMisfitHessian: Matrix-free action of $H = J^T \nabla_u^2 \Phi\, J$.
"""

import numpy as np
from ls_bayesian.posterior import interfaces


# ==================================================================================================
class GaussNewtonMisfitHessian:
    r"""Gauss-Newton Hessian $H = J^T \nabla_u^2 \Phi(u)\, J$ of the data misfit at fixed $m$.

    $J = \nabla_m F(m)$ is the Jacobian of the forward map and $\Phi$ the negative log-likelihood
    as a function of the solution $u$. For the Gaussian likelihood
    $\nabla_u^2 \Phi = B^T \Gamma_\mathrm{noise}^{-1} B$, so $H = J^T B^T \Gamma_\mathrm{noise}^{-1}
    B J$ is symmetric positive semi-definite. It omits the term with the second derivative of
    $F$, and is exact for a linear forward map. The action costs one Jacobian-vector product
    (forward triangular solve), one likelihood Hessian-vector product and one transposed Jacobian
    product (adjoint solve). The forward solution is computed once, in the constructor.

    The operator acts on the coefficient vector $\hat m$ itself (not on a Riesz representer) and is
    the `hessian_action` of `ls_bayesian.lowrank.compute_generalized_eigenpairs` together with
    the prior precision as $Q$.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        forward_map: interfaces.ParameterToSolutionMap,
        likelihood: interfaces.Likelihood,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Evaluate the forward solution at `parameter_vector` and fix the linearization point.

        Args:
            forward_map (interfaces.ParameterToSolutionMap): Forward map providing
                `evaluate_jacobian_vector_product` and the transposed product `evaluate_gradient`.
            likelihood (interfaces.Likelihood): Likelihood providing
                `evaluate_hessian_vector_product`.
            parameter_vector (np.ndarray): Linearization point $m$ (e.g. the MAP estimate), shape
                `(num_parameters,)`.

        Raises:
            ValueError: If `parameter_vector` is not one-dimensional.
        """
        if parameter_vector.ndim != 1:
            raise ValueError(
                f"parameter_vector must be one-dimensional, got shape {parameter_vector.shape}."
            )
        self._forward_map = forward_map
        self._likelihood = likelihood
        self._parameter_vector = np.array(parameter_vector, dtype=np.float64, copy=True)
        self._solution_vector = forward_map.evaluate_forward(self._parameter_vector)

    # ----------------------------------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Number of parameters, the dimension of the operator."""
        return self._parameter_vector.size

    # ----------------------------------------------------------------------------------------------
    def apply(
        self, direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply $H \hat m$.

        Args:
            direction_vector (np.ndarray): Direction $\hat m$, shape `(num_parameters,)`.

        Returns:
            np.ndarray: $H \hat m$, shape `(num_parameters,)`, float64.

        Raises:
            ValueError: If `direction_vector` does not have shape `(num_parameters,)`.
        """
        if direction_vector.shape != self._parameter_vector.shape:
            raise ValueError(
                f"direction_vector must have shape {self._parameter_vector.shape}, "
                f"got {direction_vector.shape}."
            )
        direction_vector = np.asarray(direction_vector, dtype=np.float64)
        solution_direction = self._forward_map.evaluate_jacobian_vector_product(
            self._solution_vector, self._parameter_vector, direction_vector
        )
        weighted_solution_direction = self._likelihood.evaluate_hessian_vector_product(
            self._solution_vector, solution_direction
        )
        return np.asarray(
            self._forward_map.evaluate_gradient(
                self._solution_vector, self._parameter_vector, weighted_solution_direction
            ),
            dtype=np.float64,
        )

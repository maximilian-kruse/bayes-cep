r"""Adapts a `LogPosterior` to the `OptimizationModel` interface.

Classes:
    CameronMartinPosteriorModel: `OptimizationModel` expressing a `LogPosterior`'s gradient in the
        prior's Cameron-Martin inner product.
    EuclideanPosteriorModel: `OptimizationModel` exposing a `LogPosterior`'s gradient
        unchanged, for optimizers that assume the standard Euclidean inner product.
"""

from typing import override

import numpy as np
from ls_bayesian.optimization.model import OptimizationModel
from ls_bayesian.posterior import interfaces
from ls_bayesian.posterior.posterior import LogPosterior


# ==================================================================================================
class CameronMartinPosteriorModel(OptimizationModel):
    r"""Expresses a `LogPosterior`'s gradient in the prior's Cameron-Martin inner product
    $(\cdot,\cdot)_{\mathrm{CM}} = (\mathcal{C}_\text{prior}^{-1}\cdot,\cdot)$.

    The dual (functional-space) gradient is the sum of a likelihood contribution $d_\text{lik}(m) =
    J_G(m)^T\Gamma_\text{noise}^{-1}(\mathcal{B}\mathcal{G}(m)-y)$ (the raw transposed-Jacobian
    action from `EikonalParameterToSolutionMap.evaluate_gradient`, no mass-matrix inverse, returned
    by `LogPosterior.evaluate_likelihood_gradient`) and a prior contribution
    $\mathcal{C}_\text{prior}^{-1}(m-\bar m)$ (what `LogPosterior.evaluate_prior_gradient`/
    `FiberAnglePrior.evaluate_gradient` would return). The Cameron-Martin representer is
    $g_{\mathrm{CM}}(m) = \mathcal{C}_\text{prior}\,(d_\text{lik}(m) +
    \mathcal{C}_\text{prior}^{-1}(m-\bar m)) = \mathcal{C}_\text{prior}\,d_\text{lik}(m) +
    (m-\bar m)$: the prior term collapses to the identity algebraically, so `evaluate_gradient`
    below adds it directly as $m-\bar m$ (via `prior.mean_vector`) rather than calling
    `evaluate_prior_gradient` and applying `apply_covariance_operator` to
    $\mathcal{C}_\text{prior}^{-1}(m-\bar m)$.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, log_posterior: LogPosterior, prior: interfaces.GaussianPrior) -> None:
        r"""Wrap a `LogPosterior` and the `GaussianPrior` it was built with.

        Args:
            log_posterior (LogPosterior): Negative log-posterior to minimize.
            prior (interfaces.GaussianPrior): The same prior `log_posterior` was built with;
                supplies the Cameron-Martin covariance/precision operators, e.g.
                `PosteriorBuilder.prior`.
        """
        self._log_posterior = log_posterior
        self._prior = prior

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        return self._log_posterior.evaluate_cost(parameter_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        likelihood_gradient = self._log_posterior.evaluate_likelihood_gradient(parameter_vector)
        return self._prior.apply_covariance_operator(likelihood_gradient) + (
            parameter_vector - self._prior.mean_vector
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_inner_product(
        self,
        first_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        second_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        return float(first_vector @ self._prior.apply_precision_operator(second_vector))


# ==================================================================================================
class EuclideanPosteriorModel(OptimizationModel):
    r"""Exposes a `LogPosterior`'s gradient unchanged, for optimizers that assume the
    standard Euclidean inner product (e.g. `ScipyLBFGSBOptimizer`).

    `LogPosterior.evaluate_gradient` is itself a "dual" (functional-space) vector, not a genuine
    Euclidean Riesz representer -- see `CameronMartinPosteriorModel`'s docstring, and the discrete
    full-space gradient would need an extra inverse-mass-matrix application to become one. This
    class does not perform that correction, resembling a naive discretize-then-optimize approach.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, log_posterior: LogPosterior) -> None:
        """Wrap a `LogPosterior`.

        Args:
            log_posterior (LogPosterior): Negative log-posterior to minimize.
        """
        self._log_posterior = log_posterior

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        return self._log_posterior.evaluate_cost(parameter_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._log_posterior.evaluate_gradient(parameter_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Not implemented: `LogPosterior.evaluate_hessian_vector_product` isn't either, and
        `ScipyLBFGSBOptimizer` never calls it (`requires_hessian = False`)."""
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_inner_product(
        self,
        first_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        second_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        """Standard Euclidean dot product. Never called by `ScipyLBFGSBOptimizer` itself (hardcoded
        to Euclidean geometry); provided only to satisfy the `OptimizationModel` interface."""
        return float(first_vector @ second_vector)

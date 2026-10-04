r"""Adapters from a `LogPosterior` and its prior to `ls_bayesian`'s MCMC measure interfaces.

The posterior $\mu$ is sampled relative to the prior as reference measure $\mu_0 = \mathcal
N(\bar m, C)$, i.e. $\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(m))$ with $\Phi$ the negative
log-likelihood of the forward-mapped parameter, $\Phi(m) = \Phi(F(m))$.

Classes:
    LikelihoodTargetMeasure: `DifferentiableTargetMeasure` for $\Phi$, from a `LogPosterior`.
    PriorGaussianMeasure: `GaussianMeasure` for the prior, from a `GaussianPrior`.
"""

from typing import override

import numpy as np
from ls_bayesian.mcmc.measures import DifferentiableTargetMeasure, GaussianMeasure
from ls_bayesian.posterior import interfaces
from ls_bayesian.posterior.posterior import LogPosterior


# ==================================================================================================
class LikelihoodTargetMeasure(DifferentiableTargetMeasure):
    r"""Potential $\Phi(m) = \Phi(F(m))$ and its gradient, relative to the prior as reference
    measure.

    The gradient is `LogPosterior.evaluate_likelihood_gradient`, a dual vector: contracted with a
    coefficient vector it gives a directional derivative, and applying the prior's covariance to it
    gives the Cameron-Martin representer $C\nabla\Phi$. This is exactly how
    [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm] uses $\nabla\Phi$, so no
    further transformation is needed here.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, log_posterior: LogPosterior) -> None:
        """Wrap a `LogPosterior`; only its likelihood contribution is ever evaluated."""
        self._log_posterior = log_posterior

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_potential(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        return self._log_posterior.evaluate_likelihood_cost(state)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._log_posterior.evaluate_likelihood_gradient(state)


# ==================================================================================================
class PriorGaussianMeasure(GaussianMeasure):
    r"""The prior $\mathcal N(\bar m, C)$ as a `GaussianMeasure`, by delegation to a
    `GaussianPrior`, following the same decoupling pattern as
    [`FiberAnglePrior`][bayes_cep.posterior.prior.FiberAnglePrior].

    `evaluate_cost` is inherited: it is concrete in terms of `mean` and `apply_precision_operator`.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, prior: interfaces.GaussianPrior) -> None:
        """Wrap the same `GaussianPrior` the `LogPosterior` was built with."""
        self._prior = prior

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def mean(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.mean_vector

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def random_vector_size(self) -> int:
        return self._prior.random_vector_size

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.apply_covariance_factorization(random_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.apply_covariance_operator(vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.apply_precision_operator(vector)

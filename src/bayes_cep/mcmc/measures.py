r"""Adapters from a `LogPosterior` and its prior to `ls_bayesian`'s MCMC measure interfaces.

The posterior $\mu$ is sampled relative to the prior as reference measure $\mu_0 = \mathcal
N(\bar m, C)$, i.e. $\frac{d\mu}{d\mu_0} \propto \exp(-\Phi(m))$ with $\Phi$ the negative
log-likelihood of the forward-mapped parameter, $\Phi(m) = \Phi(F(m))$.

Classes:
    LikelihoodTargetMeasure: `DifferentiableTargetMeasure` for $\Phi$, from a `LogPosterior`.
    MeanTiltedTargetMeasure: The potential relative to the centred prior, for a prior mean.
    PriorGaussianMeasure: `GaussianMeasure` for the prior, from a `GaussianPrior`.
    CenteredGaussianMeasure: The same Gaussian with its mean moved to zero, the centred reference
        measure.
    LowRankGaussianMeasure: `GaussianMeasure` for a low-rank Gaussian such as a Laplace
        approximation, from a `LowRankGaussian`.
"""

from typing import override

import numpy as np
from ls_bayesian.lowrank.gaussian import LowRankGaussian
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
class MeanTiltedTargetMeasure(DifferentiableTargetMeasure):
    r"""The potential of the target relative to the *centred* prior $\mathcal N(0, C)$, for a prior
    $\mathcal N(\bar m, C)$.

    The target has the density $\propto \exp(-\Phi(u))$ relative to $\mathcal N(\bar m, C)$. By the
    Cameron-Martin formula for translations of a Gaussian measure (Da Prato and Zabczyk,
    Theorem 2.21; for $\bar m$ in the Cameron-Martin space, which always holds after
    discretization),

    $$
    \frac{d\mathcal N(\bar m, C)}{d\mathcal N(0, C)}(u) = \exp\left(\langle C^{-1}\bar m, u\rangle -
    \tfrac{1}{2}\langle C^{-1}\bar m, \bar m\rangle\right),
    $$

    so the target has the density $\propto \exp(-\tilde\Phi(u))$ relative to $\mathcal N(0, C)$,
    with the tilted potential $\tilde\Phi(u) = \Phi(u) - \langle C^{-1}\bar m, u\rangle$ (up to a
    constant) and the gradient $\nabla\Phi(u) - C^{-1}\bar m$. An algorithm that states the
    potential relative to a centred reference measure, such as
    [`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm], then samples the same
    target for any prior mean, with the state still the angle $u$: its score
    $\nabla\tilde\Phi(u) + C^{-1}u = \nabla\Phi(u) + C^{-1}(u - \bar m)$ is the gradient of the
    negative log posterior. For a zero mean, $\tilde\Phi = \Phi$.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self, target: DifferentiableTargetMeasure, prior: interfaces.GaussianPrior
    ) -> None:
        """Wrap `target`, the potential relative to `prior`, with the mean of `prior` moved into
        the potential."""
        self._target = target
        self._precision_weighted_mean = prior.apply_precision_operator(
            np.asarray(prior.mean_vector, dtype=np.float64)
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_potential(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        return self._target.evaluate_potential(state) - float(
            np.dot(self._precision_weighted_mean, state)
        )

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._target.evaluate_gradient(state) - self._precision_weighted_mean


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


# ==================================================================================================
class CenteredGaussianMeasure(GaussianMeasure):
    r"""A Gaussian $\mathcal N(\bar m, C)$ with its mean moved to zero: $\mathcal N(0, C)$.

    The covariance and precision actions are those of the wrapped measure, which do not depend on
    the mean. It is the reference measure of an algorithm that states the potential relative to a
    centred measure, together with
    [`MeanTiltedTargetMeasure`][bayes_cep.mcmc.measures.MeanTiltedTargetMeasure].
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, measure: GaussianMeasure) -> None:
        """Wrap `measure`; only its mean is replaced."""
        self._measure = measure

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def mean(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.zeros_like(self._measure.mean)

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def random_vector_size(self) -> int:
        return self._measure.random_vector_size

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._measure.apply_covariance_factorization(random_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._measure.apply_covariance_operator(vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._measure.apply_precision_operator(vector)


# ==================================================================================================
class LowRankGaussianMeasure(GaussianMeasure):
    r"""A low-rank Gaussian $\mathcal N(\bar m, \Gamma)$ as a `GaussianMeasure`, by delegation.

    `ls_bayesian.lowrank` and `ls_bayesian.mcmc` are independent subpackages, so a
    [`LowRankGaussian`][ls_bayesian.lowrank.gaussian.LowRankGaussian] does not implement the measure
    interface itself, which the MCMC model checks at runtime. For a Laplace approximation, $\bar m$
    is the MAP estimate and $\Gamma$ the low-rank update of the prior covariance; the measure is the
    `approximation` of the model, i.e. the proposal of preconditioned pCN or the preconditioner of
    preconditioned MALA. `evaluate_cost` is delegated, since the low-rank form needs only one
    application of the prior precision.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, gaussian: LowRankGaussian) -> None:
        """Wrap `gaussian`, whose reference covariance must be the covariance of the prior that
        serves as reference measure of the model."""
        self._gaussian = gaussian

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def mean(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._gaussian.mean

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def random_vector_size(self) -> int:
        return self._gaussian.random_vector_size

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._gaussian.apply_covariance_factorization(random_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._gaussian.apply_covariance_operator(vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._gaussian.apply_precision_operator(vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        return self._gaussian.evaluate_cost(state)

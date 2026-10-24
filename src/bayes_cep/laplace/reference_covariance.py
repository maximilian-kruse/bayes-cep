r"""Adapter of the fiber-angle prior to the reference covariance of `ls_bayesian.lowrank`.

Classes:
    PriorReferenceCovariance: The prior covariance $\Gamma_0$ as a `ReferenceCovariance`.
"""

from typing import override

import numpy as np
from ls_bayesian.lowrank.interfaces import ReferenceCovariance
from ls_bayesian.posterior.interfaces import GaussianPrior


# ==================================================================================================
class PriorReferenceCovariance(ReferenceCovariance):
    r"""The covariance $\Gamma_0$ and precision $\Gamma_0^{-1}$ of a prior, without its mean.

    `ls_bayesian.lowrank` and `ls_bayesian.posterior` were developed independently, so the prior
    does not implement `ReferenceCovariance` itself. This adapter only delegates; the mean of a
    Laplace approximation is the MAP estimate, which is passed to `LowRankGaussian` separately.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, prior: GaussianPrior) -> None:
        """Wrap `prior`.

        Args:
            prior (GaussianPrior): The prior whose covariance, factorization and precision are
                used.
        """
        self._prior = prior

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

r"""Adapts `ls_bayesian`'s SPDE prior to a fiber-angle `GaussianPrior`.

Classes:
    PriorSettings: Bilaplacian SPDE prior parameters for the fiber-angle field.
    FiberAnglePrior: `GaussianPrior` adapter delegating to an `ls_bayesian` `SPDEPrior`.

Functions:
    build_fiber_angle_prior: Build a `FiberAnglePrior` from settings and a dolfinx mesh.
"""

from dataclasses import dataclass
from typing import override

import dolfinx as dlx
import numpy as np
from ls_bayesian.posterior import interfaces
from ls_bayesian.spde_prior import builder, spde_prior, strategies


# ==================================================================================================
@dataclass(frozen=True)
class PriorSettings:
    r"""Bilaplacian SPDE prior parameters for the fiber-angle field.

    Attributes:
        mean_vector (np.ndarray): Prior mean angle field, given on mesh vertices.
        kappa (float): SPDE parameter $\kappa > 0$, controlling correlation length.
        tau (float): SPDE parameter $\tau > 0$, controlling marginal variance.
        seed (int): Random seed for prior sampling. Defaults to `0`.
    """

    mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    kappa: float
    tau: float
    seed: int = 0


# ==================================================================================================
class FiberAnglePrior(interfaces.GaussianPrior):
    """`GaussianPrior` adapter delegating to an `ls_bayesian` `SPDEPrior`.

    `SPDEPrior` already matches the `GaussianPrior` interface structurally, but per `ls_bayesian`'s
    own architecture is not coupled to that interface directly (subpackages there don't import each
    other); this adapter makes the connection explicit for use in this package, by delegation.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, prior: spde_prior.SPDEPrior) -> None:
        """Wrap an already-built `SPDEPrior`."""
        self._prior = prior

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def random_vector_size(self) -> int:
        return self._prior.random_vector_size

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        return self._prior.evaluate_cost(parameter_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.evaluate_gradient(parameter_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def evaluate_hessian_vector_product(
        self, direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.evaluate_hessian_vector_product(direction_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def generate_sample(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.generate_sample()

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_operator(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.apply_covariance_operator(parameter_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.apply_covariance_factorization(random_vector)

    # ----------------------------------------------------------------------------------------------
    @override
    def apply_precision_operator(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._prior.apply_precision_operator(parameter_vector)


# ==================================================================================================
def build_fiber_angle_prior(dlx_mesh: dlx.mesh.Mesh, settings: PriorSettings) -> FiberAnglePrior:
    """Build a `FiberAnglePrior` (Bilaplacian SPDE) on the given dolfinx mesh.

    Args:
        dlx_mesh (dlx.mesh.Mesh): Dolfinx mesh to build the prior on, e.g. via
            `bayes_cep.mesh.io.create_dolfinx_mesh`.
        settings (PriorSettings): Prior hyperparameters.

    Returns:
        FiberAnglePrior: The assembled fiber-angle prior.
    """
    prior_settings = builder.SPDEPriorSettings(
        mesh=dlx_mesh,
        mean_vector=settings.mean_vector,
        kappa=settings.kappa,
        tau=settings.tau,
        seed=settings.seed,
    )
    prior_builder = builder.SPDEPriorBuilder(
        prior_settings, strategies.BilaplacianComponentStrategy()
    )
    return FiberAnglePrior(prior_builder.build())

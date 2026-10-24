r"""Builds the low-rank Laplace approximation of the fiber-angle posterior.

The posterior is approximated by $\mathcal{N}(m_\mathrm{MAP}, \Gamma)$ with the precision
$\Gamma^{-1} = \Gamma_0^{-1} + H$, where $\Gamma_0$ is the prior covariance and $H$ the
Gauss-Newton Hessian of the data misfit at the MAP estimate
([`GaussNewtonMisfitHessian`][bayes_cep.laplace.hessian.GaussNewtonMisfitHessian]). The dominant
eigenpairs of $Hv = \lambda \Gamma_0^{-1} v$ give the low-rank form $\Gamma = \Gamma_0 - V D V^T$
([`LowRankGaussian`][ls_bayesian.lowrank.gaussian.LowRankGaussian]).

Classes:
    LaplaceSettings: Settings for `LaplaceBuilder`.
    LaplaceApproximation: The Gaussian approximation and the eigenpairs it is built from.
    LaplaceBuilder: Builds a `LaplaceApproximation` at a MAP estimate.
"""

from dataclasses import dataclass

import numpy as np
from ls_bayesian.lowrank.eigensolvers import (
    GeneralizedEigenpairs,
    LanczosSolver,
    RandomizedDoublePassSolver,
    compute_generalized_eigenpairs,
)
from ls_bayesian.lowrank.gaussian import LowRankGaussian

from bayes_cep.laplace.hessian import GaussNewtonMisfitHessian
from bayes_cep.laplace.reference_covariance import PriorReferenceCovariance
from bayes_cep.posterior.builder import PosteriorComponents


# ==================================================================================================
@dataclass(frozen=True)
class LaplaceSettings:
    """Settings for `LaplaceBuilder`.

    There are no defaults: the reference values are in `single_runs.reference`.

    Attributes:
        eigensolver (RandomizedDoublePassSolver | LanczosSolver): Solver of the generalized
            eigenproblem of the misfit Hessian and the prior precision. The double-pass solver
            symmetrizes its projected matrix and so tolerates the single precision of the forward
            model; the Lanczos solver needs a symmetric operator up to its tolerance.
        num_eigenvalues (int): Number of dominant eigenpairs to compute.
        minimum_eigenvalue (float): Eigenpairs with an eigenvalue at most this large are dropped.
        sample_seed (int): Seed of the sample generator of the approximation.
    """

    eigensolver: RandomizedDoublePassSolver | LanczosSolver
    num_eigenvalues: int
    minimum_eigenvalue: float
    sample_seed: int

    def __post_init__(self) -> None:
        if self.num_eigenvalues < 1:
            raise ValueError(f"num_eigenvalues must be at least 1, got {self.num_eigenvalues}.")
        if self.minimum_eigenvalue < 0.0:
            raise ValueError(
                f"minimum_eigenvalue must be non-negative, got {self.minimum_eigenvalue}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class LaplaceApproximation:
    """The Gaussian approximation and the eigenpairs it is built from.

    Attributes:
        gaussian (LowRankGaussian): The approximation, centered at the MAP estimate.
        eigenpairs (GeneralizedEigenpairs): The retained eigenpairs of the misfit Hessian relative
            to the prior precision, in descending order of the eigenvalues.
    """

    gaussian: LowRankGaussian
    eigenpairs: GeneralizedEigenpairs


# ==================================================================================================
class LaplaceBuilder:
    """Builds the low-rank Laplace approximation at a MAP estimate.

    Methods:
        build: Evaluate the Gauss-Newton Hessian at the estimate and compute its eigenpairs.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: LaplaceSettings, components: PosteriorComponents) -> None:
        """Store the settings and the components of the posterior.

        Args:
            settings (LaplaceSettings): Eigensolver and rank.
            components (PosteriorComponents): Likelihood, forward map and prior of the posterior,
                e.g. from `PosteriorBuilder.build_components`.
        """
        self._settings = settings
        self._components = components

    # ----------------------------------------------------------------------------------------------
    def build(
        self, map_estimate: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> LaplaceApproximation:
        """Evaluate the Gauss-Newton Hessian at `map_estimate` and compute its eigenpairs.

        Args:
            map_estimate (np.ndarray): MAP estimate [rad] per vertex, shape `(num_vertices,)`.

        Returns:
            LaplaceApproximation: The approximation centered at `map_estimate`.

        Raises:
            ValueError: If `map_estimate` does not have the shape of the prior mean, or if no
                eigenvalue exceeds `minimum_eigenvalue` (the data informs no direction).
        """
        settings = self._settings
        prior = self._components.prior
        if map_estimate.shape != prior.mean_vector.shape:
            raise ValueError(
                f"The MAP estimate has shape {map_estimate.shape}, but the prior mean has shape "
                f"{prior.mean_vector.shape}."
            )
        hessian = GaussNewtonMisfitHessian(
            self._components.forward_map, self._components.likelihood, map_estimate
        )
        eigenpairs = compute_generalized_eigenpairs(
            hessian.apply,
            prior.apply_precision_operator,
            prior.apply_covariance_operator,
            hessian.size,
            settings.num_eigenvalues,
            settings.eigensolver,
            settings.minimum_eigenvalue,
        )
        if eigenpairs.eigenvalues.size == 0:
            raise ValueError(
                f"No eigenvalue above minimum_eigenvalue = {settings.minimum_eigenvalue}: the "
                "data does not inform any direction, so the approximation is the prior."
            )
        gaussian = LowRankGaussian(
            PriorReferenceCovariance(prior),
            eigenpairs,
            mean=map_estimate,
            seed=settings.sample_seed,
        )
        return LaplaceApproximation(gaussian=gaussian, eigenpairs=eigenpairs)

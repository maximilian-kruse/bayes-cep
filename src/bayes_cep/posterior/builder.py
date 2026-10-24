r"""Builds an `ls_bayesian` posterior for the fiber-orientation forward model.

Classes:
    PosteriorSettings: Settings for `PosteriorBuilder`.
    PosteriorComponents: Likelihood, forward map and prior of one dataset.
    PosteriorBuilder: Builds the prior, forward map, and likelihood from a given mesh and dataset,
        then composes an `ls_bayesian` `LogPosterior`.
"""

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.posterior import likelihood as ls_likelihood
from ls_bayesian.posterior import posterior

from bayes_cep.mesh.interpolation import InterpolationStrategy
from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap, EikonalSolverSettings
from bayes_cep.posterior.likelihood import LikelihoodSettings, build_activation_time_likelihood
from bayes_cep.posterior.prior import FiberAnglePrior, PriorParameters, build_fiber_angle_prior


# ==================================================================================================
@dataclass(frozen=True)
class PosteriorSettings:
    """Settings for `PosteriorBuilder`.

    Attributes:
        mesh (pv.UnstructuredGrid): Triangular surface mesh, e.g. loaded via
            `bayes_cep.mesh.io.load_pyvista_mesh`.
        basis_vectors (np.ndarray): Local tangent-plane basis vectors per simplex, shape
            `(num_simplices, 3, 2)`.
        prior_mean_vector (np.ndarray): Prior mean angle field, given on mesh vertices.
        prior_parameters (PriorParameters): Bilaplacian SPDE parameters of the prior.
        eikonal_settings (EikonalSolverSettings): Eikonal forward solver settings.
        likelihood_settings (LikelihoodSettings): Observed activation-time data settings.
    """

    mesh: pv.UnstructuredGrid
    basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    prior_mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    prior_parameters: PriorParameters
    eikonal_settings: EikonalSolverSettings
    likelihood_settings: LikelihoodSettings


# ==================================================================================================
@dataclass(frozen=True)
class PosteriorComponents:
    """The three components of a posterior for one dataset.

    Attributes:
        likelihood (GaussianLogLikelihood): Gaussian likelihood of the observed activation times.
        forward_map (EikonalParameterToSolutionMap): Eikonal forward map.
        prior (FiberAnglePrior): Fiber-angle prior.
    """

    likelihood: ls_likelihood.GaussianLogLikelihood
    forward_map: EikonalParameterToSolutionMap
    prior: FiberAnglePrior

    # ----------------------------------------------------------------------------------------------
    def to_log_posterior(self, logger: BaseLogger | None = None) -> posterior.LogPosterior:
        """Compose the components into a `LogPosterior`; they are shared, not copied.

        Args:
            logger (BaseLogger | None, optional): Logger for the evaluation diagnostics of the
                `LogPosterior`. Nothing is logged if `None`. Defaults to `None`.

        Returns:
            posterior.LogPosterior: The negative log-posterior of the components.
        """
        return posterior.LogPosterior(self.likelihood, self.forward_map, self.prior, logger=logger)


# ==================================================================================================
class PosteriorBuilder:
    """Builds the prior, forward map, and likelihood from a given mesh and dataset, then composes
    an `ls_bayesian` `LogPosterior`.

    All three components, including the likelihood, are built in `build`, not `__init__`: a
    `LogPosterior` is tied to one fixed dataset, so there is no cheaper alternative to rebuilding
    the whole set of components for a different dataset (e.g. a different synthetic replicate in a
    simulation study) — an accepted trade-off for the simplicity of a no-argument `build`. The
    builder keeps no state: the prior is read from the result, `LogPosterior.prior`.

    Methods:
        build: Build the prior, forward map, and likelihood, then compose a `LogPosterior`.
        build_components: Build the same three components without composing them.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: PosteriorSettings,
        interpolation_strategy: InterpolationStrategy,
        prior_logger: BaseLogger | None = None,
        posterior_logger: BaseLogger | None = None,
    ) -> None:
        """Store the settings, interpolation strategy, and loggers for `build`.

        Args:
            settings (PosteriorSettings): Mesh, prior, eikonal solver, and likelihood settings.
            interpolation_strategy (InterpolationStrategy): Strategy for interpolating the
                vertex-based parameter onto simplices for the forward map.
            prior_logger (BaseLogger | None, optional): Logger for the prior's evaluation
                diagnostics, passed to `build_fiber_angle_prior`. Nothing is logged if `None`.
                Defaults to `None`.
            posterior_logger (BaseLogger | None, optional): Logger for the composed
                `LogPosterior`'s evaluation diagnostics. Nothing is logged if `None`. Defaults to
                `None`.
        """
        self._settings = settings
        self._interpolation_strategy = interpolation_strategy
        self._prior_logger = prior_logger
        self._posterior_logger = posterior_logger

    # ----------------------------------------------------------------------------------------------
    def build(self) -> posterior.LogPosterior:
        """Build the prior, forward map, and likelihood, then compose a `LogPosterior`.

        Returns:
            posterior.LogPosterior: The assembled negative log-posterior.
        """
        return self.build_components().to_log_posterior(self._posterior_logger)

    # ----------------------------------------------------------------------------------------------
    def build_components(self) -> PosteriorComponents:
        """Build the prior, forward map, and likelihood without composing them.

        For consumers that need the parts a `LogPosterior` keeps private, e.g. the Gauss-Newton
        Hessian of the data misfit, which composes the forward map and the likelihood itself.

        Returns:
            PosteriorComponents: The three components of one dataset.
        """
        settings = self._settings
        prior = build_fiber_angle_prior(
            create_dolfinx_mesh(settings.mesh),
            settings.prior_mean_vector,
            settings.prior_parameters,
            logger=self._prior_logger,
        )
        forward_map = EikonalParameterToSolutionMap(
            settings.mesh,
            settings.basis_vectors,
            settings.eikonal_settings,
            self._interpolation_strategy,
        )
        likelihood = build_activation_time_likelihood(settings.likelihood_settings)
        return PosteriorComponents(likelihood=likelihood, forward_map=forward_map, prior=prior)

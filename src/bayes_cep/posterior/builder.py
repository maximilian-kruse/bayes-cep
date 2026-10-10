r"""Builds an `ls_bayesian` posterior for the fiber-orientation forward model.

Classes:
    PosteriorSettings: Settings for `PosteriorBuilder`.
    PosteriorBuilder: Builds the prior, forward map, and likelihood from a given mesh and dataset,
        then composes an `ls_bayesian` `LogPosterior`.
"""

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.posterior import posterior

from bayes_cep.mesh.interpolation import InterpolationStrategy
from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap, EikonalSolverSettings
from bayes_cep.posterior.likelihood import LikelihoodSettings, build_activation_time_likelihood
from bayes_cep.posterior.prior import PriorSettings, build_fiber_angle_prior


# ==================================================================================================
@dataclass(frozen=True)
class PosteriorSettings:
    """Settings for `PosteriorBuilder`.

    Attributes:
        mesh (pv.UnstructuredGrid): Triangular surface mesh, e.g. loaded via
            `bayes_cep.mesh.io.load_pyvista_mesh`.
        basis_vectors (np.ndarray): Local tangent-plane basis vectors per simplex, shape
            `(num_simplices, 3, 2)`.
        prior_settings (PriorSettings): Fiber-angle SPDE prior settings.
        eikonal_settings (EikonalSolverSettings): Eikonal forward solver settings.
        likelihood_settings (LikelihoodSettings): Observed activation-time data settings.
    """

    mesh: pv.UnstructuredGrid
    basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    prior_settings: PriorSettings
    eikonal_settings: EikonalSolverSettings
    likelihood_settings: LikelihoodSettings


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
        settings = self._settings
        prior = build_fiber_angle_prior(
            create_dolfinx_mesh(settings.mesh), settings.prior_settings, logger=self._prior_logger
        )
        forward_map = EikonalParameterToSolutionMap(
            settings.mesh,
            settings.basis_vectors,
            settings.eikonal_settings,
            self._interpolation_strategy,
        )
        likelihood = build_activation_time_likelihood(settings.likelihood_settings)
        return posterior.LogPosterior(likelihood, forward_map, prior, logger=self._posterior_logger)

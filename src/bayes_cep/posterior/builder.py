r"""Builds an `ls_bayesian` posterior for the fiber-orientation forward model.

Classes:
    PosteriorSettings: Settings for `PosteriorBuilder`.
    PosteriorBuilder: Builds the prior, forward map, and likelihood from a given mesh and dataset,
        then composes an `ls_bayesian` `LogPosterior`.
"""

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from ls_bayesian.posterior import interfaces, posterior

from bayes_cep.mesh.interpolation import InterpolationStrategy
from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.posterior.eikonal_map import EikonalParameterToSolutionMap, EikonalSolverSettings
from bayes_cep.posterior.likelihood import LikelihoodSettings, build_activation_time_likelihood
from bayes_cep.posterior.prior import FiberAnglePrior, PriorSettings, build_fiber_angle_prior


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
    simulation study) — an accepted trade-off for the simplicity of a no-argument `build`.

    Attributes:
        pv_mesh (pv.UnstructuredGrid): The triangular surface mesh. Set by `build`.
        prior (FiberAnglePrior): The fiber-angle prior. Set by `build`.
        forward_map (EikonalParameterToSolutionMap): The eikonal activation-time forward map. Set
            by `build`.
        likelihood (interfaces.Likelihood): The activation-time observation likelihood. Set by
            `build`.

    Methods:
        build: Build the prior, forward map, and likelihood, then compose a `LogPosterior`.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: PosteriorSettings,
        interpolation_strategy: InterpolationStrategy | None = None,
    ) -> None:
        """Store the settings and interpolation strategy for `build`.

        Args:
            settings (PosteriorSettings): Mesh, prior, eikonal solver, and likelihood settings.
            interpolation_strategy (InterpolationStrategy | None, optional): Strategy for
                interpolating the vertex-based parameter onto simplices for the forward map.
                Defaults to `LinearInterpolationStrategy` if `None`.
        """
        self._settings = settings
        self._interpolation_strategy = interpolation_strategy

    # ----------------------------------------------------------------------------------------------
    def build(self) -> posterior.LogPosterior:
        """Build the prior, forward map, and likelihood, then compose a `LogPosterior`.

        Returns:
            posterior.LogPosterior: The assembled negative log-posterior.
        """
        self.pv_mesh: pv.UnstructuredGrid = self._settings.mesh
        dlx_mesh = create_dolfinx_mesh(self.pv_mesh)
        self.prior: FiberAnglePrior = build_fiber_angle_prior(
            dlx_mesh, self._settings.prior_settings
        )
        self.forward_map: EikonalParameterToSolutionMap = EikonalParameterToSolutionMap(
            self.pv_mesh,
            self._settings.basis_vectors,
            self._settings.eikonal_settings,
            self._interpolation_strategy,
        )
        self.likelihood: interfaces.Likelihood = build_activation_time_likelihood(
            self._settings.likelihood_settings
        )
        return posterior.LogPosterior(self.likelihood, self.forward_map, self.prior)

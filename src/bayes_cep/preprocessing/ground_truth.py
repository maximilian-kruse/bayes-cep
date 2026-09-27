r"""Ground-truth fiber-angle field: a synthetic (SPDE-sampled) source, and the strategy choosing
between it and the real-data source in `angle_field.py`.

A simulation study picks exactly one `GroundTruthStrategy` per run — never both — via
`PreprocessingSettings.ground_truth` in `scripts/preprocess_data.py`. The prior mean is not part of
this strategy: it's computed uniformly from whichever ground truth comes out, via
[`prior_mean.build_constant_prior_mean`][bayes_cep.preprocessing.prior_mean.build_constant_prior_mean].

Classes:
    SyntheticGroundTruthSettings: SPDE prior parameters for the synthetic ground truth.
    GroundTruthStrategy: ABC interface for producing a ground-truth angle field.
    RealDataGroundTruthStrategy: Derive it from one patient's raw fiber data.
    SyntheticGroundTruthStrategy: Sample it from the SPDE prior itself, decoupled from any patient
        data.

Functions:
    build_synthetic_ground_truth: Sample a synthetic ground truth from the Bilaplacian SPDE prior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

import numpy as np
import pyvista as pv

from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.posterior.prior import PriorSettings, build_fiber_angle_prior
from bayes_cep.preprocessing.angle_field import build_angle_field_from_fiber_field
from bayes_cep.preprocessing.axial_statistics import shift_angles_to_minimize_axial_variance


# ==================================================================================================
@dataclass(frozen=True)
class SyntheticGroundTruthSettings:
    r"""SPDE prior parameters for the synthetic ground truth.

    Attributes:
        kappa (float): SPDE parameter $\kappa > 0$, controlling correlation length. Defaults to the
            legacy reference value `5.0`.
        tau (float): SPDE parameter $\tau > 0$, controlling marginal variance. Defaults to the
            legacy reference value `0.01`.
        seed (int): Random seed for prior sampling. Defaults to `0`.
    """

    kappa: float = 5.0
    tau: float = 0.01
    seed: int = 0


# ==================================================================================================
def build_synthetic_ground_truth(
    mesh: pv.UnstructuredGrid, settings: SyntheticGroundTruthSettings
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Sample a synthetic ground-truth angle field from the Bilaplacian SPDE prior.

    Draws one sample from a zero-mean Bilaplacian SPDE prior (`settings.kappa`, `settings.tau`) and
    re-branches it onto a single $\pi$-periodic branch.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh to sample the prior on.
        settings (SyntheticGroundTruthSettings): SPDE prior parameters and random seed.

    Returns:
        np.ndarray: Synthetic ground-truth angle field, shape `(num_vertices,)`.
    """
    dlx_mesh = create_dolfinx_mesh(mesh)
    num_vertices = mesh.points.shape[0]
    prior = build_fiber_angle_prior(
        dlx_mesh,
        PriorSettings(
            mean_vector=np.zeros(num_vertices),
            kappa=settings.kappa,
            tau=settings.tau,
            seed=settings.seed,
        ),
    )
    return shift_angles_to_minimize_axial_variance(prior.generate_sample()).flatten()


# ==================================================================================================
class GroundTruthStrategy(ABC):
    """ABC interface for producing a ground-truth fiber-angle field.

    Methods:
        build: Produce the ground-truth angle field.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def build(
        self,
        mesh: pv.UnstructuredGrid,
        fiber_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Produce the ground-truth angle field, shape `(num_vertices,)`.

        Args:
            mesh (pv.UnstructuredGrid): Triangular surface mesh.
            fiber_field (np.ndarray): Per-simplex 3D fiber vectors, shape `(num_simplices, 3)`.
            basis_vectors (np.ndarray): Local tangent-plane basis vectors per simplex, shape
                `(num_simplices, 3, 2)`.

        Returns:
            np.ndarray: Ground-truth angle field.
        """


# ==================================================================================================
@dataclass(frozen=True)
class RealDataGroundTruthStrategy(GroundTruthStrategy):
    """Derive the ground truth from one patient's raw fiber data."""

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        mesh: pv.UnstructuredGrid,
        fiber_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Build the real-data ground truth."""
        return build_angle_field_from_fiber_field(mesh, fiber_field, basis_vectors)


# ==================================================================================================
@dataclass(frozen=True)
class SyntheticGroundTruthStrategy(GroundTruthStrategy):
    """Sample the ground truth from the SPDE prior itself.

    Fully decoupled from any real patient data: ignores `fiber_field` and `basis_vectors`.

    Attributes:
        settings (SyntheticGroundTruthSettings): SPDE prior parameters and random seed.
    """

    settings: SyntheticGroundTruthSettings = field(default_factory=SyntheticGroundTruthSettings)

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        mesh: pv.UnstructuredGrid,
        fiber_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Build the synthetic ground truth."""
        return build_synthetic_ground_truth(mesh, self.settings)

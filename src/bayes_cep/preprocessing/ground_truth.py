r"""Ground-truth fiber-angle field: a synthetic (SPDE-sampled) source, and the strategy choosing
between it and the real-data source in `angle_field.py`.

A simulation study picks exactly one `GroundTruthStrategy` per run — never both — via
`PreprocessingRunConfig.ground_truth` (in `single_runs/preprocessing.py`). The prior mean is not
part of this strategy: it is chosen from the observations alone (see `prior_mean.py`), as the mean
of the ground truth is not known in a realistic scenario.

Classes:
    GroundTruthStrategy: ABC interface for producing a ground-truth angle field.
    RealDataGroundTruthStrategy: Derive it from one patient's raw fiber data.
    SmoothedFieldGroundTruthStrategy: Read a stored smoothed per-vertex angle field of a patient
        (see `scripts/smooth_patient_fibers.py`).
    SyntheticGroundTruthStrategy: Sample it from the SPDE prior itself, decoupled from any patient
        data.

Functions:
    build_synthetic_ground_truth: Sample a synthetic ground truth from the Bilaplacian SPDE prior.

Note:
    The concrete strategies and their settings are dataclasses on purpose: `tyro` only exposes a
    class as a selectable subcommand (e.g.
    `--config.ground-truth:synthetic-ground-truth-strategy`) in the CLI of `bcep-single` if it
    is a dataclass; otherwise it freezes the argument to the default instance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pyvista as pv

from bayes_cep.mesh.io import create_dolfinx_mesh
from bayes_cep.posterior.prior import PriorParameters, build_fiber_angle_prior
from bayes_cep.preprocessing.angle_field import build_angle_field_from_fiber_field
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.statistics.axial_statistics import shift_angles_to_minimize_axial_variance


# ==================================================================================================
def build_synthetic_ground_truth(
    mesh: pv.UnstructuredGrid, prior_parameters: PriorParameters
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Sample a synthetic ground-truth angle field from the Bilaplacian SPDE prior.

    Draws one sample from a zero-mean Bilaplacian SPDE prior (`prior_parameters`). The sample is
    kept as drawn: shifting it by multiples of $\pi$ would add jumps that the real-valued prior
    does not allow. The forward map is $\pi$-periodic, and errors are measured axially.

    Args:
        mesh (pv.UnstructuredGrid): Triangular surface mesh to sample the prior on.
        prior_parameters (PriorParameters): SPDE prior parameters and random seed.

    Returns:
        np.ndarray: Synthetic ground-truth angle field, shape `(num_vertices,)`.
    """
    dlx_mesh = create_dolfinx_mesh(mesh)
    num_vertices = mesh.points.shape[0]
    prior = build_fiber_angle_prior(
        dlx_mesh,
        np.zeros(num_vertices),
        prior_parameters,
    )
    return prior.generate_sample()


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
class SmoothedFieldGroundTruthStrategy(GroundTruthStrategy):
    r"""Read the smoothed per-vertex angle field of a patient from a `.npy` file.

    The stored angles lie in $(-\pi/2, \pi/2]$ and jump by $\pi$ where the fiber crosses the
    window; the prior is real-valued, so they are shifted by multiples of $\pi$ into one window
    around the axial mean (`shift_angles_to_minimize_axial_variance`), as for the raw patient data.
    Ignores `fiber_field` and `basis_vectors`.

    Attributes:
        path (Path): The file, one angle [rad] per vertex of the mesh; a relative path is relative
            to the repository.
    """

    path: Path

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        mesh: pv.UnstructuredGrid,
        fiber_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Build the ground truth from the stored smoothed field."""
        resolved_path = resolve_repository_path(self.path)
        angles = np.load(resolved_path)
        if angles.shape != (mesh.n_points,):
            raise ValueError(
                f"The angle field in {resolved_path} has shape {angles.shape}, but the mesh has "
                f"{mesh.n_points} vertices."
            )
        return shift_angles_to_minimize_axial_variance(angles)


# ==================================================================================================
@dataclass(frozen=True)
class SyntheticGroundTruthStrategy(GroundTruthStrategy):
    """Sample the ground truth from the SPDE prior itself.

    Fully decoupled from any real patient data: ignores `fiber_field` and `basis_vectors`.

    Attributes:
        prior (PriorParameters): SPDE prior parameters and random seed; there are no defaults.
    """

    prior: PriorParameters

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        mesh: pv.UnstructuredGrid,
        fiber_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        basis_vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Build the synthetic ground truth."""
        return build_synthetic_ground_truth(mesh, self.prior)

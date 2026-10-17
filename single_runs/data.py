"""The data the runs read and write: the raw data directory and the preprocessed data.

The file names of the raw data and of the preprocessed data are defined here and nowhere else, so
that the runs, their `input_files` and their reports cannot disagree about them.

Classes:
    RawDataDirectory: The directory with the raw patient data (mesh, basis vectors, fiber field).
    PreprocessedData: The data the inference runs work on, as written by the preprocessing run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pyvista as pv

from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path

MESH_FILE_NAME = "mesh.vtu"
BASIS_VECTORS_FILE_NAME = "basis_vecs.npy"
FIBER_FIELD_FILE_NAME = "fiber_field.npy"


# ==================================================================================================
@dataclass(frozen=True)
class RawDataDirectory:
    """The directory with the raw patient data.

    Attributes:
        path (Path): The directory, already resolved against the repository root.
    """

    path: Path

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def from_configured_path(cls, raw_dir: Path) -> Self:
        """The directory a configuration names; a relative path is relative to the repository."""
        return cls(resolve_repository_path(raw_dir))

    # ----------------------------------------------------------------------------------------------
    @property
    def mesh_path(self) -> Path:
        """The triangular surface mesh (`.vtu`)."""
        return self.path / MESH_FILE_NAME

    # ----------------------------------------------------------------------------------------------
    @property
    def basis_vectors_path(self) -> Path:
        """The local tangent-plane basis vectors per simplex, shape `(num_simplices, 3, 2)`."""
        return self.path / BASIS_VECTORS_FILE_NAME

    # ----------------------------------------------------------------------------------------------
    @property
    def fiber_field_path(self) -> Path:
        """The per-simplex 3D fiber vectors, shape `(num_simplices, 3)`."""
        return self.path / FIBER_FIELD_FILE_NAME

    # ----------------------------------------------------------------------------------------------
    def load_mesh(self) -> pv.UnstructuredGrid:
        """Load and validate the triangular surface mesh."""
        return load_pyvista_mesh(self.mesh_path)

    # ----------------------------------------------------------------------------------------------
    def load_basis_vectors(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """Load the basis vectors, shape `(num_simplices, 3, 2)`."""
        return np.load(self.basis_vectors_path)

    # ----------------------------------------------------------------------------------------------
    def load_fiber_field(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Load the fiber vectors, shape `(num_simplices, 3)`."""
        return np.load(self.fiber_field_path)


# ==================================================================================================
@dataclass(frozen=True)
class PreprocessedData(ArrayRecord):
    """The data the inference runs work on, as written by the preprocessing run.

    Attributes:
        ground_truth (np.ndarray): Ground-truth fiber angle [rad] per vertex.
        prior_mean (np.ndarray): Prior mean [rad] per vertex: zero, as the mean of the ground truth
            is not known in a realistic scenario.
        observed_vertex_indices (np.ndarray): Indices of the observed vertices.
        observed_activation_times (np.ndarray): Noisy activation times at those vertices.
        noise_variance (float): Variance of the noise of the observations; the likelihood of the
            inference uses the same value, so that data and inference cannot disagree.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "ground_truth": "ground_truth_angle_field.npy",
        "prior_mean": "prior_mean_angle_field.npy",
        "observed_vertex_indices": "observed_vertex_indices.npy",
        "observed_activation_times": "observed_activation_times.npy",
        "noise_variance": "observation_noise_variance.npy",
    }

    ground_truth: np.ndarray
    prior_mean: np.ndarray
    observed_vertex_indices: np.ndarray
    observed_activation_times: np.ndarray
    noise_variance: float

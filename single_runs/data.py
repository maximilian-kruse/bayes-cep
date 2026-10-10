"""The data the runs read and write: the raw data directory, and arrays stored as `.npy` files.

The file names of the raw data and of the preprocessed data are defined here and nowhere else, so
that the runs, their `input_files` and their reports cannot disagree about them.

Classes:
    RawDataDirectory: The directory with the raw patient data (mesh, basis vectors, fiber field).
    ArrayRecord: Base class of a set of arrays that is stored as one `.npy` file per field.
    PreprocessedData: The data the inference runs work on, as written by the preprocessing run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pyvista as pv

from bayes_cep.mesh.io import load_pyvista_mesh
from bayes_cep.run.directories import resolve_repository_path

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
class ArrayRecord:
    """A set of arrays (and scalars) stored as one `.npy` file per field.

    A subclass is a frozen dataclass that names the file of each of its fields in `FILE_NAMES`
    (field name to file name). Saving, loading and the list of files follow from that mapping, so
    there is no second place where the order or the names of the files could differ. A scalar
    field is stored as a 0-d array and read back as a Python scalar.

    Attributes:
        FILE_NAMES (Mapping[str, str]): File name of each field.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]]

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def file_paths(cls, directory: Path) -> list[Path]:
        """The files of the record in `directory`."""
        return [directory / file_name for file_name in cls.FILE_NAMES.values()]

    # ----------------------------------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """Write the fields into `directory`, which is created if missing."""
        directory.mkdir(parents=True, exist_ok=True)
        for field_name, file_name in self.FILE_NAMES.items():
            np.save(directory / file_name, getattr(self, field_name))

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, directory: Path) -> Self:
        """Read the fields from `directory`."""
        values = {}
        for field_name, file_name in cls.FILE_NAMES.items():
            array = np.load(directory / file_name)
            values[field_name] = array.item() if array.ndim == 0 else array
        return cls(**values)


# ==================================================================================================
@dataclass(frozen=True)
class PreprocessedData(ArrayRecord):
    """The data the inference runs work on, as written by the preprocessing run.

    Attributes:
        ground_truth (np.ndarray): Ground-truth fiber angle [rad] per vertex.
        prior_mean (np.ndarray): Constant prior mean [rad] per vertex: the axial mean of the ground
            truth.
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

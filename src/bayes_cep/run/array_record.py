"""A set of arrays stored as one `.npy` file per field.

Generic helper for the result files of runs: a frozen dataclass names the file of each field, and
saving, loading and the list of files follow from that, so the names exist in one place only.

Classes:
    ArrayRecord: Base class of a set of arrays that is stored as one `.npy` file per field.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self

import numpy as np


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

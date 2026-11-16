"""Restrict the vertex statistics of a chain to a subset of the vertices.

Classes:
    ColumnSelectedSampleStore: Lazy view of a sample store with a subset of its columns.

Functions:
    scatter_to_vertices: Put per-selected-vertex values into a full-length array, `nan` elsewhere.
"""

import numpy as np
from ls_bayesian.mcmc.storage import SampleStore


# ==================================================================================================
class ColumnSelectedSampleStore:
    """Lazy view of a sample store that shows only some of its columns.

    Nothing is copied: each slice is read from the base store and the columns are picked on access.
    Satisfies the `SampleStore` protocol, so the streaming statistics work on it unchanged.

    Attributes:
        shape (tuple[int, ...]): `(num_samples, num_selected)`.
    """

    def __init__(self, base: SampleStore, column_indices: np.ndarray) -> None:
        """Wrap `base`.

        Args:
            base (SampleStore): Samples, shape `(num_samples, num_columns)`.
            column_indices (np.ndarray): Integer indices of the columns to show, in this order.

        Raises:
            ValueError: If an index is outside the columns of `base`.
        """
        indices = np.asarray(column_indices, dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= base.shape[1]):
            raise ValueError(
                f"Column indices range over [{indices.min()}, {indices.max()}], but the store has "
                f"{base.shape[1]} columns."
            )
        self._base = base
        self._indices = indices

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape `(num_samples, num_selected)`."""
        return (self._base.shape[0], self._indices.size)

    def __getitem__(self, key: slice, /) -> np.ndarray:
        """Return the selected columns of the samples in `key`."""
        return np.asarray(self._base[key])[:, self._indices]


# ==================================================================================================
def scatter_to_vertices(
    values: np.ndarray,
    vertex_indices: np.ndarray,
    num_vertices: int,
    fill_value: float | bool = np.nan,
) -> np.ndarray:
    """Put values of selected vertices into an array over all vertices.

    Args:
        values (np.ndarray): Values of the selected vertices, shape `(num_selected, ...)`; the
            first axis is the vertex axis.
        vertex_indices (np.ndarray): Indices of these vertices, shape `(num_selected,)`.
        num_vertices (int): Number of all vertices.
        fill_value (float | bool): Entry of the other vertices. Defaults to `nan`; use `False`
            for boolean values.

    Returns:
        np.ndarray: Array of shape `(num_vertices, ...)` with the dtype of `values` (float for
            `nan`).
    """
    full = np.full((num_vertices, *values.shape[1:]), fill_value, dtype=values.dtype)
    full[vertex_indices] = values
    return full

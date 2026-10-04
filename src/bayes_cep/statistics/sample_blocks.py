"""Block-wise reading of sample arrays, so large sample stores never have to fit in memory.

Classes:
    SampleStore: Protocol for sample arrays sliceable along their first axis.

Functions:
    iterate_sample_blocks: Yield blocks of samples after discarding a burn-in.
"""

from collections.abc import Iterator
from typing import Protocol

import numpy as np


# ==================================================================================================
class SampleStore(Protocol):
    """Sample array read in slices along its first axis, e.g. a `numpy` or `zarr` array."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape `(num_samples, num_components)`."""
        ...

    def __getitem__(self, key: slice, /) -> np.ndarray:
        """Return the samples in `key`."""
        ...


# --------------------------------------------------------------------------------------------------
def iterate_sample_blocks(
    samples: SampleStore, burn_in: int = 0, block_size: int = 100
) -> Iterator[np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    """Yield blocks of samples after discarding the first `burn_in` samples.

    Args:
        samples (SampleStore): Samples, shape `(num_samples, num_components)`.
        burn_in (int): Number of leading samples to discard. Defaults to `0`.
        block_size (int): Maximum number of samples per block. Defaults to `100`.

    Raises:
        ValueError: If `burn_in` is negative or leaves no samples, or `block_size` is not positive.

    Yields:
        np.ndarray: Consecutive blocks, shape `(<= block_size, num_components)`.
    """
    num_samples = samples.shape[0]
    if not 0 <= burn_in < num_samples:
        raise ValueError(f"burn_in must be in [0, {num_samples}), got {burn_in}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    for start in range(burn_in, num_samples, block_size):
        yield np.asarray(samples[start : min(start + block_size, num_samples)])

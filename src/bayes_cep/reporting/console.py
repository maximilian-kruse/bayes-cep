"""Structured console output for command-line scripts.

Functions:
    report_step: Context manager printing a numbered progress line and the step's wall-clock time.
    describe_array: One-line summary of an array's name, shape, and value range.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np


# ==================================================================================================
@contextmanager
def report_step(index: int, num_steps: int, description: str) -> Generator[None]:
    """Print a numbered progress line before a step and its wall-clock time after.

    Prints `[index/num_steps] description ...` on entry and `done in <t> s` on a normal exit. If the
    body raises, no completion line is printed and the exception propagates.

    Args:
        index (int): 1-based index of the step.
        num_steps (int): Total number of steps.
        description (str): What the step does.

    Yields:
        None: Control to the body of the step.
    """
    print(f"[{index}/{num_steps}] {description} ...", flush=True)
    start = time.perf_counter()
    yield
    print(f"      done in {time.perf_counter() - start:.2f} s", flush=True)


# ==================================================================================================
def describe_array(name: str, array: np.ndarray) -> str:
    """One-line summary of an array: name, shape, and value range.

    Args:
        name (str): Label for the array.
        array (np.ndarray): Non-empty numeric array.

    Returns:
        str: `"<name>: shape <shape>, min <min>, max <max>, mean <mean>"`.
    """
    return (
        f"{name}: shape {array.shape}, min {array.min():.4g}, max {array.max():.4g}, "
        f"mean {array.mean():.4g}"
    )

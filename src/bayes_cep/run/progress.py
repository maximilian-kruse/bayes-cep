"""Progress output of the runs: numbered steps with timings, and array summaries.

Classes:
    StepReporter: Numbers the steps of a run and logs each with its wall-clock time.

Functions:
    describe_array: One-line summary of an array's name, shape, and value range.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
from ls_bayesian.common.logging import BaseLogger


# ==================================================================================================
class StepReporter:
    """Numbers the steps of a run and logs each with its wall-clock time.

    `with reporter.step("Loading data"):` logs `[1] Loading data ...` on entry and
    `done in <t> s` on a normal exit; the next step is `[2]`, and so on. If the body raises, no
    completion line is logged and the exception propagates.
    """

    def __init__(self, logger: BaseLogger) -> None:
        """Report the steps of a run to `logger`."""
        self._logger = logger
        self._num_started_steps = 0

    # ----------------------------------------------------------------------------------------------
    @contextmanager
    def step(self, description: str) -> Generator[None]:
        """Report the next step of the run: its numbered description, then its duration.

        Args:
            description (str): What the step does.

        Yields:
            None: Control to the body of the step.
        """
        self._num_started_steps += 1
        self._logger.info(f"[{self._num_started_steps}] {description} ...")
        start = time.perf_counter()
        yield
        self._logger.info(f"      done in {time.perf_counter() - start:.2f} s")


# ==================================================================================================
def describe_array(name: str, array: np.ndarray) -> str:
    """One-line summary of an array: name, shape, value range and non-finite entries.

    Minimum, maximum and mean are taken over the finite entries only, so that a few `NaN`s do not
    hide the range of the rest; the counts of `NaN` and infinite entries are always reported.

    Args:
        name (str): Label for the array.
        array (np.ndarray): Numeric array.

    Returns:
        str: `"<name>: shape <shape>, min <min>, max <max>, mean <mean>, nan <count>, inf <count>"`;
            the statistics read `n/a` if there are no finite entries.
    """
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        statistics = "min n/a, max n/a, mean n/a"
    else:
        statistics = f"min {finite.min():.4g}, max {finite.max():.4g}, mean {finite.mean():.4g}"
    return (
        f"{name}: shape {array.shape}, {statistics}, "
        f"nan {np.isnan(array).sum()}, inf {np.isinf(array).sum()}"
    )

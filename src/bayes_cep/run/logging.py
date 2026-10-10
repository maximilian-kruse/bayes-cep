"""Progress output of a run: numbered steps with timings, written to the console and the run log.

Classes:
    StepReporter: Numbers the steps of a run and logs each with its wall-clock time.

Functions:
    run_logger: Context manager providing a prefix-free logger for one run, writing to the
        console and, optionally, a log file.
    describe_array: One-line summary of an array's name, shape, and value range.
"""

import sys
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from ls_bayesian.common.logging import BaseLogger, LoggerSettings


# ==================================================================================================
@contextmanager
def run_logger(logfile_path: Path | None, print_to_console: bool = True) -> Generator[BaseLogger]:
    """Provide a prefix-free logger for one run, writing to the console and a log file.

    Intended to wrap a script's `main`, so the settings and progress output are stored with the
    run. A header with the start time and the command line is logged first. If the run raises, the
    error is logged (with the traceback in the log file only, since the interpreter prints it to
    the console) and the exception propagates. The log file is overwritten and its missing parent
    directories are created.

    Args:
        logfile_path (Path | None): Log file; if `None`, the output only goes to the console.
        print_to_console (bool): Whether the output is also printed to the console. Defaults to
            `True`.

    Yields:
        BaseLogger: The logger for the run, closed on exit.
    """
    with BaseLogger(
        LoggerSettings(print_to_console=print_to_console, logfile_path=logfile_path)
    ) as logger:
        logger.info(f"Run started {datetime.now().astimezone().isoformat(timespec='seconds')}")
        logger.info(f"Command: {' '.join(sys.argv)}")
        if logfile_path is not None:
            logger.info(f"Log file: {logfile_path}")
        logger.info("")
        try:
            yield logger
        except BaseException as error:
            logger.error(f"Run failed: {type(error).__name__}: {error}")
            logger.debug(traceback.format_exc())
            raise


# ==================================================================================================
class StepReporter:
    """Numbers the steps of a run and logs each with its wall-clock time.

    `with reporter.step("Loading data"):` logs `[1/total] Loading data ...` on entry and
    `done in <t> s` on a normal exit; the next step is `[2/total]`, and so on. If the body raises,
    no completion line is logged and the exception propagates.
    """

    def __init__(self, logger: BaseLogger, total_steps: int) -> None:
        """Report the steps of a run with `total_steps` steps to `logger`."""
        self._logger = logger
        self._total_steps = total_steps
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
        self._logger.info(f"[{self._num_started_steps}/{self._total_steps}] {description} ...")
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

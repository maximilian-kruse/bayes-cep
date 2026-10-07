"""Progress output of a run: numbered steps with timings, written to the console and the run log.

Functions:
    run_logger: Context manager providing a prefix-free logger for one run, writing to the
        console and, optionally, a log file.
    report_step: Context manager logging a numbered progress line and the step's wall-clock time.
    Steps: Counter of the numbered steps of a run.
    describe_array: One-line summary of an array's name, shape, and value range.
"""

import sys
import time
import traceback
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
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
@contextmanager
def report_step(
    logger: BaseLogger, index: int, num_steps: int, description: str
) -> Generator[None]:
    """Log a numbered progress line before a step and its wall-clock time after.

    Logs `[index/num_steps] description ...` on entry and `done in <t> s` on a normal exit. If the
    body raises, no completion line is logged and the exception propagates.

    Args:
        logger (BaseLogger): Logger receiving the progress lines.
        index (int): 1-based index of the step.
        num_steps (int): Total number of steps.
        description (str): What the step does.

    Yields:
        None: Control to the body of the step.
    """
    logger.info(f"[{index}/{num_steps}] {description} ...")
    start = time.perf_counter()
    yield
    logger.info(f"      done in {time.perf_counter() - start:.2f} s")


# ==================================================================================================
class Steps:
    """Numbers the steps of a run: `with steps("Loading data"):` logs `[1/total] Loading data ...`,
    the next call `[2/total]`, and so on."""

    def __init__(self, logger: BaseLogger, total: int) -> None:
        """Count the steps of a run with `total` steps, logged to `logger`."""
        self._logger = logger
        self._total = total
        self._count = 0

    def __call__(self, description: str) -> AbstractContextManager[None]:
        """Context manager for the next step; see `report_step`."""
        self._count += 1
        return report_step(self._logger, self._count, self._total, description)


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

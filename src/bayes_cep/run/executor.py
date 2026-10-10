"""Execution of runs: one after the other in this process, or as a job array on SLURM.

The executor knows nothing about studies: it is given `Run` objects and their run directories.
Choosing which runs to execute is up to the caller. The executor is the only owner of the clean-up
of earlier attempts: before a run starts, the files of any earlier attempt in its run directory are
removed (the status record stays, see `RunDirectory.clear_for_new_attempt`). Before a SLURM job may
start, the executor records the run as `SUBMITTED`.

Classes:
    RunOutcome: What became of a run in an execution request.
    LocalExecutorSettings: Run the tasks one after the other in this process.
    SlurmExecutorSettings: Run the tasks as a SLURM job array, with their resources.
    Executor: Interface of the executors.
    LocalExecutor: Executes runs one after the other in this process.
    SlurmExecutor: Executes runs as a SLURM job array via `submitit`.

Functions:
    create_executor: The executor for the given settings.
    describe_executor_settings: JSON-compatible form of executor settings.

Type aliases:
    ExecutorSettings: Where and with which resources to run the tasks.
"""

import sys
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import override

import submitit

from bayes_cep.run.directories import RunDirectory, RunState
from bayes_cep.run.provenance import Environment
from bayes_cep.run.template import Run


# ==================================================================================================
class RunOutcome(StrEnum):
    """What became of one run in an execution request.

    `SKIPPED` (already done) and `ACTIVE` (submitted or running) are decided by the caller; the
    executor only produces `DONE`, `FAILED` and `SUBMITTED`.
    """

    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    ACTIVE = "active"
    SUBMITTED = "submitted"


# ==================================================================================================
@dataclass(frozen=True)
class LocalExecutorSettings:
    """Run the tasks one after the other in this process, with console output."""


# ==================================================================================================
@dataclass(frozen=True)
class SlurmExecutorSettings:
    """Run the tasks on the SLURM cluster as a job array, one task per run.

    Attributes:
        time_min (int): Wall-clock limit per run, in minutes.
        cpus_per_task (int): Cores per run; also the thread count of numpy and BLAS, but not of
            jax.
        mem_gb (int | None): Memory per run in GB; the cluster default if `None`.
        partition (str | None): SLURM partition; the cluster default if `None`.
        max_parallel (int | None): Maximum number of runs executing at once; unlimited if `None`.
    """

    time_min: int = 240
    cpus_per_task: int = 4
    mem_gb: int | None = None
    partition: str | None = None
    max_parallel: int | None = None


ExecutorSettings = LocalExecutorSettings | SlurmExecutorSettings


# ==================================================================================================
def _execute_run(run: Run, run_dir: Path, environment: Environment) -> RunOutcome:
    """Execute one run in its run directory, after removing the files of an earlier attempt.

    This is the function of every task, in process and on SLURM. A failure is recorded in the
    status record of the run and reported, not raised.
    """
    RunDirectory(run_dir).clear_for_new_attempt()
    if run.execute(run_dir, environment) == RunState.DONE:
        return RunOutcome.DONE
    print(f"Run {run.config.run_id} failed, see its status record in {run_dir}", file=sys.stderr)
    return RunOutcome.FAILED


# ==================================================================================================
class Executor(ABC):
    """Executes runs, each in its run directory."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def run(
        self,
        runs: Sequence[Run],
        run_dirs: Sequence[Path],
        environment: Environment,
        wait: bool = True,
    ) -> list[RunOutcome]:
        """Execute the runs, each in its run directory.

        Args:
            runs (Sequence[Run]): The runs to execute.
            run_dirs (Sequence[Path]): The run directory of each run; the files of an earlier
                attempt are removed.
            environment (Environment): The environment recorded by every run, collected once by
                the caller.
            wait (bool): Whether to wait for the runs to finish. `False` queues the runs and
                returns, which only an executor with a scheduler can do. Defaults to `True`.

        Returns:
            list[RunOutcome]: The outcome of each run: `DONE` or `FAILED`, or `SUBMITTED` if not
                waiting.

        Raises:
            ValueError: If `runs` and `run_dirs` differ in length, or if not waiting is not
                possible.
        """

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _check_lengths(runs: Sequence[Run], run_dirs: Sequence[Path]) -> None:
        if len(runs) != len(run_dirs):
            raise ValueError(f"Got {len(runs)} runs but {len(run_dirs)} run directories.")


# ==================================================================================================
class LocalExecutor(Executor):
    """Executes runs one after the other in this process."""

    # ----------------------------------------------------------------------------------------------
    @override
    def run(
        self,
        runs: Sequence[Run],
        run_dirs: Sequence[Path],
        environment: Environment,
        wait: bool = True,
    ) -> list[RunOutcome]:
        self._check_lengths(runs, run_dirs)
        if not wait:
            raise ValueError("Not waiting requires the SLURM executor; local runs always wait.")
        return [
            _execute_run(run, run_dir, environment)
            for run, run_dir in zip(runs, run_dirs, strict=True)
        ]


# ==================================================================================================
class SlurmExecutor(Executor):
    """Executes runs as one SLURM job array via `submitit`.

    SLURM tasks run the Python interpreter of the submitting process, so the pixi environment must
    be reachable from the compute nodes (shared file system).

    Attributes:
        settings (SlurmExecutorSettings): The resources of the runs.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: SlurmExecutorSettings, job_dir: Path, job_name: str) -> None:
        """Create an executor.

        Args:
            settings (SlurmExecutorSettings): The resources of the runs.
            job_dir (Path): Where submitit writes its job folders and logs.
            job_name (str): Name of the jobs on the cluster.
        """
        self.settings = settings
        self._job_dir = job_dir
        self._job_name = job_name

    # ----------------------------------------------------------------------------------------------
    @override
    def run(
        self,
        runs: Sequence[Run],
        run_dirs: Sequence[Path],
        environment: Environment,
        wait: bool = True,
    ) -> list[RunOutcome]:
        self._check_lengths(runs, run_dirs)
        if not runs:
            return []
        directories = [RunDirectory(run_dir) for run_dir in run_dirs]
        for directory in directories:
            directory.record_submitted()
        settings = self.settings
        executor = submitit.AutoExecutor(folder=self._job_dir, cluster="slurm")
        parameters: dict[str, object] = {
            "name": self._job_name,
            "timeout_min": settings.time_min,
            "cpus_per_task": settings.cpus_per_task,
            "mem_gb": settings.mem_gb,
            "slurm_partition": settings.partition,
            "slurm_array_parallelism": settings.max_parallel,
            "slurm_setup": [
                f"export OMP_NUM_THREADS={settings.cpus_per_task}",
                f"export OPENBLAS_NUM_THREADS={settings.cpus_per_task}",
            ],
        }
        try:
            executor.update_parameters(**{k: v for k, v in parameters.items() if v is not None})
            jobs = executor.map_array(
                partial(_execute_run, environment=environment), list(runs), list(run_dirs)
            )
        except BaseException as error:
            # No job exists, so the runs are not queued: do not leave them looking active.
            for directory in directories:
                directory.record_failure(error, traceback.format_exc())
            raise

        if not wait:
            print(f"Submitted {len(jobs)} runs: {[job.job_id for job in jobs]}")
            return [RunOutcome.SUBMITTED] * len(jobs)
        outcomes = []
        for job in jobs:
            try:
                outcomes.append(job.result())
            except Exception as error:  # the job was lost (killed, timed out), not just failed
                print(f"Job {job.job_id} failed: {str(error)[:500]}", file=sys.stderr)
                outcomes.append(RunOutcome.FAILED)
        return outcomes


# ==================================================================================================
def create_executor(settings: ExecutorSettings, job_dir: Path, job_name: str) -> Executor:
    """The executor for the given settings.

    Args:
        settings (ExecutorSettings): Where and with which resources the runs execute.
        job_dir (Path): Where submitit writes its job folders and logs (SLURM only).
        job_name (str): Name of the jobs on the cluster (SLURM only).

    Returns:
        Executor: A `LocalExecutor` or a `SlurmExecutor`.
    """
    if isinstance(settings, SlurmExecutorSettings):
        return SlurmExecutor(settings, job_dir, job_name)
    return LocalExecutor()


# --------------------------------------------------------------------------------------------------
def describe_executor_settings(settings: ExecutorSettings) -> dict[str, object]:
    """JSON-compatible form of executor settings: the kind (class name) and the settings."""
    return {"kind": type(settings).__name__, **asdict(settings)}

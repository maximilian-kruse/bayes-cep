"""Execution of runs: in this process, or through `submitit` as local processes or on SLURM.

The executor knows nothing about studies: it is given `Run` objects and their run directories,
and each becomes one task of a job array. Choosing which runs to execute is up to the caller.

Classes:
    RunOutcome: What became of a run in an execution request.
    ExecutorSettings: Where and with which resources to run the tasks.
    Executor: Execute runs in their run directories.
"""

import os
import shutil
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import submitit

from bayes_cep.run.directories import RunState
from bayes_cep.run.template import Run


# ==================================================================================================
class RunOutcome(StrEnum):
    """What became of one run in an execution request."""

    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    SUBMITTED = "submitted"


# ==================================================================================================
@dataclass(frozen=True)
class ExecutorSettings:
    """Where and with which resources the tasks run.

    Attributes:
        cluster (Literal["debug", "local", "slurm"]): `"debug"` runs the tasks one after the other
            in this process, with console output and without submitit; `"local"` runs each task as
            a separate local process; `"slurm"` submits to the SLURM cluster.
        time_min (int): Wall-clock limit per run, in minutes (`"local"` and `"slurm"`).
        cpus_per_task (int): Cores per run, also the thread count of numpy and jax.
        mem_gb (int | None): Memory per run in GB (SLURM only); the cluster default if `None`.
        partition (str | None): SLURM partition; the cluster default if `None`.
        max_parallel (int | None): Maximum number of runs executing at once; unlimited if `None`.
    """

    cluster: Literal["debug", "local", "slurm"] = "debug"
    time_min: int = 240
    cpus_per_task: int = 4
    mem_gb: int | None = None
    partition: str | None = None
    max_parallel: int | None = None


# ==================================================================================================
class Executor:
    """Executes runs through `submitit`, as one job array per batch.

    Tasks run the Python interpreter of the submitting process, so the pixi environment must be
    reachable from the compute nodes (shared file system).

    Attributes:
        settings (ExecutorSettings): Where and with which resources the runs execute.
    """

    def __init__(self, settings: ExecutorSettings, job_dir: Path, job_name: str) -> None:
        """Create an executor.

        Args:
            settings (ExecutorSettings): Where and with which resources the runs execute.
            job_dir (Path): Where submitit writes its job folders and logs.
            job_name (str): Name of the jobs on the cluster.
        """
        self.settings = settings
        self._job_dir = job_dir
        self._job_name = job_name

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _execute_run_as_job(run: Run, run_dir: Path) -> RunOutcome:
        """Execute one run as a job; the function every task of the job array calls.

        Unlike `Run.execute`, a failed run raises, so that the scheduler marks the job as failed.
        Any earlier partial run directory is removed first.

        Args:
            run (Run): The run to execute.
            run_dir (Path): Its run directory.

        Returns:
            RunOutcome: `DONE`.

        Raises:
            RuntimeError: If the run failed; the error is recorded in the status record of the run.
        """
        if run_dir.exists():
            shutil.rmtree(run_dir)
        print(f"=== run {run.config.run_id} ({type(run).__name__})")
        if run.execute(run_dir) == RunState.FAILED:
            raise RuntimeError(
                f"Run {run.config.run_id} failed, see its status record in {run_dir}."
            )
        return RunOutcome.DONE

    # ----------------------------------------------------------------------------------------------
    def _submit_batch(self, runs: Sequence[Run], run_dirs: Sequence[Path]) -> list[submitit.Job]:
        """Submit the runs as one job array, without waiting for them."""
        settings = self.settings
        executor = submitit.AutoExecutor(folder=self._job_dir, cluster=settings.cluster)
        parameters: dict[str, object] = {
            "name": self._job_name,
            "timeout_min": settings.time_min,
            "cpus_per_task": settings.cpus_per_task,
            "mem_gb": settings.mem_gb,
            "slurm_partition": settings.partition,
            "slurm_array_parallelism": settings.max_parallel,
            "slurm_setup": self._thread_limit_commands(),
        }
        executor.update_parameters(**{k: v for k, v in parameters.items() if v is not None})
        # Local jobs inherit the environment. submitit's `local_setup` must not carry the thread
        # limits: it wraps each job in a shell, and the jobs then never find their input.
        with self._local_thread_limits():
            return executor.map_array(self._execute_run_as_job, list(runs), list(run_dirs))

    # ----------------------------------------------------------------------------------------------
    def _thread_limits(self) -> dict[str, str]:
        """Environment variables limiting the threads of numpy and BLAS to the cores per run."""
        cores = str(self.settings.cpus_per_task)
        return {"OMP_NUM_THREADS": cores, "OPENBLAS_NUM_THREADS": cores}

    # ----------------------------------------------------------------------------------------------
    def _thread_limit_commands(self) -> list[str]:
        """The thread limits as shell commands, for the batch script of a SLURM job."""
        return [f"export {name}={value}" for name, value in self._thread_limits().items()]

    # ----------------------------------------------------------------------------------------------
    @contextmanager
    def _local_thread_limits(self) -> Generator[None]:
        """Set the thread limits in this process's environment while local jobs are launched.

        Local jobs are subprocesses and inherit it. Does nothing for other clusters: in process,
        the limits would come too late, and SLURM jobs get them from their batch script.
        """
        if self.settings.cluster != "local":
            yield
            return
        previous = {name: os.environ.get(name) for name in self._thread_limits()}
        os.environ.update(self._thread_limits())
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    # ----------------------------------------------------------------------------------------------
    def _run_in_process(self, runs: Sequence[Run], run_dirs: Sequence[Path]) -> list[RunOutcome]:
        """Execute the runs one after the other in this process.

        Not done through submitit, whose debug mode opens an interactive debugger when a job fails.
        A failed run is reported and the others continue.
        """
        outcomes: list[RunOutcome] = []
        for run, run_dir in zip(runs, run_dirs, strict=True):
            try:
                outcomes.append(self._execute_run_as_job(run, run_dir))
            except Exception as error:
                print(f"Run failed: {error}", file=sys.stderr)
                outcomes.append(RunOutcome.FAILED)
        return outcomes

    # ----------------------------------------------------------------------------------------------
    def run(
        self, runs: Sequence[Run], run_dirs: Sequence[Path], wait: bool = True
    ) -> list[RunOutcome]:
        """Execute the runs, each in its run directory.

        With `"local"` and `max_parallel`, the runs are submitted in batches of that size.

        Args:
            runs (Sequence[Run]): The runs to execute.
            run_dirs (Sequence[Path]): The run directory of each run; earlier contents are removed.
            wait (bool): Whether to wait for the runs to finish. `False` queues the runs and
                returns, which is only possible on `"slurm"`, where the scheduler owns the jobs.
                Defaults to `True`.

        Returns:
            list[RunOutcome]: The outcome of each run: `DONE` or `FAILED`, or `SUBMITTED` if not
                waiting.

        Raises:
            ValueError: If `runs` and `run_dirs` differ in length, or if not waiting on a cluster
                other than `"slurm"`.
        """
        if len(runs) != len(run_dirs):
            raise ValueError(f"Got {len(runs)} runs but {len(run_dirs)} run directories.")
        if not wait and self.settings.cluster != "slurm":
            raise ValueError(
                f"Not waiting requires the slurm cluster, got {self.settings.cluster!r}."
            )
        if self.settings.cluster == "debug":
            return self._run_in_process(runs, run_dirs)
        local_limit = self.settings.max_parallel if self.settings.cluster == "local" else None
        batch_size = local_limit or max(len(runs), 1)
        outcomes: list[RunOutcome] = []
        for start in range(0, len(runs), batch_size):
            jobs = self._submit_batch(
                runs[start : start + batch_size], run_dirs[start : start + batch_size]
            )
            if not wait:
                print(f"Submitted {len(jobs)} runs: {[job.job_id for job in jobs]}")
                outcomes.extend([RunOutcome.SUBMITTED] * len(jobs))
                continue
            for job in jobs:
                try:
                    outcomes.append(job.result())
                except Exception as error:
                    print(f"Job {job.job_id} failed: {str(error)[:500]}", file=sys.stderr)
                    outcomes.append(RunOutcome.FAILED)
        return outcomes

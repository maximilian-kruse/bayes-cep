"""Execution of runs: in this process, or through `submitit` as local processes or on SLURM.

The executor knows nothing about studies: it is given `Run` objects and their run directories.
Choosing which runs to execute is up to the caller. Before a job may start, the executor records
the run as `SUBMITTED`; the job removes the files of any earlier attempt, and records the rest.

Classes:
    RunOutcome: What became of a run in an execution request.
    ExecutorSettings: Where and with which resources to run the tasks.
    Executor: Execute runs in their run directories.
"""

import os
import sys
import traceback
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Literal

import submitit

from bayes_cep.run.directories import RunDirectory, RunState
from bayes_cep.run.metadata import Environment
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
class ExecutorSettings:
    """Where and with which resources the tasks run.

    Attributes:
        cluster (Literal["debug", "local", "slurm"]): `"debug"` runs the tasks one after the other
            in this process, with console output and without submitit; `"local"` runs each task as
            a separate local process; `"slurm"` submits to the SLURM cluster.
        time_min (int): Wall-clock limit per run, in minutes (`"local"` and `"slurm"`).
        cpus_per_task (int): Cores per run; also the thread count of numpy and BLAS (not of jax).
        mem_gb (int | None): Memory per run in GB (SLURM only); the cluster default if `None`.
        partition (str | None): SLURM partition; the cluster default if `None`.
        max_parallel (int | None): Maximum number of runs executing at once (`"local"` and
            `"slurm"`); unlimited if `None`. Ignored for `"debug"`, where runs are sequential.
    """

    cluster: Literal["debug", "local", "slurm"] = "debug"
    time_min: int = 240
    cpus_per_task: int = 4
    mem_gb: int | None = None
    partition: str | None = None
    max_parallel: int | None = None


# ==================================================================================================
class Executor:
    """Executes runs in process, or through `submitit` as local processes or as a SLURM job array.

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
    def _execute_run_as_job(run: Run, run_dir: Path, environment: Environment) -> None:
        """Execute one run as a job; the function every task calls.

        The files of an earlier attempt are removed first, except the status record.

        Args:
            run (Run): The run to execute.
            run_dir (Path): Its run directory.
            environment (Environment): The environment to record, collected by the submitter.

        Raises:
            RuntimeError: If the run failed, so that the scheduler marks the job as failed; the
                error itself is recorded in the status record of the run.
        """
        RunDirectory(run_dir).clear_for_new_attempt()
        print(f"=== run {run.config.run_id} ({type(run).__name__})")
        if run.execute(run_dir, environment) == RunState.FAILED:
            raise RuntimeError(
                f"Run {run.config.run_id} failed, see its status record in {run_dir}."
            )

    # ----------------------------------------------------------------------------------------------
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
        if not runs:
            return []
        if self.settings.cluster == "debug":
            return [
                self._run_in_process(run, run_dir, environment)
                for run, run_dir in zip(runs, run_dirs, strict=True)
            ]
        for run_dir in run_dirs:
            RunDirectory(run_dir).record_submitted()
        try:
            if self.settings.cluster == "local":
                return self._run_local(runs, run_dirs, environment)
            return self._run_slurm_array(runs, run_dirs, environment, wait)
        except BaseException as error:
            self._record_unstarted_as_failed(run_dirs, error)
            raise

    # ----------------------------------------------------------------------------------------------
    def _run_in_process(self, run: Run, run_dir: Path, environment: Environment) -> RunOutcome:
        """Execute the run in this process.

        Not done through submitit, whose debug mode opens an interactive debugger when a job fails.
        """
        RunDirectory(run_dir).clear_for_new_attempt()
        state = run.execute(run_dir, environment)
        if state == RunState.FAILED:
            print(
                f"Run {run.config.run_id} failed, see its status record in {run_dir}",
                file=sys.stderr,
            )
            return RunOutcome.FAILED
        return RunOutcome.DONE

    # ----------------------------------------------------------------------------------------------
    def _run_local(
        self, runs: Sequence[Run], run_dirs: Sequence[Path], environment: Environment
    ) -> list[RunOutcome]:
        """Run each task as a local process, at most `max_parallel` of them at once.

        One thread per slot submits a task and waits for it, so that a finished task is replaced at
        once, without waiting for the slowest of a batch.
        """
        executor = self._build_submitit_executor()
        task = partial(self._execute_run_as_job, environment=environment)

        def run_and_wait(run: Run, run_dir: Path) -> RunOutcome:
            return self._wait_for(executor.submit(task, run, run_dir))

        slots = self.settings.max_parallel or len(runs)
        with self._local_thread_limits(), ThreadPoolExecutor(max_workers=slots) as pool:
            return list(pool.map(run_and_wait, runs, run_dirs))

    # ----------------------------------------------------------------------------------------------
    def _run_slurm_array(
        self,
        runs: Sequence[Run],
        run_dirs: Sequence[Path],
        environment: Environment,
        wait: bool,
    ) -> list[RunOutcome]:
        """Submit the tasks as one SLURM job array; wait for them if `wait`."""
        executor = self._build_submitit_executor()
        task = partial(self._execute_run_as_job, environment=environment)
        jobs = executor.map_array(task, list(runs), list(run_dirs))
        if not wait:
            print(f"Submitted {len(jobs)} runs: {[job.job_id for job in jobs]}")
            return [RunOutcome.SUBMITTED] * len(jobs)
        return [self._wait_for(job) for job in jobs]

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _wait_for(job: submitit.Job) -> RunOutcome:
        """Wait for a job; `FAILED` if it raised or was lost, with a message on stderr."""
        try:
            job.result()
        except Exception as error:
            print(f"Job {job.job_id} failed: {str(error)[:500]}", file=sys.stderr)
            return RunOutcome.FAILED
        return RunOutcome.DONE

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _record_unstarted_as_failed(run_dirs: Sequence[Path], error: BaseException) -> None:
        """Mark the runs that are still `SUBMITTED` as failed, after the submission was aborted."""
        formatted_traceback = traceback.format_exc()
        for run_dir in run_dirs:
            directory = RunDirectory(run_dir)
            if directory.read_state() == RunState.SUBMITTED:
                directory.record_failure(error, formatted_traceback)

    # ----------------------------------------------------------------------------------------------
    def _build_submitit_executor(self) -> submitit.AutoExecutor:
        """The submitit executor for the `local` or `slurm` cluster, with the run resources."""
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
        return executor

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

        Local jobs are subprocesses and inherit it. (submitit's `local_setup` must not carry the
        limits: it wraps each job in a shell, and the jobs then never find their input.)
        """
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

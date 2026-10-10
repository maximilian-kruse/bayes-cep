"""Execution of runs: one after the other in this process, or as a job array on SLURM.

The executor knows nothing about studies: it is given `Run` objects and their run directories.
Choosing which runs to execute is up to the caller. Before a SLURM job may start, the executor
records the run as `SUBMITTED`; the job removes the files of any earlier attempt, and records the
rest.

Classes:
    RunOutcome: What became of a run in an execution request.
    ExecutorSettings: Where and with which resources to run the tasks.
    Executor: Execute runs in their run directories.
"""

import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Literal

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
class ExecutorSettings:
    """Where and with which resources the tasks run.

    Attributes:
        cluster (Literal["local", "slurm"]): `"local"` runs the tasks one after the other in this
            process, with console output; the resource settings below do not apply. `"slurm"`
            submits the tasks to the SLURM cluster as a job array.
        time_min (int): Wall-clock limit per run, in minutes (SLURM only).
        cpus_per_task (int): Cores per run; also the thread count of numpy and BLAS, but not of
            jax (SLURM only).
        mem_gb (int | None): Memory per run in GB; the cluster default if `None` (SLURM only).
        partition (str | None): SLURM partition; the cluster default if `None`.
        max_parallel (int | None): Maximum number of runs executing at once; unlimited if `None`
            (SLURM only).
    """

    cluster: Literal["local", "slurm"] = "local"
    time_min: int = 240
    cpus_per_task: int = 4
    mem_gb: int | None = None
    partition: str | None = None
    max_parallel: int | None = None

    def __post_init__(self) -> None:
        if self.cluster not in ("local", "slurm"):
            raise ValueError(f"cluster must be 'local' or 'slurm', got {self.cluster!r}.")


# ==================================================================================================
class Executor:
    """Executes runs one after the other in this process, or as a SLURM job array via `submitit`.

    SLURM tasks run the Python interpreter of the submitting process, so the pixi environment must
    be reachable from the compute nodes (shared file system).

    Attributes:
        settings (ExecutorSettings): Where and with which resources the runs execute.
    """

    def __init__(self, settings: ExecutorSettings, job_dir: Path, job_name: str) -> None:
        """Create an executor.

        Args:
            settings (ExecutorSettings): Where and with which resources the runs execute.
            job_dir (Path): Where submitit writes its job folders and logs (SLURM only).
            job_name (str): Name of the jobs on the cluster.
        """
        self.settings = settings
        self._job_dir = job_dir
        self._job_name = job_name

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
        if self.settings.cluster == "local":
            return self._run_locally(runs, run_dirs, environment)
        return self._run_on_slurm(runs, run_dirs, environment, wait)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _execute_run(run: Run, run_dir: Path, environment: Environment) -> RunOutcome:
        """Execute one run in its run directory, after removing the files of an earlier attempt.

        This is the function of every task, in process and on SLURM. A failure is recorded in the
        status record of the run and reported, not raised.
        """
        RunDirectory(run_dir).clear_for_new_attempt()
        if run.execute(run_dir, environment) == RunState.DONE:
            return RunOutcome.DONE
        print(
            f"Run {run.config.run_id} failed, see its status record in {run_dir}", file=sys.stderr
        )
        return RunOutcome.FAILED

    # ----------------------------------------------------------------------------------------------
    def _run_locally(
        self, runs: Sequence[Run], run_dirs: Sequence[Path], environment: Environment
    ) -> list[RunOutcome]:
        """Execute the runs one after the other in this process."""
        return [
            self._execute_run(run, run_dir, environment)
            for run, run_dir in zip(runs, run_dirs, strict=True)
        ]

    # ----------------------------------------------------------------------------------------------
    def _run_on_slurm(
        self,
        runs: Sequence[Run],
        run_dirs: Sequence[Path],
        environment: Environment,
        wait: bool,
    ) -> list[RunOutcome]:
        """Submit the runs as one SLURM job array; wait for the jobs if `wait`."""
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
                partial(self._execute_run, environment=environment), list(runs), list(run_dirs)
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

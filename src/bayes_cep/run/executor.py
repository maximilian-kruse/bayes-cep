"""Execution of the runs of a study through `submitit`: locally, in process, or on SLURM.

One job array task per unfinished run. The task needs nothing but the study directory and a run
index, since the configuration is read back from `study/runs.json`.

Constants:
    SKIPPED: Outcome of a run that was already finished.

Classes:
    ExecutorSettings: Where and with which resources to run the tasks.
    Executor: Submit the unfinished runs of a study.

Functions:
    run_task: Execute one run of a study as a job array task.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import submitit

from bayes_cep.run.directories import DONE, FAILED, RunDirectory, StudyDirectory
from bayes_cep.run.study import load_resolved_runs

SKIPPED = "skipped"


# ==================================================================================================
@dataclass(frozen=True)
class ExecutorSettings:
    """Where and with which resources the tasks run.

    Attributes:
        cluster (Literal["debug", "local", "slurm"]): `"debug"` runs the tasks one after the other
            in this process, with console output; `"local"` runs each task as a separate local
            process; `"slurm"` submits to the SLURM cluster.
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
def run_task(study_dir: Path, index: int, force: bool = False) -> str:
    """Execute one run of a study as a job array task.

    Unlike `Run.execute`, a failed run raises, so that the scheduler marks the task as failed. Any
    earlier partial run directory is removed first.

    Args:
        study_dir (Path): Study directory.
        index (int): Index of the run.
        force (bool): Whether to rerun a finished run. Defaults to `False`.

    Returns:
        str: `DONE` or `SKIPPED`.

    Raises:
        RuntimeError: If the run failed; the error is recorded in the run's `status.json`.
    """
    run_type, runs = load_resolved_runs(study_dir)
    run = runs[index]
    run_dir = StudyDirectory(study_dir).run(run.run_id).path
    if RunDirectory(run_dir).state() == DONE and not force:
        return SKIPPED
    if run_dir.exists():
        shutil.rmtree(run_dir)
    print(f"=== run {index} ({run.run_id}) {run.overrides}")
    if run_type(run.config).execute(run_dir) == FAILED:
        raise RuntimeError(f"Run {index} ({run.run_id}) failed, see {run_dir / 'status.json'}.")
    return DONE


# ==================================================================================================
class Executor:
    """Submits the unfinished runs of a study as one `submitit` job array.

    Finished runs are not submitted unless forced, so submitting again after failures or timeouts
    only repeats those runs. Tasks run the Python interpreter of the submitting process, so the
    pixi environment must be reachable from the compute nodes (shared file system).
    """

    def __init__(self, study_dir: Path, settings: ExecutorSettings) -> None:
        """Bind the executor to a created study directory."""
        self.study_dir = study_dir
        self.settings = settings
        self._study = StudyDirectory(study_dir)
        self._ids = [entry["id"] for entry in self._study.run_index()]

    # ----------------------------------------------------------------------------------------------
    def _pending(self, indices: list[int] | None, force: bool) -> tuple[list[int], list[int]]:
        """Split the selected indices into the runs to execute and the finished ones."""
        selected = list(range(len(self._ids))) if indices is None else indices
        for index in selected:
            if not 0 <= index < len(self._ids):
                raise ValueError(f"Run index {index} is out of range for {len(self._ids)} runs.")
        finished = [
            index
            for index in selected
            if not force and self._study.run(self._ids[index]).state() == DONE
        ]
        return [index for index in selected if index not in finished], finished

    # ----------------------------------------------------------------------------------------------
    def submit(
        self, indices: list[int] | None = None, force: bool = False
    ) -> dict[int, submitit.Job]:
        """Submit the unfinished runs as one job array and return without waiting.

        Args:
            indices (list[int] | None): Runs to consider; all runs if `None`.
            force (bool): Whether to also resubmit finished runs. Defaults to `False`.

        Returns:
            dict[int, submitit.Job]: The jobs by run index; empty if nothing needs to run.

        Raises:
            ValueError: If an index is out of range.
        """
        pending, _ = self._pending(indices, force)
        if not pending:
            return {}
        settings = self.settings
        executor = submitit.AutoExecutor(folder=self._study.slurm_dir, cluster=settings.cluster)
        if settings.cluster != "debug":
            thread_setup = [
                f"export OMP_NUM_THREADS={settings.cpus_per_task}",
                f"export OPENBLAS_NUM_THREADS={settings.cpus_per_task}",
            ]
            parameters: dict[str, object] = {
                "name": self.study_dir.name,
                "timeout_min": settings.time_min,
                "cpus_per_task": settings.cpus_per_task,
                "slurm_setup": thread_setup,
                "local_setup": thread_setup,
                "mem_gb": settings.mem_gb,
                "slurm_partition": settings.partition,
                "slurm_array_parallelism": settings.max_parallel,
            }
            executor.update_parameters(**{k: v for k, v in parameters.items() if v is not None})
        jobs = executor.map_array(
            run_task, [self.study_dir] * len(pending), pending, [force] * len(pending)
        )
        return dict(zip(pending, jobs, strict=True))

    # ----------------------------------------------------------------------------------------------
    def run(self, indices: list[int] | None = None, force: bool = False) -> dict[int, str]:
        """Execute the unfinished runs and wait for them.

        With `"local"` and `max_parallel`, the runs are submitted in batches of that size.

        Args:
            indices (list[int] | None): Runs to consider; all runs if `None`.
            force (bool): Whether to also rerun finished runs. Defaults to `False`.

        Returns:
            dict[int, str]: Outcome by run index: `DONE`, `FAILED` or `SKIPPED`.
        """
        pending, finished = self._pending(indices, force)
        outcomes = dict.fromkeys(finished, SKIPPED)
        local_limit = self.settings.max_parallel if self.settings.cluster == "local" else None
        batch_size = local_limit or max(len(pending), 1)
        for start in range(0, len(pending), batch_size):
            jobs = self.submit(pending[start : start + batch_size], force)
            for index, job in jobs.items():
                try:
                    outcomes[index] = job.result()
                except Exception:
                    outcomes[index] = FAILED
        return dict(sorted(outcomes.items()))

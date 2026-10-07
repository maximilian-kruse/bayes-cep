"""Command line of the simulation studies: create, run, inspect and summarize.

A study is a Python module in `studies/` defining `STUDY`: a base run configuration plus the
sweeps over it. `create` resolves it into a fixed list of runs and writes the study directory;
`run` executes runs from that directory in this process (`debug`) or as local processes (`local`).

    pixi run study create studies/prior_investigation.py
    pixi run study show studies/prior_investigation.py
    pixi run study run working_data/prior_investigation --executor.cluster local
    pixi run study status working_data/prior_investigation
    pixi run study collect working_data/prior_investigation
    pixi run -e dev study report working_data/prior_investigation

On a cluster, `submit` sends the unfinished runs to SLURM as one job array via `submitit` (task `i`
runs run `i`); the tasks use this pixi environment, so it must be reachable from the compute nodes:

    pixi run study submit working_data/prior_investigation --executor.time-min 120

Finished runs are skipped, so submitting again only repeats failed or unfinished runs. Logs go to
`<study>/slurm/`; `status` shows the progress.
"""

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from bayes_cep.run.collector import RunTableCollector
from bayes_cep.run.config import format_tree
from bayes_cep.run.directories import DONE, FAILED, StudyDirectory, read_json
from bayes_cep.run.executor import Executor, ExecutorSettings
from bayes_cep.run.study import load_resolved_runs, load_study


# ==================================================================================================
@dataclass(frozen=True)
class Create:
    """Resolve a study module and write the study directory.

    Attributes:
        module (Path): Python file defining `STUDY`.
        root (Path): Directory holding all study directories.
    """

    module: tyro.conf.Positional[Path]
    root: Path = Path("working_data")


# ==================================================================================================
@dataclass(frozen=True)
class Show:
    """Print an overview of a study module, a study directory or a config JSON file.

    Attributes:
        path (Path): A study module (`.py`), a study directory, or a `config.json`.
    """

    path: tyro.conf.Positional[Path]


# ==================================================================================================
@dataclass(frozen=True)
class Run:
    """Execute the unfinished runs and wait for them.

    Attributes:
        study_dir (Path): Study directory.
        index (int | None): Only this run; all runs if `None`.
        force (bool): Whether to rerun finished runs.
        executor (ExecutorSettings): Where the runs execute.
    """

    study_dir: tyro.conf.Positional[Path]
    index: int | None = None
    force: bool = False
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)


# ==================================================================================================
@dataclass(frozen=True)
class Submit:
    """Submit the unfinished runs to the cluster as one job array (via submitit).

    Attributes:
        study_dir (Path): Study directory.
        index (int | None): Only this run; all unfinished runs if `None`.
        force (bool): Whether to also resubmit finished runs.
        executor (ExecutorSettings): Cluster and resources per run.
    """

    study_dir: tyro.conf.Positional[Path]
    index: int | None = None
    force: bool = False
    executor: ExecutorSettings = field(default_factory=lambda: ExecutorSettings(cluster="slurm"))


# ==================================================================================================
@dataclass(frozen=True)
class Status:
    """Print the state of every run.

    Attributes:
        study_dir (Path): Study directory.
    """

    study_dir: tyro.conf.Positional[Path]


# ==================================================================================================
@dataclass(frozen=True)
class Collect:
    """Write the cross-run summary table `summary/run_table.parquet`.

    Attributes:
        study_dir (Path): Study directory.
    """

    study_dir: tyro.conf.Positional[Path]


# ==================================================================================================
@dataclass(frozen=True)
class Report:
    """Render the plots of finished runs (needs the dev environment).

    Attributes:
        study_dir (Path): Study directory.
        index (int | None): Only this run; all finished runs if `None`.
    """

    study_dir: tyro.conf.Positional[Path]
    index: int | None = None


# ==================================================================================================
def main(command: Create | Show | Run | Submit | Status | Collect | Report) -> None:
    """Dispatch the subcommand."""
    match command:
        case Create():
            study = load_study(command.module)
            study_dir = study.create(command.module, command.root)
            print(f"Created {study_dir} with {len(study.resolve())} runs.")
        case Show():
            if command.path.suffix == ".py":
                print(load_study(command.path).describe())
            elif command.path.suffix == ".json":
                print(format_tree(read_json(command.path)))
            else:
                print(load_study(StudyDirectory(command.path).definition_path).describe())
        case Run():
            indices = None if command.index is None else [command.index]
            outcomes = Executor(command.study_dir, command.executor).run(indices, command.force)
            print(dict(Counter(outcomes.values())))
            if FAILED in outcomes.values():
                sys.exit(1)
        case Submit():
            indices = None if command.index is None else [command.index]
            jobs = Executor(command.study_dir, command.executor).submit(indices, command.force)
            print(f"Submitted {len(jobs)} runs: {[job.job_id for job in jobs.values()]}")
        case Status():
            table = RunTableCollector(command.study_dir).run_table()
            print(table[["index", "run_id", "state"]].to_string(index=False))
            print(dict(Counter(table["state"])))
        case Collect():
            study = load_study(StudyDirectory(command.study_dir).definition_path)
            collector = study.collector(command.study_dir)
            print(f"Wrote {collector.collect()}")
            print(collector.run_table().to_string(index=False))
        case Report():
            run_type, runs = load_resolved_runs(command.study_dir)
            for run in runs:
                if command.index is not None and run.index != command.index:
                    continue
                run_dir = StudyDirectory(command.study_dir).run(run.run_id)
                if run_dir.state() != DONE:
                    continue
                print(f"Plotting run {run.index} ({run.run_id})")
                run_type(run.config).report(run_dir.path)


if __name__ == "__main__":
    main(tyro.cli(Create | Show | Run | Submit | Status | Collect | Report))

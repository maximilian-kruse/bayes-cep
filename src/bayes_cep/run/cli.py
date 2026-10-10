"""Command line interface of simulation studies: create, run, inspect and summarize.

A study is a Python module in `studies/` defining `STUDY`: a base run configuration plus the
sweeps over it. `create` resolves it into a fixed list of runs and writes the study directory;
`run` executes runs from that directory in this process (`debug`), as local processes (`local`) or
on a SLURM cluster (`slurm`).

Example:

    pixi run study create studies/prior_investigation.py
    pixi run study show studies/prior_investigation.py
    pixi run study run working_data/prior_investigation --executor.cluster local
    pixi run study status working_data/prior_investigation
    pixi run study collect working_data/prior_investigation
    pixi run study report working_data/prior_investigation

On a cluster, `--executor.cluster slurm` sends the unfinished runs to SLURM as one job array via
`submitit` (task `i` runs run `i`); the tasks use this pixi environment, so it must be reachable
from the compute nodes. With `--no-wait`, the command returns after queueing:

    pixi run study run working_data/prior_investigation --executor.cluster slurm --no-wait

Finished runs are skipped, so running again only repeats failed or unfinished runs. Logs go to
`<study>/jobs/`; `status` shows the progress. To detach a `local` run, use `nohup` or `tmux`.
"""

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from bayes_cep.run.collector import RunTableCollector
from bayes_cep.run.config import format_config_tree
from bayes_cep.run.directories import RunDirectory
from bayes_cep.run.executor import ExecutorSettings, RunOutcome
from bayes_cep.run.study import Study


# ==================================================================================================
@dataclass(frozen=True)
class CreateCommand:
    """Resolve a study module and write the study directory.

    Attributes:
        module (Path): Python file defining `STUDY`.
        root (Path): Directory holding all study directories.
    """

    module: tyro.conf.Positional[Path]
    root: Path = Path("working_data")


# ==================================================================================================
@dataclass(frozen=True)
class ShowCommand:
    """Print an overview of a study module, a study directory or a config JSON file.

    Attributes:
        path (Path): A study module (`.py`), a study directory, or a `config.json`.
    """

    path: tyro.conf.Positional[Path]


# ==================================================================================================
@dataclass(frozen=True)
class RunCommand:
    """Execute the unfinished runs, in this process, as local processes or on SLURM.

    Attributes:
        study_dir (Path): Study directory.
        index (int | None): Only this run; all runs if `None`.
        force (bool): Whether to rerun finished runs.
        wait (bool): Whether to wait for the runs to finish; `False` (queue and return) is only
            possible with the `"slurm"` cluster, where the scheduler owns the jobs.
        executor (ExecutorSettings): Where and with which resources the runs execute.
    """

    study_dir: tyro.conf.Positional[Path]
    index: int | None = None
    force: bool = False
    wait: bool = True
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)


# ==================================================================================================
@dataclass(frozen=True)
class StatusCommand:
    """Print the state of every run.

    Attributes:
        study_dir (Path): Study directory.
    """

    study_dir: tyro.conf.Positional[Path]


# ==================================================================================================
@dataclass(frozen=True)
class CollectCommand:
    """Write the cross-run summary table `summary/run_table.parquet`.

    Attributes:
        study_dir (Path): Study directory.
    """

    study_dir: tyro.conf.Positional[Path]


# ==================================================================================================
@dataclass(frozen=True)
class ReportCommand:
    """Render the plots of finished runs.

    Attributes:
        study_dir (Path): Study directory.
        index (int | None): Only this run; all finished runs if `None`.
    """

    study_dir: tyro.conf.Positional[Path]
    index: int | None = None


# ==================================================================================================
Command = (
    Annotated[CreateCommand, tyro.conf.subcommand("create")]
    | Annotated[ShowCommand, tyro.conf.subcommand("show")]
    | Annotated[RunCommand, tyro.conf.subcommand("run")]
    | Annotated[StatusCommand, tyro.conf.subcommand("status")]
    | Annotated[CollectCommand, tyro.conf.subcommand("collect")]
    | Annotated[ReportCommand, tyro.conf.subcommand("report")]
)


# ==================================================================================================
def main(command: Command) -> None:
    """Dispatch the subcommand."""
    match command:
        case CreateCommand():
            # Resolve the study module into its run list and write the study directory.
            study = Study.load_from_module_file(command.module)
            study_dir = study.create_directory(command.module, command.root)
            print(f"Created {study_dir} with {len(study.resolve_runs())} runs.")
        case ShowCommand():
            # Print an overview of a study module, a study directory or a single run config.
            if command.path.suffix == ".py":
                print(Study.load_from_module_file(command.path).describe())
            elif command.path.suffix == ".json":
                print(
                    format_config_tree(
                        RunDirectory(command.path.parent).records.read_record(command.path.name)
                    )
                )
            else:
                print(Study.load_from_directory(command.path).describe())
        case RunCommand():
            # Execute unfinished runs (blocking), or queue them on SLURM and return (`--no-wait`).
            indices = None if command.index is None else [command.index]
            outcomes = Study.load_from_directory(command.study_dir).execute_runs(
                command.study_dir, command.executor, indices, command.force, command.wait
            )
            print(dict(Counter(str(outcome) for outcome in outcomes.values())))
            if RunOutcome.FAILED in outcomes.values():
                sys.exit(1)
        case StatusCommand():
            # Print the state (pending/running/done/failed) of every run.
            table = RunTableCollector(command.study_dir).run_table()
            print(table[["index", "run_id", "state"]].to_string(index=False))
            print(dict(Counter(table["state"])))
        case CollectCommand():
            # Write the cross-run table, then run the study's own analysis.
            collector = Study.load_from_directory(command.study_dir).collector(command.study_dir)
            print(f"Wrote {collector.collect()}")
            print(collector.run_table().to_string(index=False))
        case ReportCommand():
            # Plot the finished runs; unfinished ones are skipped.
            Study.load_from_directory(command.study_dir).plot_finished_runs(
                command.study_dir, command.index
            )


if __name__ == "__main__":
    main(tyro.cli(Command))

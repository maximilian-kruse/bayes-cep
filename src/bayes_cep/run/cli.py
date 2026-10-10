"""Command line interface of simulation studies: create, run, inspect and summarize.

A study is a Python module in `studies/` defining `STUDY`: a base run configuration plus the
sweeps over it. Every command names the study by its module; the study directory is
`<root>/<study name>` with `--root` defaulting to `working_data`. `create` resolves the module into
a fixed list of runs and writes the study directory; `run` executes the runs one after the other in
this process (`local`) or on a SLURM cluster (`slurm`).

Example:

    pixi run study create studies/prior_investigation.py
    pixi run study show studies/prior_investigation.py
    pixi run study run studies/prior_investigation.py --executor.cluster local
    pixi run study status studies/prior_investigation.py
    pixi run study collect studies/prior_investigation.py
    pixi run study report studies/prior_investigation.py

On a cluster, `--executor.cluster slurm` sends the unfinished runs to SLURM as one job array via
`submitit` (one task per unfinished run); the tasks use this pixi environment, so it must be
reachable from the compute nodes. With `--no-wait`, the command returns after queueing:

    pixi run study run studies/prior_investigation.py --executor.cluster slurm --no-wait

Finished runs are skipped, so running again only repeats failed or unstarted runs. Runs that are
submitted or running are skipped too (a second job would delete the files of the first); after a
crash or an interrupted submission, `--include-active` restarts them. SLURM logs go to
`<study>/jobs/`; `status` shows the progress. To detach a `local` run, use `nohup` or `tmux`.
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from bayes_cep.run.config import format_config_tree
from bayes_cep.run.executor import ExecutorSettings, RunOutcome
from bayes_cep.run.study import CreatedStudy, Study


# ==================================================================================================
@dataclass(frozen=True)
class StudyCommand:
    """Base of the commands that act on a study.

    Attributes:
        module (Path): Python file defining `STUDY`.
        root (Path): Directory holding all study directories.
    """

    module: tyro.conf.Positional[Path]
    root: Path = Path("working_data")


# ==================================================================================================
@dataclass(frozen=True)
class CreateCommand(StudyCommand):
    """Resolve a study module and write the study directory."""


# ==================================================================================================
@dataclass(frozen=True)
class ShowCommand:
    """Print an overview of a study module or of a `config.json` file.

    Attributes:
        path (Path): A study module (`.py`) or a `config.json`.
    """

    path: tyro.conf.Positional[Path]


# ==================================================================================================
@dataclass(frozen=True)
class RunCommand(StudyCommand):
    """Execute the unfinished runs, one after the other in this process or on SLURM.

    Attributes:
        index (int | None): Only this run; all runs if `None`.
        force (bool): Whether to rerun finished runs.
        include_active (bool): Whether to restart runs that are submitted or running, e.g. after a
            crash.
        wait (bool): Whether to wait for the runs to finish; `False` (queue and return) is only
            possible with the `"slurm"` cluster, where the scheduler owns the jobs.
        executor (ExecutorSettings): Where and with which resources the runs execute.
    """

    index: int | None = None
    force: bool = False
    include_active: bool = False
    wait: bool = True
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)


# ==================================================================================================
@dataclass(frozen=True)
class StatusCommand(StudyCommand):
    """Print the state of every run."""


# ==================================================================================================
@dataclass(frozen=True)
class CollectCommand(StudyCommand):
    """Write the cross-run summary table `summary/run_table.parquet`."""


# ==================================================================================================
@dataclass(frozen=True)
class ReportCommand(StudyCommand):
    """Render the plots of finished runs.

    Attributes:
        index (int | None): Only this run; all finished runs if `None`.
    """

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
            study = CreatedStudy.create(command.module, command.root)
            print(f"Created {study.directory.path} with {len(study.runs)} runs.")
        case ShowCommand():
            # Print an overview of a study module or a single run config.
            if command.path.suffix == ".py":
                print(Study.load_from_module_file(command.path).describe())
            else:
                print(format_config_tree(json.loads(command.path.read_text())))
        case RunCommand():
            # Execute unfinished runs (blocking), or queue them on SLURM and return (`--no-wait`).
            indices = None if command.index is None else [command.index]
            outcomes = CreatedStudy.load(command.module, command.root).execute_runs(
                command.executor, indices, command.force, command.include_active, command.wait
            )
            print(dict(Counter(str(outcome) for outcome in outcomes.values())))
            if RunOutcome.ACTIVE in outcomes.values():
                print("Skipped runs that are submitted or running; --include-active restarts them.")
            if RunOutcome.FAILED in outcomes.values():
                sys.exit(1)
        case StatusCommand():
            # Print the state (pending/submitted/running/done/failed) of every run.
            study = CreatedStudy.load(command.module, command.root)
            states = study.read_states()
            for run in study.runs:
                print(f"{run.index:>4}  {run.run_id}  {states[run.index].value}")
            print(dict(Counter(state.value for state in states.values())))
        case CollectCommand():
            # Write the cross-run table.
            study = CreatedStudy.load(command.module, command.root)
            print(f"Wrote {study.write_run_table()}")
            print(study.build_run_table().to_string(index=False))
        case ReportCommand():
            # Plot the finished runs; unfinished ones are skipped.
            CreatedStudy.load(command.module, command.root).plot_finished_runs(command.index)


if __name__ == "__main__":
    main(tyro.cli(Command))

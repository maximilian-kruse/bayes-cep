"""Run one configuration into a run directory.

Choose the run type as a subcommand: a run defined in `single_runs/` by its `name` (`prior`,
`preprocessing`, `map`, `mcmc`; every option given as a flag), a run of a file given with
`--run-file` (see below), or one of the reference setups of
`single_runs.reference`: `reference-synthetic-preprocessing`, `reference-patient-preprocessing`,
`reference-map` and `reference-mcmc`, with the data directories (and the initial state file of the
chain) as arguments (`--config.raw-dir`, ...); no path is a default.
Everything is written to `--run-dir`: `config.json`, `metadata.json`, `status.json`, `run.log`,
`metrics.json` and the outputs of the run.

The example data is the output of three runs, each in its own run directory, with the directories
given in the pixi tasks (flat, with `--flat-results`); they are made in sequence by
`pixi run example`, or one by one:

    pixi run example-preprocessing
    pixi run example-map
    pixi run example-mcmc

To run a configuration recorded in a `config.json`, e.g. of one run of a study (edit the file to
change a value; the run then gets a new run id):

    bcep-single --run-dir working_data/debug config:from-file path/to/config.json

Another run type is registered explicitly by the Python file that defines it (a `Run` subclass
with a `name`, see `bayes_cep.run.registry`); `--run-file` may be repeated and must come before
the subcommand:

    bcep-single --run-file path/to/my_run.py --run-dir working_data/mine config:my-run ...

Add `--executor.cluster slurm` (and `--no-wait` to return after queueing) to run on SLURM; the
resource options are listed by `--help`, as for `study run`.

A single prior run, with every option listed by `--help`:

    bcep-single --run-dir working_data/test config:prior --config.raw-dir example_data/raw \
        --config.correlation.max-distance 40
"""

import argparse
import functools
import json
import operator
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

import single_runs
from bayes_cep.run.config import TYPE_KEY, RunConfig
from bayes_cep.run.directories import STATUS_RECORD
from bayes_cep.run.executor import Executor, ExecutorSettings, RunOutcome
from bayes_cep.run.provenance import Environment
from bayes_cep.run.registry import RunRegistry
from single_runs.reference import (
    reference_map_config,
    reference_mcmc_config,
    reference_patient_preprocessing_config,
    reference_synthetic_preprocessing_config,
)

# The subcommands that build a configuration from a file or a function; no run may take these names.
CONFIG_FILE_SUBCOMMAND = "from-file"
REFERENCE_SUBCOMMANDS = {
    "reference-synthetic-preprocessing": reference_synthetic_preprocessing_config,
    "reference-patient-preprocessing": reference_patient_preprocessing_config,
    "reference-map": reference_map_config,
    "reference-mcmc": reference_mcmc_config,
}


# ==================================================================================================
def build_registry(run_files: list[Path]) -> RunRegistry:
    """Register the runs of `single_runs/` and of the given files.

    Args:
        run_files (list[Path]): Python files defining further runs.

    Returns:
        RunRegistry: The registry.
    """
    registry = RunRegistry(reserved_names=[CONFIG_FILE_SUBCOMMAND, *REFERENCE_SUBCOMMANDS])
    registry.register_run_types_in_package(single_runs)
    for run_file in run_files:
        registry.register_run_types_in_file(run_file)
    return registry


# --------------------------------------------------------------------------------------------------
def build_config_type(registry: RunRegistry) -> object:
    """The type of the `config` argument: a union of one subcommand per run in the registry, the
    configuration file and the reference setups.

    A config class takes all its fields as flags; a function builds the config from its arguments.
    """

    def load_config_file(path: tyro.conf.Positional[Path]) -> RunConfig:
        """Run the configuration recorded in a `config.json`, e.g. of a run of a study.

        Args:
            path (Path): The `config.json`; its `__type__` entry tells which run type it is.

        Raises:
            SystemExit: If the file is not the configuration of one of the registered runs.
        """
        data = json.loads(path.read_text())
        config_types = registry.config_types_by_name
        config_type = config_types.get(data.get(TYPE_KEY) if isinstance(data, dict) else None)
        if config_type is None:
            raise SystemExit(f"{path} is not a run configuration; known: {sorted(config_types)}")
        return config_type.from_json_dict(data)

    subcommands = [
        *(
            Annotated[run_type.config_type, tyro.conf.subcommand(run_type.name)]
            for run_type in registry.run_types
        ),
        Annotated[
            RunConfig, tyro.conf.subcommand(CONFIG_FILE_SUBCOMMAND, constructor=load_config_file)
        ],
        *(
            Annotated[RunConfig, tyro.conf.subcommand(name, constructor=constructor)]
            for name, constructor in REFERENCE_SUBCOMMANDS.items()
        ),
    ]
    return functools.reduce(operator.or_, subcommands)


# --------------------------------------------------------------------------------------------------
def build_settings_type(registry: RunRegistry) -> type:
    """The settings class of `bcep-single`, whose `config` is chosen among the registered runs.

    The class is built here, not at import time, because its subcommands are the runs that are
    registered from the command line.
    """
    config_type = build_config_type(registry)

    @dataclass(frozen=True)
    class SingleRunSettings:
        """Settings of `bcep-single`.

        `--run-file PATH` (before the subcommand, may be repeated) registers the runs defined in
        a Python file as further subcommands; it is taken out of the arguments before this
        parsing.

        Attributes:
            run_dir (Path): Directory the run is written to; must not exist yet unless
                `overwrite`.
            config (RunConfig): The run to execute, chosen as a subcommand.
            overwrite (bool): Whether to replace an existing run directory. Only a directory that
                is a run directory (has a status record) is replaced; the executor removes its
                files.
            flat_results (bool): Whether to write the result files directly into the run directory
                instead of its `results/` subdirectory, as the example data does.
            wait (bool): Whether to wait for the run to finish. `--no-wait` queues the run on
                SLURM and returns.
            executor (ExecutorSettings): Where and with which resources to run; `--executor.cluster
                slurm` submits the run as a SLURM job, whose logs go to `<run-dir>-jobs`.
        """

        run_dir: Path
        config: config_type  # type: ignore[valid-type]
        overwrite: bool = False
        flat_results: bool = False
        wait: bool = True
        executor: ExecutorSettings = ExecutorSettings()

    return SingleRunSettings


# ==================================================================================================
def split_run_file_arguments(arguments: list[str]) -> tuple[list[Path], list[str]]:
    """Take the `--run-file` options out of the command line arguments.

    They are needed before the rest can be parsed, because the registered runs are its subcommands.

    Returns:
        tuple[list[Path], list[str]]: The files, and the remaining arguments.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--run-file", action="append", type=Path, default=[])
    known, remaining = parser.parse_known_args(arguments)
    return known.run_file, remaining


# --------------------------------------------------------------------------------------------------
def _check_run_directory_is_replaceable(run_dir: Path, overwrite: bool) -> None:
    """Refuse to start in a directory that holds something else than the run to replace.

    Removing the files of the earlier run is the executor's job; this only decides whether it may
    do that here.

    Raises:
        SystemExit: If there is a previous run and `overwrite` is not set, or if the run directory
            is not empty and not one made by a run.
    """
    if (run_dir / STATUS_RECORD).exists():
        if not overwrite:
            raise SystemExit(f"{run_dir} exists; pass --overwrite to replace it.")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty and not a run directory; not replacing it.")


# ==================================================================================================
def main(arguments: list[str] | None = None) -> None:
    """Parse the command line, register the runs, and execute the run; exit with status 1 if it
    failed.

    Args:
        arguments (list[str] | None): Command line arguments; `sys.argv[1:]` if `None`.
    """
    if arguments is None:
        arguments = sys.argv[1:]
    run_files, remaining_arguments = split_run_file_arguments(arguments)
    registry = build_registry(run_files)
    settings = tyro.cli(build_settings_type(registry), args=remaining_arguments)
    run_dir = settings.run_dir
    _check_run_directory_is_replaceable(run_dir, settings.overwrite)
    run_type = registry.run_type_of(settings.config)
    run = run_type(settings.config, flat_results=settings.flat_results)
    executor = Executor(
        settings.executor, run_dir.parent / f"{run_dir.name}-jobs", job_name=run_dir.name
    )
    environment = Environment.collect_from_current_process()
    (outcome,) = executor.run([run], [run_dir], environment, settings.wait)
    if outcome == RunOutcome.FAILED:
        print(f"Run failed, see {run_dir / STATUS_RECORD}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

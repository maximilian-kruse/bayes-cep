"""Run one configuration into a run directory.

Choose the kind of run as a subcommand: `prior`, `preprocessing`, `map` or `mcmc` (every option
given as a flag), or `reference-preprocessing`, `reference-map` and `reference-mcmc`: the reference
settings of `single_runs.reference`, with the data directories (and the ground truth, and the
initial state file of the chain) as arguments (`--config.raw-dir`, ...); no path is a default.
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

    pixi run single --run-dir working_data/debug config:from-file path/to/config.json

Add `--executor.cluster slurm` (and `--no-wait` to return after queueing) to run on SLURM; the
resource options are listed by `--help`, as for `study run`.

A single prior run, with every option listed by `--help`:

    pixi run single --run-dir working_data/test config:prior --config.raw-dir example_data/raw \
        --config.correlation.max-distance 40
"""

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from bayes_cep.run.config import TYPE_KEY, RunConfig
from bayes_cep.run.directories import STATUS_RECORD
from bayes_cep.run.executor import Executor, ExecutorSettings, RunOutcome
from bayes_cep.run.provenance import Environment
from bayes_cep.run.template import Run
from single_runs.map import MapRun, MapRunConfig
from single_runs.mcmc import McmcRun, McmcRunConfig
from single_runs.preprocessing import PreprocessingRun, PreprocessingRunConfig
from single_runs.prior import PriorRun, PriorRunConfig
from single_runs.reference import (
    reference_map_config,
    reference_mcmc_config,
    reference_preprocessing_config,
)

RUN_TYPES: dict[type[RunConfig], type[Run]] = {
    PriorRunConfig: PriorRun,
    PreprocessingRunConfig: PreprocessingRun,
    MapRunConfig: MapRun,
    McmcRunConfig: McmcRun,
}


# ==================================================================================================
def load_config_file(path: tyro.conf.Positional[Path]) -> RunConfig:
    """Run the configuration recorded in a `config.json`, e.g. of a run of a study.

    Args:
        path (Path): The `config.json`; its `__type__` entry tells which kind of run it is.

    Raises:
        SystemExit: If the file is not the configuration of one of the run kinds.
    """
    data = json.loads(path.read_text())
    config_types = {config_type.__name__: config_type for config_type in RUN_TYPES}
    config_type = config_types.get(data.get(TYPE_KEY) if isinstance(data, dict) else None)
    if config_type is None:
        raise SystemExit(f"{path} is not a run configuration; known: {sorted(config_types)}")
    return config_type.from_json_dict(data)


# The subcommands: a config class takes all its fields as flags; a function builds the config
# from its arguments.
Config = (
    Annotated[PriorRunConfig, tyro.conf.subcommand("prior")]
    | Annotated[PreprocessingRunConfig, tyro.conf.subcommand("preprocessing")]
    | Annotated[MapRunConfig, tyro.conf.subcommand("map")]
    | Annotated[McmcRunConfig, tyro.conf.subcommand("mcmc")]
    | Annotated[RunConfig, tyro.conf.subcommand("from-file", constructor=load_config_file)]
    | Annotated[
        RunConfig,
        tyro.conf.subcommand("reference-preprocessing", constructor=reference_preprocessing_config),
    ]
    | Annotated[RunConfig, tyro.conf.subcommand("reference-map", constructor=reference_map_config)]
    | Annotated[
        RunConfig, tyro.conf.subcommand("reference-mcmc", constructor=reference_mcmc_config)
    ]
)


# ==================================================================================================
@dataclass(frozen=True)
class SingleRun:
    """Settings of `run.py`.

    Attributes:
        run_dir (Path): Directory the run is written to; must not exist yet unless `overwrite`.
        config (RunConfig): The run to execute, chosen as a subcommand.
        overwrite (bool): Whether to delete an existing run directory first. Only a directory that
            is a run directory (has a status record) is deleted.
        flat_results (bool): Whether to write the result files directly into the run directory
            instead of its `results/` subdirectory, as the example data does.
        wait (bool): Whether to wait for the run to finish. `--no-wait` queues the run on SLURM
            and returns.
        executor (ExecutorSettings): Where and with which resources to run; `--executor.cluster
            slurm` submits the run as a SLURM job, whose logs go to `<run-dir>-jobs`.
    """

    run_dir: Path
    config: Config
    overwrite: bool = False
    flat_results: bool = False
    wait: bool = True
    executor: ExecutorSettings = ExecutorSettings()


# ==================================================================================================
def _remove_previous_output(run_dir: Path, overwrite: bool) -> None:
    """Delete what a previous run left in the run directory, if `overwrite` allows it.

    Raises:
        SystemExit: If there is previous output and `overwrite` is not set, or if the run
            directory to delete is not one made by a run.
    """
    if (run_dir / STATUS_RECORD).exists():
        if not overwrite:
            raise SystemExit(f"{run_dir} exists; pass --overwrite to replace it.")
        shutil.rmtree(run_dir)
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty and not a run directory; not deleting it.")


# ==================================================================================================
def main(settings: SingleRun) -> None:
    """Execute the run; exit with status 1 if it failed."""
    _remove_previous_output(settings.run_dir, settings.overwrite)
    run = RUN_TYPES[type(settings.config)](settings.config, flat_results=settings.flat_results)
    run_dir = settings.run_dir
    executor = Executor(
        settings.executor, run_dir.parent / f"{run_dir.name}-jobs", job_name=run_dir.name
    )
    environment = Environment.collect_from_current_process()
    (outcome,) = executor.run([run], [run_dir], environment, settings.wait)
    if outcome == RunOutcome.FAILED:
        print(f"Run failed, see {run_dir / STATUS_RECORD}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(SingleRun))

"""Run one configuration into a run directory.

Choose the kind of run as a subcommand: `prior`, `preprocessing`, `map` or `mcmc`, or one of the
presets that reproduce the example data: `reference-preprocessing-synthetic` and
`reference-preprocessing-real` (synthetic or real-data ground truth; there is deliberately no
default), `reference-map` and `reference-mcmc`. Everything is written to `--run-dir`:
`config.json`, `metadata.json`, `status.json`, `run.log`, `metrics.json` and `results/`.

The MAP and MCMC runs read the preprocessed data of a preprocessing run, given as
`--config.problem.preprocessing-dir`. The example data is the output of three runs, each in its own
run directory; they are made in sequence by `pixi run example`, or one by one:

    pixi run example-preprocessing
    pixi run example-map
    pixi run example-mcmc

A single prior run, with every option listed by `--help`:

    pixi run single --run-dir working_data/test config:prior --config.raw-dir example_data/raw \
        --config.correlation.max-distance 40
"""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro

from bayes_cep.preprocessing.ground_truth import (
    RealDataGroundTruthStrategy,
    SyntheticGroundTruthStrategy,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import STATUS_RECORD, RunState
from bayes_cep.run.template import Run
from single_runs.map import MapRun, MapRunConfig, reference_map_config
from single_runs.mcmc import McmcRun, McmcRunConfig, reference_mcmc_config
from single_runs.preprocessing import (
    PreprocessingRun,
    PreprocessingRunConfig,
    reference_preprocessing_config,
)
from single_runs.prior import PriorRun, PriorRunConfig

EXAMPLE_PREPROCESSING_DIR = Path("example_data/preprocessing")
RUN_TYPES: dict[type[RunConfig], type[Run]] = {
    PriorRunConfig: PriorRun,
    PreprocessingRunConfig: PreprocessingRun,
    MapRunConfig: MapRun,
    McmcRunConfig: McmcRun,
}

Prior = Annotated[PriorRunConfig, tyro.conf.subcommand("prior")]
Preprocessing = Annotated[PreprocessingRunConfig, tyro.conf.subcommand("preprocessing")]
Map = Annotated[MapRunConfig, tyro.conf.subcommand("map")]
Mcmc = Annotated[McmcRunConfig, tyro.conf.subcommand("mcmc")]
ReferencePreprocessingSynthetic = Annotated[
    PreprocessingRunConfig,
    tyro.conf.subcommand(
        "reference-preprocessing-synthetic",
        default=reference_preprocessing_config(SyntheticGroundTruthStrategy()),
    ),
]
ReferencePreprocessingReal = Annotated[
    PreprocessingRunConfig,
    tyro.conf.subcommand(
        "reference-preprocessing-real",
        default=reference_preprocessing_config(RealDataGroundTruthStrategy()),
    ),
]
ReferenceMap = Annotated[
    MapRunConfig,
    tyro.conf.subcommand("reference-map", default=reference_map_config(EXAMPLE_PREPROCESSING_DIR)),
]
ReferenceMcmc = Annotated[
    McmcRunConfig,
    tyro.conf.subcommand(
        "reference-mcmc", default=reference_mcmc_config(EXAMPLE_PREPROCESSING_DIR)
    ),
]


# ==================================================================================================
@dataclass(frozen=True)
class SingleRun:
    """Settings of `run.py`.

    Attributes:
        run_dir (Path): Directory the run is written to; must not exist yet unless `overwrite`.
        overwrite (bool): Whether to delete an existing run directory first. Only a directory that
            is a run directory (has a status record) is deleted.
        config (RunConfig): The run to execute, chosen as a subcommand.
    """

    run_dir: Path
    config: (
        Prior
        | Preprocessing
        | Map
        | Mcmc
        | ReferencePreprocessingSynthetic
        | ReferencePreprocessingReal
        | ReferenceMap
        | ReferenceMcmc
    )
    overwrite: bool = False


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
    run = RUN_TYPES[type(settings.config)](settings.config)
    if run.execute(settings.run_dir) == RunState.FAILED:
        print(f"Run failed, see {settings.run_dir / STATUS_RECORD}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(SingleRun))

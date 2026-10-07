"""Run one configuration (prior investigation or MAP estimation) into a run directory.

Choose the kind of run as a subcommand: `prior`, `map`, or one of the presets that reproduce the
example data, `example-synthetic` and `example-real` (synthetic or real-data ground truth; there is
deliberately no default). Everything is written to `--run-dir`: `config.json`, `metadata.json`,
`status.json`, `run.log`, `metrics.json` and `results/`.

With `--example-layout`, a run writes the plain example-data layout instead, without any JSON
records: `preprocessing/` (ground truth, prior mean, observations), `map/` (MAP estimate and
histories), `mcmc/` (chain) and `logs/`. Regenerate the example data like this (the raw data path
of the preset is relative to the repository root), all at once or stage by stage; a stage reads the
data of the earlier ones from `example_data`:

    pixi run example config:example-synthetic
    pixi run example-preprocessing config:example-synthetic
    pixi run example-map config:example-synthetic
    pixi run example-mcmc config:example-synthetic

A single prior run, with every option listed by `--help`:

    pixi run single --run-dir working_data/test config:prior --config.raw-dir example_data/raw \
        --config.correlation.max-distance 40
"""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro

from bayes_cep.preprocessing.ground_truth import (
    RealDataGroundTruthStrategy,
    SyntheticGroundTruthStrategy,
)
from bayes_cep.run.directories import FAILED
from bayes_cep.run.template import Run
from single_runs.config import MapRunConfig, PriorRunConfig, reference_config
from single_runs.map import MapRun
from single_runs.prior import PriorRun

STAGE_FOLDERS = {
    "all": ("preprocessing", "map", "mcmc"),
    "preprocessing": ("preprocessing",),
    "map": ("map",),
    "mcmc": ("mcmc",),
}
RUN_TYPES: dict[type, type[Run]] = {PriorRunConfig: PriorRun, MapRunConfig: MapRun}

Prior = Annotated[PriorRunConfig, tyro.conf.subcommand("prior")]
Map = Annotated[MapRunConfig, tyro.conf.subcommand("map")]
ExampleSynthetic = Annotated[
    MapRunConfig,
    tyro.conf.subcommand(
        "example-synthetic",
        default=reference_config(SyntheticGroundTruthStrategy(), with_mcmc=True),
    ),
]
ExampleReal = Annotated[
    MapRunConfig,
    tyro.conf.subcommand(
        "example-real", default=reference_config(RealDataGroundTruthStrategy(), with_mcmc=True)
    ),
]


# ==================================================================================================
@dataclass(frozen=True)
class SingleRun:
    """Settings of `run.py`.

    Attributes:
        run_dir (Path): Directory the run is written to; must not exist yet unless `overwrite`.
        overwrite (bool): Whether to delete an existing run directory first (with the example
            layout, only the folders of the stage).
        example_layout (bool): Whether to write the plain example-data layout, without JSON records.
        stage (Literal["all", "preprocessing", "map", "mcmc"]): Which part of a MAP run to perform
            (example layout only); later stages read the data of earlier ones from `run_dir`.
        config (PriorRunConfig | MapRunConfig): The run to execute, chosen as a subcommand.
    """

    run_dir: Path
    config: Prior | Map | ExampleSynthetic | ExampleReal
    overwrite: bool = False
    example_layout: bool = False
    stage: Literal["all", "preprocessing", "map", "mcmc"] = "all"


# ==================================================================================================
def main(settings: SingleRun) -> None:
    """Execute the run; exit with status 1 if it failed."""
    run_dir = settings.run_dir
    stale = (
        [run_dir / name for name in STAGE_FOLDERS[settings.stage] if (run_dir / name).exists()]
        if settings.example_layout
        else [run_dir] * run_dir.exists()
    )
    if stale and not settings.overwrite:
        raise SystemExit(f"{stale[0]} exists; pass --overwrite to replace it.")
    for path in stale:
        shutil.rmtree(path)
    config = settings.config
    run = RUN_TYPES[type(config)](
        config, example_layout=settings.example_layout, stage=settings.stage
    )
    if run.execute(run_dir) == FAILED:
        record = run_dir / (
            f"logs/{settings.stage}.log" if settings.example_layout else "status.json"
        )
        print(f"Run failed, see {record}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(SingleRun))

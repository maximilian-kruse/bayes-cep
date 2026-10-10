"""Generation of the example data: the plain layout of `example_data/`, without run records.

The example data is a fixed reference, not a run of a study, so it has no `config.json`, status or
metrics. It is written by performing the stages of a
[`MapRun`][single_runs.map.MapRun] directly, one folder per stage: `preprocessing/` (ground truth,
prior mean, observations), `map/` (MAP estimate and histories), `mcmc/` (chain) and `logs/`. A later
stage reads the data of the earlier ones from the same directory.

Constants:
    STAGE_FOLDERS: The folders of the example layout that each stage writes.

Functions:
    example_data_paths: Where the stages write in the example layout.
    generate_example_data: Write the example data, all of it or one stage.
"""

from pathlib import Path

from single_runs.config import MapRunConfig
from single_runs.map import MapPaths, MapRun, MapStage

STAGE_FOLDERS = {
    MapStage.ALL: ("preprocessing", "map", "mcmc"),
    MapStage.PREPROCESSING: ("preprocessing",),
    MapStage.MAP: ("map",),
    MapStage.MCMC: ("mcmc",),
}
"""The folders of the example layout that each stage writes."""


# ==================================================================================================
def example_data_paths(example_dir: Path) -> MapPaths:
    """The paths of the example data: one folder per stage, logs in `logs/`."""
    logs = example_dir / "logs"
    return MapPaths(
        data_dir=example_dir / "preprocessing",
        map_dir=example_dir / "map",
        mcmc_dir=example_dir / "mcmc",
        optimizer_log=logs / "optimizer.log",
        sampler_log=logs / "sampler.log",
    )


# --------------------------------------------------------------------------------------------------
def generate_example_data(
    config: MapRunConfig, example_dir: Path, stage: MapStage = MapStage.ALL
) -> None:
    """Write the example data for `config`.

    The data goes to `preprocessing/`, `map/` and `mcmc/` of `example_dir`, the log to
    `logs/<stage>.log` (`logs/run.log` for all stages).

    Args:
        config (MapRunConfig): The MAP run whose stages are performed.
        example_dir (Path): Directory of the example data; created if missing.
        stage (MapStage): The part to generate. Defaults to all parts (the MCMC part only if the
            configuration has MCMC settings).

    Raises:
        ValueError: If the `mcmc` stage is requested without MCMC settings in the configuration.
    """
    run = MapRun(config)
    stages = run.stages_for(stage)
    example_dir.mkdir(parents=True, exist_ok=True)
    log_path = example_dir / "logs" / f"{'run' if stage == MapStage.ALL else stage}.log"
    with run.open_run_logger(log_path) as logger:
        run.run_stages(logger, example_data_paths(example_dir), stages)

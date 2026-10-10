"""Study 2: MAP estimate from a synthetic ground truth.

The MAP estimate is computed for prior parameter pairs (kappa and tau varied together) and for each
preprocessing run of `preprocessing_synthetic` (number of observations and noise variance), all
combinations. Create and run the `preprocessing_synthetic` study first.
"""

from pathlib import Path

from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup, Zip
from single_runs.map import MapRun
from single_runs.reference import reference_map_config
from studies.preprocessing_synthetic import STUDY as PREPROCESSING_STUDY

RAW_DIR = Path("example_data/raw")  # the raw data all runs of this study work on

PREPROCESSED_DATA_DIRS = PREPROCESSING_STUDY.results_directories()

STUDY = StudySetup(
    run_type=MapRun,
    name="map_synthetic",
    description=(
        "MAP estimate for synthetic observations of the standard prior field, over prior "
        "parameter pairs and the preprocessing runs (number of observations, noise variance)."
    ),
    base=reference_map_config(RAW_DIR, PREPROCESSED_DATA_DIRS[0]),
    sweep=Product(
        Zip(
            Axis("problem.prior.kappa", (0.025, 0.05, 0.1)),
            Axis("problem.prior.tau", (5.0, 10.0, 20.0)),
        ),
        Axis("problem.preprocessed_data_dir", tuple(PREPROCESSED_DATA_DIRS)),
    ),
    root=Path("working_data/map_synthetic"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

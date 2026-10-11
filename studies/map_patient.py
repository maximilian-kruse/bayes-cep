"""MAP estimate from the fiber field of the real patient.

Same sweep as `map_synthetic.py`, over the preprocessing runs of `preprocessing_patient`. Create and
run the `preprocessing_patient` study first.
"""

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup, Zip
from single_runs.map import MapRun
from single_runs.reference import reference_map_config
from studies.preprocessing_patient import STUDY as PREPROCESSING_STUDY

# The raw data all runs of this study work on.
RAW_DIR = module_relative_path(__file__, "../example_data/raw")

DATA_DIRS = PREPROCESSING_STUDY.results_directories()

STUDY = StudySetup(
    run_type=MapRun,
    name="map_patient",
    description=(
        "MAP estimate for synthetic observations of the real patient fiber field, over prior "
        "parameter pairs and the preprocessing runs (number of observations, noise variance)."
    ),
    base=reference_map_config(RAW_DIR, DATA_DIRS[0]),
    sweep=Product(
        Zip(
            Axis("problem.prior.kappa", (0.025, 0.05, 0.1)),
            Axis("problem.prior.tau", (5.0, 10.0, 20.0)),
        ),
        Axis("problem.data_dir", tuple(DATA_DIRS)),
    ),
    root=module_relative_path(__file__, "../working_data/map_patient"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

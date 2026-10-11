"""MAP estimate from the fiber field of the real patient.

Same sweep as `map_synthetic.py`, over the preprocessing runs of `preprocessing_patient`. Create and
run the `preprocessing_patient` study first.
"""

from dataclasses import replace

from bayes_cep.optimization.initial_guess import PriorSampleInitialGuess
from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup, Zip
from single_runs.map import MapRun
from single_runs.reference import reference_map_config
from studies.preprocessing_patient import STUDY as PREPROCESSING_STUDY

# The raw data all runs of this study work on.
RAW_DIR = module_relative_path(__file__, "../example_data/raw")

# One optimization per seed, each started at a different prior draw. Not 0, the seed of the
# synthetic ground truth.
INITIAL_GUESS_SEEDS = (1, 2, 3, 4, 5)

DATA_DIRS = PREPROCESSING_STUDY.results_directories()

STUDY = StudySetup(
    run_type=MapRun,
    name="map_patient",
    description=(
        "MAP estimate for synthetic observations of the real patient fiber field, over prior "
        "parameter pairs and the preprocessing runs (number of observations, noise variance), "
        "each with 5 optimizations started at different prior draws."
    ),
    base=replace(
        reference_map_config(RAW_DIR, DATA_DIRS[0]), initial_guess=PriorSampleInitialGuess(seed=0)
    ),
    sweep=Product(
        Zip(
            Axis("problem.prior.kappa", (0.025, 0.05, 0.1)),
            Axis("problem.prior.tau", (5.0, 10.0, 20.0)),
        ),
        Axis("problem.data_dir", tuple(DATA_DIRS)),
        Axis("initial_guess.seed", INITIAL_GUESS_SEEDS),
    ),
    root=module_relative_path(__file__, "../working_data/map_patient"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

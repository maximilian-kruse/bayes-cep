"""MAP estimate for the smoothed patient field over a grid of prior parameters.

The study that finds suitable prior parameters: the MAP estimate is computed for every pair of
kappa and tau (all combinations) and for each preprocessing run of `preprocessing_patient_smoothed`
(number of observations and noise variance). Each run starts at the estimate of the stiff prior for
the same data (`map_patient_smoothed_stiff`), see there for why. Create and run the
`preprocessing_patient_smoothed` and `map_patient_smoothed_stiff` studies first. The summary plots
are written by `scripts/postprocess_map_study.py`.
"""

from dataclasses import replace

from bayes_cep.optimization.initial_guess import FileInitialGuess
from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup, Zip
from single_runs.map import MapResults, MapRun
from single_runs.reference import reference_map_config
from studies.map_patient_smoothed_stiff import STUDY as STIFF_STUDY
from studies.preprocessing_patient_smoothed import RAW_DIR
from studies.preprocessing_patient_smoothed import STUDY as PREPROCESSING_STUDY

# The grid reaches the rough side (large kappa, small tau) of the raw-field fits and the smooth side
# of the smoothed-field fits.
KAPPAS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
TAUS = (0.3, 0.5, 1.0, 1.4, 2.0, 3.0)

DATA_DIRS = PREPROCESSING_STUDY.results_directories()
# The stiff-prior estimate for each data set; both studies sweep the preprocessing runs in order.
STARTING_POINTS = tuple(
    directory / MapResults.FILE_NAMES["estimate"] for directory in STIFF_STUDY.results_directories()
)

STUDY = StudySetup(
    run_type=MapRun,
    name="map_patient_smoothed",
    description=(
        "MAP estimate for synthetic observations of the smoothed patient field, over a grid of "
        "kappa and tau and the preprocessing runs (number of observations, noise variance), "
        "started at the estimate of the stiff prior."
    ),
    base=replace(
        reference_map_config(RAW_DIR, DATA_DIRS[0]),
        initial_guess=FileInitialGuess(path=STARTING_POINTS[0]),
    ),
    sweep=Product(
        Axis("problem.prior.kappa", KAPPAS),
        Axis("problem.prior.tau", TAUS),
        Zip(
            Axis("problem.data_dir", tuple(DATA_DIRS)),
            Axis("initial_guess.path", STARTING_POINTS),
        ),
    ),
    root=module_relative_path(__file__, "../working_data/map_patient_smoothed"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

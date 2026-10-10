"""Study 3: MAP estimate from the fiber field of the real patient.

Same sweep as `map_synthetic.py`, over the preprocessing runs of `preprocessing_patient`. Create and
run the `preprocessing_patient` study first.
"""

from pathlib import Path

from bayes_cep.run.study import Axis, Product, StudySetup, Zip
from single_runs.map import MapRun, reference_map_config
from studies.preprocessing_patient import STUDY as PREPROCESSING_STUDY

STUDY_ROOT = Path("working_data")  # the --root the preprocessing study is created in
PREPROCESSING_DIRS = PREPROCESSING_STUDY.run_directories(STUDY_ROOT)

STUDY = StudySetup(
    run_type=MapRun,
    name="map_patient",
    description=(
        "MAP estimate for synthetic observations of the real patient fiber field, over prior "
        "parameter pairs and the preprocessing runs (number of observations, noise variance)."
    ),
    base=reference_map_config(PREPROCESSING_DIRS[0]),
    sweep=Product(
        Zip(
            Axis("problem.prior.kappa", (0.025, 0.05, 0.1)),
            Axis("problem.prior.tau", (5.0, 10.0, 20.0)),
        ),
        Axis("problem.preprocessing_dir", tuple(PREPROCESSING_DIRS)),
    ),
)

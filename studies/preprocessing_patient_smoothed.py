"""Preprocessing for the smoothed patient studies: synthetic observations of a smoothed field.

The ground truth is the fiber field of patient 01 smoothed at scale 3
(`scripts/smooth_patient_fibers.py`, stored in `patient_data/`), which has none of the narrow steep
strips of the raw field that the SPDE prior cannot represent. Same sweep as
`preprocessing_synthetic.py`. The MAP studies of the smoothed patient field read these runs.
"""

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.preprocessing import PreprocessingRun
from single_runs.reference import reference_smoothed_patient_preprocessing_config

# The raw data all runs of this study work on, and the smoothed angle field derived from it.
RAW_DIR = module_relative_path(__file__, "../patient_data/patient_01")
SMOOTHED_ANGLE_FIELD = module_relative_path(
    __file__, "../patient_data/patient_01/smoothed_scale_3/angle_field.npy"
)

STUDY = StudySetup(
    run_type=PreprocessingRun,
    name="preprocessing_patient_smoothed",
    description=(
        "Synthetic observations of the smoothed (scale 3) fiber field of patient 01, over the "
        "number of observations and the noise variance."
    ),
    base=reference_smoothed_patient_preprocessing_config(RAW_DIR, SMOOTHED_ANGLE_FIELD),
    sweep=Product(
        Axis("observations.num_observations", (100, 1000)),
        Axis("observations.noise_variance", (1e-2, 1e-3)),
    ),
    root=module_relative_path(__file__, "../working_data/preprocessing_patient_smoothed"),
)

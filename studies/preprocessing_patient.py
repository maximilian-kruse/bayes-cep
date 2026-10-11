"""Preprocessing for the patient studies: synthetic observations of the real patient fiber field.

Same sweep as `preprocessing_synthetic.py`, with the ground truth derived from the patient's fiber
field. The MAP and MCMC studies of the patient problem read these runs.
"""

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.preprocessing import PreprocessingRun
from single_runs.reference import reference_patient_preprocessing_config

# The raw data all runs of this study work on.
RAW_DIR = module_relative_path(__file__, "../example_data/raw")

STUDY = StudySetup(
    run_type=PreprocessingRun,
    name="preprocessing_patient",
    description=(
        "Synthetic observations of the real patient fiber field, over the number of observations "
        "and the noise variance."
    ),
    base=reference_patient_preprocessing_config(RAW_DIR),
    sweep=Product(
        Axis("observations.num_observations", (100, 1000)),
        Axis("observations.noise_variance", (1e-2, 1e-3)),
    ),
    root=module_relative_path(__file__, "../working_data/preprocessing_patient"),
)

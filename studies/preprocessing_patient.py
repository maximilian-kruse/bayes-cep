"""Preprocessing for the patient studies: synthetic observations of the real patient fiber field.

Same sweep as `preprocessing_synthetic.py`, with the ground truth derived from the patient's fiber
field. The MAP and MCMC studies of the patient problem read these runs.
"""

from bayes_cep.preprocessing.ground_truth import RealDataGroundTruthStrategy
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.preprocessing import PreprocessingRun, reference_preprocessing_config

STUDY = StudySetup(
    run_type=PreprocessingRun,
    name="preprocessing_patient",
    description=(
        "Synthetic observations of the real patient fiber field, over the number of observations "
        "and the noise variance."
    ),
    base=reference_preprocessing_config(RealDataGroundTruthStrategy()),
    sweep=Product(
        Axis("observations.num_observations", (100, 1000)),
        Axis("observations.noise_variance", (1e-2, 1e-3)),
    ),
)

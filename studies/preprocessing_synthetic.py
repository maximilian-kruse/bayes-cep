"""Preprocessing for the synthetic studies: synthetic observations from the standard prior field.

The ground truth is the standard prior field of the example data (kappa 0.05, tau 10, seed 0).
Synthetic observations are generated from it for the number of observations and the noise variance
(all combinations). The MAP and MCMC studies of the synthetic problem read these runs.
"""

from pathlib import Path

from bayes_cep.preprocessing.ground_truth import SyntheticGroundTruthStrategy
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.preprocessing import PreprocessingRun
from single_runs.reference import reference_preprocessing_config

RAW_DIR = Path("example_data/raw")  # the raw data all runs of this study work on

STUDY = StudySetup(
    run_type=PreprocessingRun,
    name="preprocessing_synthetic",
    description=(
        "Synthetic observations of the standard prior field, over the number of observations "
        "and the noise variance."
    ),
    base=reference_preprocessing_config(RAW_DIR, SyntheticGroundTruthStrategy()),
    sweep=Product(
        Axis("observations.num_observations", (100, 1000)),
        Axis("observations.noise_variance", (1e-2, 1e-3)),
    ),
    root=Path("working_data/preprocessing_synthetic"),
)

"""Study 4: optimizer comparison.

One fixed problem (the reference one: prior kappa 0.05, tau 10, and the preprocessing run of
`preprocessing_synthetic` with 1000 observations and noise variance 1e-3), solved with the
Cameron-Martin L-BFGS and with scipy's Euclidean L-BFGS-B. Create and run the
`preprocessing_synthetic` study first.
"""

from pathlib import Path

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import ScipyLBFGSBSettings

from bayes_cep.optimization.strategies import ScipyLBFGSBStrategy
from bayes_cep.preprocessing.ground_truth import SyntheticGroundTruthStrategy
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.map import MapRun, reference_map_config, reference_optimizer_strategy
from single_runs.preprocessing import reference_preprocessing_config
from studies.preprocessing_synthetic import STUDY as PREPROCESSING_STUDY

STUDY_ROOT = Path("working_data")  # the --root the preprocessing study is created in
REFERENCE_PREPROCESSING_DIR = PREPROCESSING_STUDY.run_directory(
    reference_preprocessing_config(SyntheticGroundTruthStrategy()), STUDY_ROOT
)

STUDY = StudySetup(
    run_type=MapRun,
    name="optimizer_comparison",
    description="MAP estimate of one problem with the custom L-BFGS and with scipy's L-BFGS-B.",
    base=reference_map_config(REFERENCE_PREPROCESSING_DIR),
    sweep=Product(
        Axis(
            "optimizer",
            (
                reference_optimizer_strategy(),
                ScipyLBFGSBStrategy(ScipyLBFGSBSettings(maximum_num_iterations=1000)),
            ),
        )
    ),
)

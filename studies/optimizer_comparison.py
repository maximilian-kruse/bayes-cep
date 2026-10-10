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
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.map import MapRun
from single_runs.reference import (
    reference_map_config,
    reference_optimizer_strategy,
    reference_preprocessing_config,
)
from studies.preprocessing_synthetic import STUDY as PREPROCESSING_STUDY

RAW_DIR = Path("example_data/raw")  # the raw data all runs of this study work on

REFERENCE_PREPROCESSED_DATA_DIR = PREPROCESSING_STUDY.results_directory(
    reference_preprocessing_config(RAW_DIR, SyntheticGroundTruthStrategy())
)

STUDY = StudySetup(
    run_type=MapRun,
    name="optimizer_comparison",
    description="MAP estimate of one problem with the custom L-BFGS and with scipy's L-BFGS-B.",
    base=reference_map_config(RAW_DIR, REFERENCE_PREPROCESSED_DATA_DIR),
    sweep=Product(
        Axis(
            "optimizer",
            (
                reference_optimizer_strategy(),
                ScipyLBFGSBStrategy(ScipyLBFGSBSettings(maximum_num_iterations=1000)),
            ),
        )
    ),
    root=Path("working_data/optimizer_comparison"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

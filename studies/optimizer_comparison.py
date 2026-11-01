"""Optimizer comparison.

One fixed problem (the reference one: prior kappa 0.1, tau 5, and the preprocessing run of
`preprocessing_synthetic` with 1000 observations and noise variance 1e-3), solved with the
Cameron-Martin L-BFGS, with scipy's Euclidean L-BFGS-B, with Newton-CG, and with the L-BFGS
followed by Newton-CG of the reference run. Create and run the
`preprocessing_synthetic` study first.
"""

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import ScipyLBFGSBSettings

from bayes_cep.optimization.strategies import ScipyLBFGSBStrategy
from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.map import MapRun
from single_runs.reference import (
    reference_map_config,
    reference_map_stages,
    reference_newton_cg_strategy,
    reference_optimizer_strategy,
    reference_synthetic_preprocessing_config,
)
from studies.preprocessing_synthetic import STUDY as PREPROCESSING_STUDY

# The raw data all runs of this study work on.
RAW_DIR = module_relative_path(__file__, "../example_data/raw")

REFERENCE_DATA_DIR = PREPROCESSING_STUDY.results_directory(
    reference_synthetic_preprocessing_config(RAW_DIR)
)

STUDY = StudySetup(
    run_type=MapRun,
    name="optimizer_comparison",
    description=(
        "MAP estimate of one problem with the custom L-BFGS, scipy's L-BFGS-B, Newton-CG and the "
        "L-BFGS followed by Newton-CG."
    ),
    base=reference_map_config(RAW_DIR, REFERENCE_DATA_DIR),
    sweep=Product(
        Axis(
            "stages",
            (
                (reference_optimizer_strategy(),),
                (ScipyLBFGSBStrategy(ScipyLBFGSBSettings(maximum_num_iterations=1000)),),
                (reference_newton_cg_strategy(),),
                reference_map_stages(),
            ),
        )
    ),
    root=module_relative_path(__file__, "../working_data/optimizer_comparison"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

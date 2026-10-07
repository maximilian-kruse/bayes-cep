"""Study 4: optimizer comparison.

One fixed parameter combination (the reference one: prior kappa 0.05, tau 10, 1000 observations,
noise variance 1e-3, synthetic ground truth), solved with the Cameron-Martin L-BFGS and with scipy's
Euclidean L-BFGS-B.
"""

from ls_bayesian.optimization.algorithms.scipy_lbfgs_b import ScipyLBFGSBSettings

from bayes_cep.optimization.strategies import ScipyLBFGSBStrategy
from bayes_cep.preprocessing.ground_truth import SyntheticGroundTruthStrategy
from bayes_cep.run.study import Axis, Product, Study
from single_runs.config import reference_config, reference_optimizer_strategy
from single_runs.map import MapRun

STUDY = Study(
    run_type=MapRun,
    name="optimizer_comparison",
    description="MAP estimate of one problem with the custom L-BFGS and with scipy's L-BFGS-B.",
    base=reference_config(SyntheticGroundTruthStrategy()),
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

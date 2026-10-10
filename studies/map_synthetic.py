"""Study 2: MAP estimate from a synthetic ground truth.

The ground truth is the standard prior field of the example data (kappa 0.05, tau 10, seed 0).
Synthetic observations are generated from it, and the MAP estimate is computed for prior parameter
pairs (kappa and tau varied together), the number of observations and the noise variance (all
combinations).
"""

from bayes_cep.preprocessing.ground_truth import SyntheticGroundTruthStrategy
from bayes_cep.run.study import Study
from bayes_cep.run.sweep import Axis, Product, Zip
from single_runs.config import reference_config
from single_runs.map import MapRun

STUDY = Study(
    run_type=MapRun,
    name="map_synthetic",
    description=(
        "MAP estimate for synthetic observations of the standard prior field, over prior "
        "parameter pairs, number of observations and noise variance."
    ),
    base=reference_config(SyntheticGroundTruthStrategy()),
    sweep=Product(
        Zip(
            Axis("prior.kappa", (0.025, 0.05, 0.1)),
            Axis("prior.tau", (5.0, 10.0, 20.0)),
        ),
        Axis("observations.num_observations", (100, 1000)),
        Axis("observations.noise_variance", (1e-2, 1e-3)),
    ),
)

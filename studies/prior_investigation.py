"""Study 1: prior investigation.

Samples a zero-mean Bilaplacian SPDE prior for a grid of kappa and tau, and estimates the pointwise
variance and the correlation length from 1000 samples.
"""

from pathlib import Path

from bayes_cep.run.study import Study
from bayes_cep.run.sweep import Axis, Product
from bayes_cep.statistics.correlation_length import CorrelationLengthSettings
from single_runs.config import PriorParameters, PriorRunConfig
from single_runs.prior import PriorRun

STUDY = Study(
    run_type=PriorRun,
    name="prior_investigation",
    description=(
        "Prior samples, pointwise variance and correlation length of the Bilaplacian SPDE prior "
        "as a function of kappa and tau (all combinations)."
    ),
    base=PriorRunConfig(
        raw_dir=Path("example_data/raw"),
        correlation=CorrelationLengthSettings(max_distance=40.0, boundary_margin=10.0),
        prior=PriorParameters(seed=0),
        num_samples=1000,
    ),
    sweep=Product(
        Axis("prior.kappa", (0.025, 0.05, 0.1)),
        Axis("prior.tau", (5.0, 10.0, 20.0)),
    ),
)

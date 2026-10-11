"""Prior investigation.

Samples a zero-mean Bilaplacian SPDE prior for a grid of kappa and tau, and estimates the pointwise
variance and the correlation length from 1000 samples.
"""

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.prior import PriorRun
from single_runs.reference import reference_prior_config

# The raw data all runs of this study work on.
RAW_DIR = module_relative_path(__file__, "../example_data/raw")

STUDY = StudySetup(
    run_type=PriorRun,
    name="prior_investigation",
    description=(
        "Prior samples, pointwise variance and correlation length of the Bilaplacian SPDE prior "
        "as a function of kappa and tau (all combinations)."
    ),
    base=reference_prior_config(RAW_DIR),
    sweep=Product(
        Axis("prior.kappa", (0.025, 0.05, 0.1)),
        Axis("prior.tau", (5.0, 10.0, 20.0)),
    ),
    root=module_relative_path(__file__, "../working_data/prior_investigation"),
)

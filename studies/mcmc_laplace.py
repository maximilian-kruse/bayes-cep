"""Laplace-preconditioned MCMC.

One fixed problem (the reference one, as in `optimizer_comparison`) sampled with pCN with the
Laplace approximation at the MAP estimate as proposal and with MALA preconditioned by its
covariance, each for several step widths and chain seeds. The MAP estimate is the one of the
reference run of the `optimizer_comparison` study, which is also the start of the chain. The
algorithm and the step width vary together, because the admissible step widths differ (pCN needs
less than 1). `mcmc_baseline` has the plain samplers of the same problem for comparison. Create
and run the `preprocessing_synthetic` and `optimizer_comparison` studies first.
"""

from bayes_cep.mcmc.builder import MCMCAlgorithmName
from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup, Zip
from single_runs.map import MapResults
from single_runs.mcmc_laplace import PreconditionedMcmcRun
from single_runs.reference import (
    reference_map_config,
    reference_preconditioned_mcmc_config,
    reference_synthetic_preprocessing_config,
)
from studies.optimizer_comparison import STUDY as MAP_STUDY
from studies.preprocessing_synthetic import STUDY as PREPROCESSING_STUDY

# The raw data all runs of this study work on.
RAW_DIR = module_relative_path(__file__, "../example_data/raw")

DATA_DIR = PREPROCESSING_STUDY.results_directory(reference_synthetic_preprocessing_config(RAW_DIR))
MAP_ESTIMATE_PATH = (
    MAP_STUDY.results_directory(reference_map_config(RAW_DIR, DATA_DIR))
    / MapResults.FILE_NAMES["estimate"]
)

ALGORITHMS = (MCMCAlgorithmName.PCN_PRECONDITIONED,) * 3 + (MCMCAlgorithmName.PMALA,) * 3
STEP_WIDTHS = (0.01, 0.03, 0.1, 0.01, 0.03, 0.1)
SEEDS = (0, 1, 2)

_BASE = reference_preconditioned_mcmc_config(
    RAW_DIR, DATA_DIR, MAP_ESTIMATE_PATH, MCMCAlgorithmName.PCN_PRECONDITIONED
)

STUDY = StudySetup(
    run_type=PreconditionedMcmcRun,
    name="mcmc_laplace",
    description=(
        "Laplace-preconditioned pCN and MALA of the reference problem, started at its MAP "
        "estimate, over step widths and chain seeds."
    ),
    base=_BASE,
    sweep=Product(
        Zip(Axis("sampler.algorithm", ALGORITHMS), Axis("sampler.step_width", STEP_WIDTHS)),
        Axis("seed", SEEDS),
    ),
    root=module_relative_path(__file__, "../working_data/mcmc_laplace"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

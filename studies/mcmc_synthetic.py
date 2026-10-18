"""MCMC (MALA) of the reference synthetic problem, started at its MAP estimate.

One long chain for the problem of the optimizer comparison: the standard prior (kappa 0.1, tau 5),
and the preprocessing run of `preprocessing_synthetic` with 1000 observations and noise variance
1e-3. The chain starts at the MAP estimate of the custom L-BFGS run of `optimizer_comparison`
(which started at the prior mean). Create and run the `preprocessing_synthetic` and
`optimizer_comparison` studies first.

The number of samples is chosen for a limit of 72 hours, with a margin: the sampler needed about
1.1 s per sample on a development machine (3 cores), so 150000 samples take about 46 hours and
leave time for the chain diagnostics. Time a pilot on the cluster before relying on this, and
tune the step width: the reference step width of `single_runs/reference.py` accepted every proposal
of a short test chain, which means it is too small for an efficient chain.
"""

from dataclasses import replace

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup
from single_runs.map import MapResults
from single_runs.mcmc import McmcRun
from single_runs.reference import MCMC_SEED, reference_map_config, reference_mcmc_config
from studies.optimizer_comparison import RAW_DIR, REFERENCE_DATA_DIR
from studies.optimizer_comparison import STUDY as OPTIMIZER_COMPARISON_STUDY

NUM_SAMPLES = 150_000
LOG_INTERVAL = 1000  # a row of the progress table per 1000 samples instead of one per sample

MAP_ESTIMATE = (
    OPTIMIZER_COMPARISON_STUDY.results_directory(reference_map_config(RAW_DIR, REFERENCE_DATA_DIR))
    / MapResults.FILE_NAMES["estimate"]
)

STUDY = StudySetup(
    run_type=McmcRun,
    name="mcmc_synthetic",
    description=(
        "MALA chain of the reference synthetic problem (standard prior, 1000 observations, noise "
        "variance 1e-3), started at the MAP estimate."
    ),
    base=replace(
        reference_mcmc_config(RAW_DIR, REFERENCE_DATA_DIR, MAP_ESTIMATE),
        num_samples=NUM_SAMPLES,
        log_interval=LOG_INTERVAL,
    ),
    # One chain; more chains are further seeds.
    sweep=Product(Axis("seed", (MCMC_SEED,))),
    root=module_relative_path(__file__, "../working_data/mcmc_synthetic"),
    # 72 hours. A short chain used 3 cores and about 1 GB; the diagnostics read the chain in blocks.
    executor=ExecutorSettings(cluster="slurm", time_min=72 * 60, cpus_per_task=4, mem_gb=8),
)

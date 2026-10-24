"""Plain pCN and MALA, the baseline of `mcmc_laplace`.

The same problem, MAP start, chain length and seeds as `mcmc_laplace`, sampled with pCN and MALA
without preconditioner. The step widths are the reference one of MALA and values around it, and
the pCN values of the ls-bayesian tutorials. Create and run the `preprocessing_synthetic` and
`optimizer_comparison` studies first.
"""

from bayes_cep.mcmc.builder import MCMCAlgorithmName
from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup, Zip
from single_runs.mcmc import McmcRun
from single_runs.reference import reference_mcmc_config
from studies.mcmc_laplace import DATA_DIR, MAP_ESTIMATE_PATH, RAW_DIR, SEEDS

ALGORITHMS = (MCMCAlgorithmName.PCN,) * 3 + (MCMCAlgorithmName.MALA,) * 3
STEP_WIDTHS = (0.01, 0.05, 0.1, 1e-8, 1e-7, 1e-6)

STUDY = StudySetup(
    run_type=McmcRun,
    name="mcmc_baseline",
    description=(
        "Plain pCN and MALA of the reference problem, started at its MAP estimate, over step "
        "widths and chain seeds; the baseline of the Laplace-preconditioned samplers."
    ),
    base=reference_mcmc_config(RAW_DIR, DATA_DIR, MAP_ESTIMATE_PATH),
    sweep=Product(
        Zip(Axis("sampler.algorithm", ALGORITHMS), Axis("sampler.step_width", STEP_WIDTHS)),
        Axis("seed", SEEDS),
    ),
    root=module_relative_path(__file__, "../working_data/mcmc_baseline"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

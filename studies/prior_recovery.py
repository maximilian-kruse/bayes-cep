"""Recovery test of the prior parameter estimation (simulated moments).

Prior samples with known kappa and tau are fitted on the grid of `prior_moment_fit.py` for two
patient meshes, without smoothing and smoothed at scale 3, for 3 true values of each parameter and
3 samples (seeds) of each. The truths are grid points, so that the error is not mixed with the
grid spacing, and lie in the range the patient fits found. The samples are draws from the model the
fit assumes: this is the easy case, and bounds what the patient fields can show. The summary
(`scripts/postprocess_prior_recovery_study.py`) shows the estimates against the truths, and the
width of the valley.

Needs only the meshes of `patient_data`; no smoothed field is read.
"""

from dataclasses import replace

from bayes_cep.posterior.prior import PriorParameters
from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup
from bayes_cep.statistics.prior_moment_fit import DistanceMetric
from single_runs.prior_recovery import PriorRecoveryRun
from single_runs.reference import reference_prior_recovery_config
from studies.prior_moment_fit import KAPPAS, PATIENT_DIRS, TAUS

NUM_PATIENTS = 2
SMOOTHING_SCALES = (0.0, 3.0)
# Indices into the grid, for 0.0627, 0.131, 0.274 and 0.298, 0.888, 2.01
TRUE_KAPPAS = tuple(KAPPAS[index] for index in (3, 6, 9))
TRUE_TAUS = tuple(TAUS[index] for index in (4, 8, 11))
# Different from the seed of the fit, which is 0
TRUE_SEEDS = (1001, 1002, 1003)

STUDY = StudySetup(
    run_type=PriorRecoveryRun,
    name="prior_recovery",
    description=(
        "Recovery of known kappa and tau from prior samples on the mesh of 2 patients, without "
        "smoothing and smoothed at scale 3, for 3 values of each parameter and 3 samples each "
        "(geodesic distance)."
    ),
    base=replace(
        reference_prior_recovery_config(
            PATIENT_DIRS[0],
            PriorParameters(kappa=TRUE_KAPPAS[0], tau=TRUE_TAUS[0], seed=TRUE_SEEDS[0]),
            SMOOTHING_SCALES[0],
            DistanceMetric.GEODESIC,
        ),
        kappas=KAPPAS,
        taus=TAUS,
    ),
    sweep=Product(
        Axis("raw_dir", PATIENT_DIRS[:NUM_PATIENTS]),
        Axis("fit.smoothing_scale", SMOOTHING_SCALES),
        Axis("true_prior.kappa", TRUE_KAPPAS),
        Axis("true_prior.tau", TRUE_TAUS),
        Axis("true_prior.seed", TRUE_SEEDS),
    ),
    root=module_relative_path(__file__, "../working_data/prior_recovery"),
    executor=ExecutorSettings(
        cluster="slurm", time_min=45, cpus_per_task=1, mem_gb=4, max_parallel=500
    ),
)

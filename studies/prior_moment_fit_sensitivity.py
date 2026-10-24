"""Sensitivity of the prior parameter loss to the simulation settings of the moment fit.

The same kappa/tau grid as `prior_moment_fit.py`, for the geodesic distance, the first three
patients and the raw and the scale-3 smoothed field, with the settings of the simulation varied:

- `fit.num_prior_samples`: prior samples per grid point, i.e. the Monte Carlo noise of the simulated
  statistics,
- `fit.seed`: seed of the simplex pairs and of the prior samples,
- `fit.num_source_simplices`: how many sources the pairs are drawn from (with the same number of
  pairs, so fewer partners each).

Two settings are not swept because they do not change the simulation: the loss is a function of the
stored statistics, so `fit.concentration_weight` and `fit.num_fitted_bins` can be varied afterwards
on the saved loss surfaces of any run (`correlation`, `concentration` and the observed statistics
in `moment_fit.npz`), without new runs.

The smoothed fields must exist as for `prior_moment_fit.py`. The runs differ from the reference
only in the swept settings, so the runs with 8 samples, seed 0 and 400 sources reproduce the
geodesic runs of `prior_moment_fit.py` for these patients.
"""

from dataclasses import replace

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup
from bayes_cep.statistics.prior_moment_fit import DistanceMetric
from single_runs.prior_fit import PriorMomentFitRun
from single_runs.reference import reference_prior_moment_fit_config
from studies.prior_moment_fit import KAPPAS, PATIENT_DIRS, TAUS

NUM_PATIENTS = 3
SMOOTHING_SCALES = (0.0, 3.0)
NUM_PRIOR_SAMPLES = (8, 32)
SEEDS = (0, 1, 2)
NUM_SOURCE_SIMPLICES = (400, 2000)

STUDY = StudySetup(
    run_type=PriorMomentFitRun,
    name="prior_moment_fit_sensitivity",
    description=(
        "Simulated-moment loss over the kappa/tau grid for 3 patients, the raw and the scale-3 "
        "smoothed field, over the number of prior samples, the seed and the number of source "
        "simplices of the pairs (geodesic distance)."
    ),
    base=replace(
        reference_prior_moment_fit_config(
            PATIENT_DIRS[0], SMOOTHING_SCALES[0], DistanceMetric.GEODESIC
        ),
        kappas=KAPPAS,
        taus=TAUS,
    ),
    sweep=Product(
        Axis("raw_dir", PATIENT_DIRS[:NUM_PATIENTS]),
        Axis("fit.smoothing_scale", SMOOTHING_SCALES),
        Axis("fit.num_prior_samples", NUM_PRIOR_SAMPLES),
        Axis("fit.seed", SEEDS),
        Axis("fit.num_source_simplices", NUM_SOURCE_SIMPLICES),
    ),
    root=module_relative_path(__file__, "../working_data/prior_moment_fit_sensitivity"),
    # 32 samples of a smoothed field take about 40 minutes for the 256 grid points
    executor=ExecutorSettings(
        cluster="slurm", time_min=90, cpus_per_task=1, mem_gb=4, max_parallel=500
    ),
)

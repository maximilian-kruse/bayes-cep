"""Loss of the prior parameters kappa and tau against the fiber fields of the patients.

Scores a fine kappa/tau grid for every patient, for the raw field and the fields smoothed at several
scales, with the Euclidean and the geodesic distance of the simplex pairs. Each run keeps the whole
loss surface. The summary (`scripts/postprocess_prior_fit_study.py`) adds the losses of the patients
to the loss map of the ensemble and shows the valley.

The smoothed fields must exist before the study is created:
`scripts/smooth_patient_fibers.py --scale <scale> --mesh-dir patient_data/<patient>
patient_data/<patient>/fiber_field.npy` for every patient and every scale of `SMOOTHING_SCALES`.
"""

from dataclasses import replace

import numpy as np

from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, Product, StudySetup
from bayes_cep.statistics.prior_moment_fit import DistanceMetric
from single_runs.prior_fit import PriorMomentFitRun
from single_runs.reference import reference_prior_moment_fit_config

NUM_PATIENTS = 7
PATIENT_DIRS = tuple(
    module_relative_path(__file__, f"../patient_data/patient_{index:02d}")
    for index in range(1, NUM_PATIENTS + 1)
)

# 0 is the raw field. The grids are logarithmic and reach well beyond the reference fits on both
# sides, so that the valley is visible over its whole length.
SMOOTHING_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
KAPPAS = tuple(float(f"{value:.3g}") for value in np.geomspace(0.03, 1.2, 16))
TAUS = tuple(float(f"{value:.3g}") for value in np.geomspace(0.1, 6.0, 16))

STUDY = StudySetup(
    run_type=PriorMomentFitRun,
    name="prior_moment_fit",
    description=(
        "Simulated-moment loss over a kappa/tau grid for the fiber fields of 7 patients, raw and "
        "smoothed at scales 0.5 to 3, with the Euclidean and the geodesic distance of the pairs."
    ),
    base=replace(
        reference_prior_moment_fit_config(
            PATIENT_DIRS[0], SMOOTHING_SCALES[0], DistanceMetric.GEODESIC
        ),
        kappas=KAPPAS,
        taus=TAUS,
    ),
    sweep=Product(
        Axis("raw_dir", PATIENT_DIRS),
        Axis("fit.smoothing_scale", SMOOTHING_SCALES),
        Axis("fit.distance_metric", tuple(DistanceMetric)),
    ),
    root=module_relative_path(__file__, "../working_data/prior_moment_fit"),
    executor=ExecutorSettings(
        cluster="slurm", time_min=45, cpus_per_task=1, mem_gb=4, max_parallel=500
    ),
)

"""First stage of the MAP study of the smoothed patient field: a stiff prior.

One MAP estimate per preprocessing run of `preprocessing_patient_smoothed`, with the stiff prior
(kappa 0.2, tau 2.0) started at the prior mean. Its estimates are the starting points of
`map_patient_smoothed`: starting the optimization of a weaker prior from them keeps it from ending
in a local minimum with narrow steep strips, where the angle winds by about pi (a branch mismatch
that the optimizer cannot undo). Create and run the `preprocessing_patient_smoothed` study first.
"""

from dataclasses import replace

from bayes_cep.posterior.prior import PriorParameters
from bayes_cep.run.directories import module_relative_path
from bayes_cep.run.executor import ExecutorSettings
from bayes_cep.run.study import Axis, StudySetup
from single_runs.map import MapRun
from single_runs.reference import PRIOR_SEED, reference_map_config
from studies.preprocessing_patient_smoothed import RAW_DIR
from studies.preprocessing_patient_smoothed import STUDY as PREPROCESSING_STUDY

STIFF_PRIOR = PriorParameters(kappa=0.2, tau=2.0, seed=PRIOR_SEED)

DATA_DIRS = PREPROCESSING_STUDY.results_directories()

_REFERENCE_CONFIG = reference_map_config(RAW_DIR, DATA_DIRS[0])

STUDY = StudySetup(
    run_type=MapRun,
    name="map_patient_smoothed_stiff",
    description=(
        "MAP estimate for synthetic observations of the smoothed patient field with the stiff "
        "prior (kappa 0.2, tau 2.0), started at the prior mean; starting points of "
        "map_patient_smoothed."
    ),
    base=replace(_REFERENCE_CONFIG, problem=replace(_REFERENCE_CONFIG.problem, prior=STIFF_PRIOR)),
    sweep=Axis("problem.data_dir", tuple(DATA_DIRS)),
    root=module_relative_path(__file__, "../working_data/map_patient_smoothed_stiff"),
    executor=ExecutorSettings(cluster="slurm", time_min=240, cpus_per_task=4),
)

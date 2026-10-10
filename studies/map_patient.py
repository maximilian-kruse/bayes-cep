"""Study 3: MAP estimate from the fiber field of the real patient.

Same sweep as `map_synthetic.py`, with the ground truth derived from the patient's fiber field.
"""

from bayes_cep.preprocessing.ground_truth import RealDataGroundTruthStrategy
from bayes_cep.run.study import Study
from bayes_cep.run.sweep import Axis, Product, Zip
from single_runs.config import reference_config
from single_runs.map import MapRun

STUDY = Study(
    run_type=MapRun,
    name="map_patient",
    description=(
        "MAP estimate for synthetic observations of the real patient fiber field, over prior "
        "parameter pairs, number of observations and noise variance."
    ),
    base=reference_config(RealDataGroundTruthStrategy()),
    sweep=Product(
        Zip(
            Axis("prior.kappa", (0.025, 0.05, 0.1)),
            Axis("prior.tau", (5.0, 10.0, 20.0)),
        ),
        Axis("observations.num_observations", (100, 1000)),
        Axis("observations.noise_variance", (1e-2, 1e-3)),
    ),
)

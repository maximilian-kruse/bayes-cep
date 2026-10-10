"""The kinds of runs the command line knows.

This is the one place a new run kind is registered (next to its own module in `single_runs/`):
`scripts/run.py` builds its subcommands and its dispatch from it.

Constants:
    RUN_KINDS: The run type of each kind, by the name of its subcommand.
    RUN_TYPE_BY_CONFIG_TYPE: The run type of each configuration type.
"""

from bayes_cep.run.config import RunConfig
from bayes_cep.run.template import Run
from single_runs.map import MapRun
from single_runs.mcmc import McmcRun
from single_runs.preprocessing import PreprocessingRun
from single_runs.prior import PriorRun

RUN_KINDS: dict[str, type[Run]] = {
    "prior": PriorRun,
    "preprocessing": PreprocessingRun,
    "map": MapRun,
    "mcmc": McmcRun,
}

RUN_TYPE_BY_CONFIG_TYPE: dict[type[RunConfig], type[Run]] = {
    run_type.config_type: run_type for run_type in RUN_KINDS.values()
}

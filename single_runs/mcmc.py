r"""MCMC run: sample the posterior of the preprocessed data.

The run reads the data of a preprocessing run (see
[`InferenceProblemConfig`][single_runs.inference.InferenceProblemConfig]) and does nothing but the
sampling. The chain starts at the vertex array in a `.npy` file, e.g. the prior mean of the
preprocessing run or the estimate of a MAP run; the MAP run is not part of this run, only its output
file is read.

Classes:
    McmcRunConfig: Which posterior to sample, and with which sampler.
    McmcRun: The MCMC run.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.mcmc.sampler import SamplerSettings

from bayes_cep.mcmc.builder import (
    MCMCAlgorithmName,
    MCMCBuilder,
    MCMCSettings,
    ZarrStorageSettings,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import resolve_repository_path
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from single_runs.inference import InferenceProblemConfig


# ==================================================================================================
@dataclass(frozen=True)
class McmcRunConfig(RunConfig):
    r"""Which posterior to sample, and with which sampler.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior.
        initial_state_path (Path): `.npy` file with the state the chain starts at, one angle [rad]
            per vertex (e.g. `prior_mean_angle_field.npy` of the preprocessing run, or
            `map_estimate.npy` of a MAP run of the same problem).
        algorithm (MCMCAlgorithmName): Sampling algorithm.
        step_width (float): Proposal step width $\delta$.
        num_samples (int): Number of samples of the chain.
        seed (int): Random seed of the chain.
        chunk_size (int): Length of the on-disk chunk along the sample axis.
        log_interval (int): Number of samples between two rows of the progress table.
    """

    problem: InferenceProblemConfig
    initial_state_path: Path
    algorithm: MCMCAlgorithmName
    step_width: float
    num_samples: int
    seed: int = 0
    chunk_size: int = 100
    log_interval: int = 1


# ==================================================================================================
class McmcRun(Run[McmcRunConfig]):
    """Sample the posterior of the preprocessed data, from the configured initial state."""

    outputs = {
        "results/samples.zarr": "MCMC chain (zarr store, one sample per row).",
        "metrics.json": "acceptance_rate (final) and seconds_sampling.",
        "sampler.log": "Progress table of the sampler.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return [
            *self.config.problem.input_files(),
            resolve_repository_path(self.config.initial_state_path),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """Nothing is plotted: MCMC diagnostics are analyzed separately."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = config.problem.assemble_posterior(steps, logger)
        initial_state_path = resolve_repository_path(config.initial_state_path)
        initial_state = np.load(initial_state_path)
        if initial_state.shape != assembled.data.prior_mean.shape:
            raise ValueError(
                f"The initial state in {initial_state_path} has shape {initial_state.shape}, but "
                f"the problem has {assembled.data.prior_mean.shape}."
            )
        results_dir = self.results_dir(run_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        mcmc_builder = MCMCBuilder(
            MCMCSettings(
                algorithm=config.algorithm,
                step_width=config.step_width,
                storage=ZarrStorageSettings(chunk_size=config.chunk_size),
            ),
            assembled.log_posterior,
            assembled.posterior_builder.prior,
            output_dir=results_dir,
        )
        sampler_logger_settings = LoggerSettings(
            print_to_console=self.write_to_console, logfile_path=run_dir / "sampler.log"
        )
        with BaseLogger(sampler_logger_settings, prefix="mcmc") as sampler_logger:
            with steps.step(f"Building the sampler ({config.algorithm.name})"):
                sampler = mcmc_builder.build(logger=sampler_logger)
            sampling_start = time.perf_counter()
            with steps.step(f"Sampling {config.num_samples} states"):
                sampler.run(
                    initial_state=initial_state,
                    settings=SamplerSettings(
                        num_samples=config.num_samples, log_interval=config.log_interval
                    ),
                    seed=config.seed,
                )
            sampling_seconds = time.perf_counter() - sampling_start
        acceptance_rate = float(mcmc_builder.outputs[0].all_values[-1])
        logger.info(f"      final acceptance rate: {acceptance_rate:.3f}")
        return {"acceptance_rate": acceptance_rate, "seconds_sampling": sampling_seconds}

r"""MCMC run: sample the posterior of the preprocessed data.

The run reads the data of a preprocessing run (see
[`InferenceProblemConfig`][single_runs.inference.InferenceProblemConfig]) and does nothing but the
sampling, started at the prior mean. It has no relation to a MAP run.

Constants:
    REFERENCE_*: Settings of the reference MCMC run, used as defaults.

Classes:
    McmcRunConfig: Which posterior to sample, and with which sampler.
    McmcRun: The MCMC run.

Functions:
    reference_mcmc_config: The MCMC run configuration of the example data.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import override

from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.mcmc.sampler import SamplerSettings

from bayes_cep.mcmc.builder import (
    MCMCAlgorithmName,
    MCMCBuilder,
    MCMCSettings,
    ZarrStorageSettings,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from single_runs.inference import InferenceProblemConfig, reference_inference_problem

REFERENCE_MCMC_STEP_WIDTH = 1e-7
REFERENCE_MCMC_NUM_SAMPLES = 1000


# ==================================================================================================
@dataclass(frozen=True)
class McmcRunConfig(RunConfig):
    r"""Which posterior to sample, and with which sampler.

    Attributes:
        problem (InferenceProblemConfig): The data and the model of the posterior.
        algorithm (MCMCAlgorithmName): Sampling algorithm. Defaults to MALA.
        step_width (float): Proposal step width $\delta$.
        num_samples (int): Number of samples of the chain.
        seed (int): Random seed of the chain.
        chunk_size (int): Length of the on-disk chunk along the sample axis.
        log_interval (int): Number of samples between two rows of the progress table.
    """

    problem: InferenceProblemConfig
    algorithm: MCMCAlgorithmName = MCMCAlgorithmName.MALA
    step_width: float = REFERENCE_MCMC_STEP_WIDTH
    num_samples: int = REFERENCE_MCMC_NUM_SAMPLES
    seed: int = 0
    chunk_size: int = 100
    log_interval: int = 1


# ==================================================================================================
def reference_mcmc_config(preprocessing_dir: Path) -> McmcRunConfig:
    """The MCMC run configuration of the example data.

    Args:
        preprocessing_dir (Path): Run directory of the preprocessing run that produced the data;
            there is deliberately no default.

    Returns:
        McmcRunConfig: The reference configuration for the given data.
    """
    return McmcRunConfig(problem=reference_inference_problem(preprocessing_dir))


# ==================================================================================================
class McmcRun(Run[McmcRunConfig]):
    """Sample the posterior of the preprocessed data, starting at the prior mean."""

    outputs = {
        "results/samples.zarr": "MCMC chain (zarr store, one sample per row).",
        "metrics.json": "acceptance_rate (final) and seconds_sampling.",
        "sampler.log": "Progress table of the sampler.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return self.config.problem.input_files()

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

        results_dir = run_dir / "results"
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
                    initial_state=assembled.data.prior_mean,
                    settings=SamplerSettings(
                        num_samples=config.num_samples, log_interval=config.log_interval
                    ),
                    seed=config.seed,
                )
            sampling_seconds = time.perf_counter() - sampling_start
        acceptance_rate = float(mcmc_builder.outputs[0].all_values[-1])
        logger.info(f"      final acceptance rate: {acceptance_rate:.3f}")
        return {"acceptance_rate": acceptance_rate, "seconds_sampling": sampling_seconds}

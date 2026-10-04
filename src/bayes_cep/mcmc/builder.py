r"""Builds an MCMC `Sampler` for the fiber-angle posterior.

The algorithm is chosen by `MCMCSettings.algorithm`; samples are always stored in a Zarr store.

Classes:
    MCMCAlgorithmName: Selectable MCMC algorithms.
    ZarrStorageSettings: Settings for the Zarr sample storage.
    MCMCSettings: Settings for `MCMCBuilder`.
    MCMCBuilder: Builds a `Sampler` from a posterior, its prior, and `MCMCSettings`.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc import output
from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.algorithms.mala import MALAAlgorithm
from ls_bayesian.mcmc.algorithms.pcn import PCNAlgorithm
from ls_bayesian.mcmc.model import MCMCModel
from ls_bayesian.mcmc.output import MCMCOutput
from ls_bayesian.mcmc.sampler import Sampler
from ls_bayesian.mcmc.storage import ZarrStorage
from ls_bayesian.posterior import interfaces
from ls_bayesian.posterior.posterior import LogPosterior

from bayes_cep.mcmc.measures import LikelihoodTargetMeasure, PriorGaussianMeasure

# Default proposal step width for both samplers. Far smaller than the usual `O(0.1)` choices: the
# data (many observations, tiny noise variance) make the likelihood potential very sharp relative
# to the broad prior, so larger steps are practically never accepted. For pCN on the example
# patient, started at the MAP estimate, short chains gave an acceptance rate of about 90% at `1e-4`
# and about 15% at `1e-3`. The problem is highly nonlinear, so acceptance falls further once the
# chain leaves the MAP's neighborhood; this is a conservative starting point, not a tuned value.
DEFAULT_STEP_WIDTH = 1e-4


# ==================================================================================================
class MCMCAlgorithmName(Enum):
    """Selectable MCMC algorithms.

    Attributes:
        PCN: Classical preconditioned Crank-Nicolson sampler (Cotter et al., 2013), proposing from
            the prior. Needs only the likelihood potential, no gradient.
        MALA: Function-space MALA (pCNL) sampler (Cotter et al., 2013). Uses the likelihood
            gradient, i.e. one adjoint solve per proposal on top of the forward solve.
    """

    PCN = "pcn"
    MALA = "mala"


# ==================================================================================================
@dataclass(frozen=True)
class ZarrStorageSettings:
    """Disk-backed Zarr storage at `<output_dir>/<store_name>`, so long chains need not fit in
    memory and survive interruption (up to the last flushed buffer).

    Attributes:
        chunk_size (int): Length of the on-disk chunk along the sample axis.
        buffer_size (int): Number of samples buffered in memory before writing. The default `1`
            writes every sample immediately; larger values trade durability for fewer writes.
        overwrite (bool): Whether to overwrite an existing store. Defaults to `True`, since
            appending to the store of an earlier run would silently mix two chains.
        store_name (str): Directory name of the Zarr store within the output directory.
    """

    chunk_size: int = 100
    buffer_size: int = 1
    overwrite: bool = True
    store_name: str = "samples.zarr"


# ==================================================================================================
@dataclass(frozen=True)
class MCMCSettings:
    r"""Settings for `MCMCBuilder`.

    Attributes:
        algorithm (MCMCAlgorithmName): Sampling algorithm. Defaults to pCN.
        step_width (float): Proposal step width $\delta$. Smaller values give higher acceptance
            but slower exploration. Must lie in $(0, 1)$ for pCN and be positive for MALA.
        tracked_components (tuple[int, ...]): Indices of state components (mesh vertices) whose
            value and running mean are logged next to the running acceptance rate in every logged
            row. Defaults to `()`, i.e. no single component. Each index must be non-negative, and
            `MCMCBuilder.build` additionally requires it to be smaller than the parameter
            dimension.
        storage (ZarrStorageSettings): Sample storage settings.
    """

    algorithm: MCMCAlgorithmName = MCMCAlgorithmName.PCN
    step_width: float = DEFAULT_STEP_WIDTH
    tracked_components: tuple[int, ...] = ()
    storage: ZarrStorageSettings = field(default_factory=ZarrStorageSettings)

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.tracked_components):
            raise ValueError(
                f"tracked_components must be non-negative, got {self.tracked_components}."
            )
        match self.algorithm:
            case MCMCAlgorithmName.PCN:
                if not 0 < self.step_width < 1:
                    raise ValueError(f"pCN step_width must be in (0, 1), got {self.step_width}.")
            case MCMCAlgorithmName.MALA:
                if self.step_width <= 0:
                    raise ValueError(f"MALA step_width must be positive, got {self.step_width}.")


# ==================================================================================================
class MCMCBuilder:
    """Builds a `Sampler` from a posterior, its prior, and `MCMCSettings`.

    Attributes:
        outputs (tuple[MCMCOutput, ...]): The standard outputs updated by the sampler: the running
            mean acceptance rate, the mean over all state components and its running mean, and,
            for each tracked component, its value and running mean. Set by `build`, so that the
            final values can be read after the run.

    Methods:
        build: Build the algorithm, storage and outputs, then compose a `Sampler`.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: MCMCSettings,
        log_posterior: LogPosterior,
        prior: interfaces.GaussianPrior,
        output_dir: Path,
    ) -> None:
        """Store the settings and the posterior to sample.

        Args:
            settings (MCMCSettings): Algorithm and storage settings.
            log_posterior (LogPosterior): Posterior to sample, e.g. from `PosteriorBuilder.build`.
            prior (interfaces.GaussianPrior): The same prior `log_posterior` was built with, e.g.
                `PosteriorBuilder.prior`.
            output_dir (Path): Directory the sample store is created in.
        """
        self._settings = settings
        self._log_posterior = log_posterior
        self._prior = prior
        self._output_dir = output_dir

    # ----------------------------------------------------------------------------------------------
    def build(self, logger: BaseLogger | None = None) -> Sampler:
        """Build the algorithm, storage and outputs, then compose a `Sampler`.

        Args:
            logger (BaseLogger | None, optional): Logger for the sampler's progress table and the
                algorithm's one-time info messages. Defaults to `None`.

        Returns:
            Sampler: The sampler, writing to a `ZarrStorage` below the output directory.

        Raises:
            ValueError: If a tracked component index is not smaller than the parameter dimension.
        """
        parameter_dimension = self._prior.mean_vector.shape[0]
        for index in self._settings.tracked_components:
            if index >= parameter_dimension:
                raise ValueError(
                    f"Tracked component {index} is out of range for a parameter of dimension "
                    f"{parameter_dimension}."
                )
        model = MCMCModel(
            target=LikelihoodTargetMeasure(self._log_posterior),
            reference=PriorGaussianMeasure(self._prior),
        )
        algorithm: MCMCAlgorithm
        match self._settings.algorithm:
            case MCMCAlgorithmName.PCN:
                algorithm = PCNAlgorithm(model, self._settings.step_width, logger=logger)
            case MCMCAlgorithmName.MALA:
                algorithm = MALAAlgorithm(model, self._settings.step_width, logger=logger)
        storage_settings = self._settings.storage
        storage = ZarrStorage(
            save_directory=self._output_dir / storage_settings.store_name,
            chunk_size=storage_settings.chunk_size,
            buffer_size=storage_settings.buffer_size,
            overwrite=storage_settings.overwrite,
        )
        self.outputs = self._build_outputs()
        return Sampler(algorithm, storage=storage, outputs=self.outputs, logger=logger)

    # ----------------------------------------------------------------------------------------------
    def _build_outputs(self) -> tuple[MCMCOutput, ...]:
        """Build the running acceptance rate, the mean over all state components and its running
        mean, and, per tracked component, its value and running mean."""
        outputs = [
            output.build(output.AcceptanceQoI(), output.RunningMeanStatistic()),
            output.build(output.MeanQoI(), output.IdentityStatistic()),
            output.build(output.MeanQoI(), output.RunningMeanStatistic()),
        ]
        for index in self._settings.tracked_components:
            qoi = output.ComponentQoI(index)
            outputs.append(output.build(qoi, output.IdentityStatistic()))
            outputs.append(output.build(qoi, output.RunningMeanStatistic()))
        return tuple(outputs)

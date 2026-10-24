r"""Builds an MCMC `Sampler` for the fiber-angle posterior.

The algorithm is chosen by `MCMCSettings.algorithm`; samples are always stored in a Zarr store.
The preconditioned algorithms additionally take a Gaussian `approximation`, e.g. the Laplace
approximation at the MAP estimate, as proposal (pCN) or preconditioner (MALA).

Constants:
    SAMPLE_STORE_NAME: Name of the Zarr store of the chain in the output directory.

Classes:
    MCMCAlgorithmName: Selectable MCMC algorithms.
    ZarrStorageSettings: Settings for the Zarr sample storage.
    MCMCSettings: Settings for `MCMCBuilder`.
    MCMCBuilder: Builds a `Sampler` from a posterior and `MCMCSettings`.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc import output
from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.algorithms.mala import MALAAlgorithm
from ls_bayesian.mcmc.algorithms.pcn import PCNAlgorithm
from ls_bayesian.mcmc.algorithms.pmala import PMALAAlgorithm
from ls_bayesian.mcmc.measures import GaussianMeasure
from ls_bayesian.mcmc.model import MCMCModel
from ls_bayesian.mcmc.output import MCMCOutput
from ls_bayesian.mcmc.sampler import Sampler
from ls_bayesian.mcmc.storage import ZarrStorage
from ls_bayesian.posterior.posterior import LogPosterior

from bayes_cep.mcmc.measures import LikelihoodTargetMeasure, PriorGaussianMeasure

# Directory name of the Zarr store within the output directory; fixed, so that the files of a run
# do not depend on its settings.
SAMPLE_STORE_NAME = "samples.zarr"


# ==================================================================================================
class MCMCAlgorithmName(Enum):
    """Selectable MCMC algorithms.

    Attributes:
        PCN: Classical preconditioned Crank-Nicolson sampler (Cotter et al., 2013), proposing from
            the prior. Needs only the likelihood potential, no gradient.
        MALA: Function-space MALA (pCNL) sampler (Cotter et al., 2013). Uses the likelihood
            gradient, i.e. one adjoint solve per proposal on top of the forward solve.
        PCN_PRECONDITIONED: pCN with a Gaussian approximation of the posterior as proposal
            (Pinski et al., 2015). Derivative-free, like `PCN`.
        PMALA: MALA preconditioned by the covariance of a Gaussian approximation of the posterior,
            the state-independent case of the infinite-dimensional manifold MALA (Beskos et al.,
            2017, Algorithm 3.7). It collapses to `MALA` for the prior as approximation. It
            assumes a centred prior.
    """

    PCN = "pcn"
    MALA = "mala"
    PCN_PRECONDITIONED = "pcn-preconditioned"
    PMALA = "pmala"

    @property
    def requires_approximation(self) -> bool:
        """Whether the algorithm needs a Gaussian approximation of the posterior."""
        return self in (MCMCAlgorithmName.PCN_PRECONDITIONED, MCMCAlgorithmName.PMALA)


# ==================================================================================================
@dataclass(frozen=True)
class ZarrStorageSettings:
    """Disk-backed Zarr storage at `<output_dir>/<SAMPLE_STORE_NAME>`, so long chains need not fit
    in memory and survive interruption (up to the last flushed buffer).

    Attributes:
        chunk_size (int): Length of the on-disk chunk along the sample axis.
        buffer_size (int): Number of samples buffered in memory before writing. The default `1`
            writes every sample immediately; larger values trade durability for fewer writes.
        overwrite (bool): Whether to overwrite an existing store. Defaults to `True`, since
            appending to the store of an earlier run would silently mix two chains.
    """

    chunk_size: int = 100
    buffer_size: int = 1
    overwrite: bool = True


# ==================================================================================================
@dataclass(frozen=True)
class MCMCSettings:
    r"""Settings for `MCMCBuilder`.

    There are no defaults for the algorithm and the step width: the reference values are in
    `single_runs.reference`.

    Attributes:
        algorithm (MCMCAlgorithmName): Sampling algorithm.
        step_width (float): Proposal step width $\delta$. Smaller values give higher acceptance
            but slower exploration. Must lie in $(0, 1)$ for the pCN variants and be positive for
            the MALA variants.
        tracked_components (tuple[int, ...]): Indices of state components (mesh vertices) whose
            value and running mean are logged next to the running acceptance rate in every logged
            row. Defaults to `()`, i.e. no single component. Each index must be non-negative, and
            `MCMCBuilder.build` additionally requires it to be smaller than the parameter
            dimension.
        storage (ZarrStorageSettings): Sample storage settings.
    """

    algorithm: MCMCAlgorithmName
    step_width: float
    tracked_components: tuple[int, ...] = ()
    storage: ZarrStorageSettings = field(default_factory=ZarrStorageSettings)

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.tracked_components):
            raise ValueError(
                f"tracked_components must be non-negative, got {self.tracked_components}."
            )
        match self.algorithm:
            case MCMCAlgorithmName.PCN | MCMCAlgorithmName.PCN_PRECONDITIONED:
                if not 0 < self.step_width < 1:
                    raise ValueError(
                        f"{self.algorithm.name} step_width must be in (0, 1), "
                        f"got {self.step_width}."
                    )
            case MCMCAlgorithmName.MALA | MCMCAlgorithmName.PMALA:
                if self.step_width <= 0:
                    raise ValueError(
                        f"{self.algorithm.name} step_width must be positive, got {self.step_width}."
                    )


# ==================================================================================================
class MCMCBuilder:
    """Builds a `Sampler` from a posterior and `MCMCSettings`; the prior is `LogPosterior.prior`.

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
        output_dir: Path,
        approximation: GaussianMeasure | None = None,
    ) -> None:
        """Store the settings and the posterior to sample.

        Args:
            settings (MCMCSettings): Algorithm and storage settings.
            log_posterior (LogPosterior): Posterior to sample, e.g. from `PosteriorBuilder.build`;
                its `prior` is the reference measure.
            output_dir (Path): Directory the sample store is created in.
            approximation (GaussianMeasure | None, optional): Gaussian approximation of the
                posterior, e.g. a `LowRankGaussianMeasure` of the Laplace approximation. Required
                for the preconditioned algorithms and not allowed for the others. Defaults to
                `None`.

        Raises:
            ValueError: If the approximation is missing for a preconditioned algorithm or given for
                another one, or if `PMALA` is used with a prior that is not centred.
        """
        algorithm = settings.algorithm
        if algorithm.requires_approximation and approximation is None:
            raise ValueError(f"The algorithm {algorithm.name} needs an approximation.")
        if not algorithm.requires_approximation and approximation is not None:
            raise ValueError(f"The algorithm {algorithm.name} does not use an approximation.")
        prior_mean_norm = float(np.linalg.norm(log_posterior.prior.mean_vector))
        if algorithm is MCMCAlgorithmName.PMALA and prior_mean_norm != 0.0:
            raise ValueError(
                "PMALA assumes a centred prior (its score is the gradient plus the prior "
                f"precision applied to the state), but the prior mean has norm {prior_mean_norm}."
            )
        self._settings = settings
        self._log_posterior = log_posterior
        self._output_dir = output_dir
        self._approximation = approximation

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
        parameter_dimension = self._log_posterior.prior.mean_vector.shape[0]
        for index in self._settings.tracked_components:
            if index >= parameter_dimension:
                raise ValueError(
                    f"Tracked component {index} is out of range for a parameter of dimension "
                    f"{parameter_dimension}."
                )
        model = MCMCModel(
            target=LikelihoodTargetMeasure(self._log_posterior),
            reference=PriorGaussianMeasure(self._log_posterior.prior),
            approximation=self._approximation,
        )
        algorithm: MCMCAlgorithm
        match self._settings.algorithm:
            case MCMCAlgorithmName.PCN:
                algorithm = PCNAlgorithm(model, self._settings.step_width, logger=logger)
            case MCMCAlgorithmName.MALA:
                algorithm = MALAAlgorithm(model, self._settings.step_width, logger=logger)
            case MCMCAlgorithmName.PCN_PRECONDITIONED:
                algorithm = PCNAlgorithm(model, self._settings.step_width, logger=logger)
            case MCMCAlgorithmName.PMALA:
                algorithm = PMALAAlgorithm(model, self._settings.step_width)
        storage_settings = self._settings.storage
        storage = ZarrStorage(
            save_directory=self._output_dir / SAMPLE_STORE_NAME,
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

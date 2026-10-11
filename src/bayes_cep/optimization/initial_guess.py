r"""Initial-guess strategies for MAP estimation: where the optimizer starts.

Classes:
    InitialGuessStrategy: ABC interface for producing the initial guess from the prior.
    PriorMeanInitialGuess: Start at the prior mean.
    PriorSampleInitialGuess: Start at a draw from the prior, with the strategy's own seed.
    FileInitialGuess: Start at a state stored in a `.npy` file, e.g. another run's estimate.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
from ls_bayesian.posterior.interfaces import GaussianPrior

from bayes_cep.run.directories import resolve_repository_path


# ==================================================================================================
class InitialGuessStrategy(ABC):
    """ABC interface for producing the initial guess of an optimization.

    Methods:
        build: Produce the initial guess.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def build(self, prior: GaussianPrior) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Produce the initial guess, shape `(prior.random_vector_size,)` in vertex space.

        Args:
            prior (GaussianPrior): The prior of the posterior that is optimized.

        Returns:
            np.ndarray: Initial fiber angle [rad] per vertex.

        Raises:
            ValueError: If the guess does not have the shape of the prior mean.
        """


# ==================================================================================================
@dataclass(frozen=True)
class PriorMeanInitialGuess(InitialGuessStrategy):
    """Start at the prior mean."""

    # ----------------------------------------------------------------------------------------------
    @override
    def build(self, prior: GaussianPrior) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.array(prior.mean_vector, dtype=np.float64)


# ==================================================================================================
@dataclass(frozen=True)
class PriorSampleInitialGuess(InitialGuessStrategy):
    r"""Start at a draw from the prior: $m_0 = \overline{m} + \widehat{\mathcal{C}}\,\xi$.

    The standard normal vector $\xi$ is drawn from a generator seeded with `seed`, independent of
    the seed of the prior, so the initial guess is reproducible and varies with `seed` alone.

    Attributes:
        seed (int): Seed of the draw. It must differ from the seed of a synthetic ground truth
            drawn from the same prior parameters, else the start is the ground truth.
    """

    seed: int

    # ----------------------------------------------------------------------------------------------
    @override
    def build(self, prior: GaussianPrior) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        random_vector = np.random.default_rng(self.seed).normal(size=prior.random_vector_size)
        return prior.mean_vector + prior.apply_covariance_factorization(random_vector)


# ==================================================================================================
@dataclass(frozen=True)
class FileInitialGuess(InitialGuessStrategy):
    """Start at the state stored in a `.npy` file, e.g. the estimate of another run.

    Attributes:
        path (Path): The file, one angle [rad] per vertex; a relative path is relative to the
            repository.
    """

    path: Path

    # ----------------------------------------------------------------------------------------------
    @override
    def build(self, prior: GaussianPrior) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        resolved_path = resolve_repository_path(self.path)
        initial_guess = np.load(resolved_path)
        if initial_guess.shape != prior.mean_vector.shape:
            raise ValueError(
                f"The initial guess in {resolved_path} has shape {initial_guess.shape}, but the "
                f"prior mean has shape {prior.mean_vector.shape}."
            )
        return initial_guess

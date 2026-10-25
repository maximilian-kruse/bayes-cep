r"""The prior mean of the inference, chosen from the observations alone.

The mean of the ground truth is not known in a realistic scenario, so a strategy may use the
observed activation times but never the ground truth. A constant angle is the simplest informed
choice: the fiber angle is $\pi$-periodic and the prior is real-valued, so the constant decides
where the seam of the branch window $(-\pi/2, \pi/2]$ lies relative to the angles of the field, and
a constant offset costs the prior almost nothing.

Classes:
    PriorMeanStrategy: ABC interface for producing the prior mean angle field.
    ZeroPriorMeanStrategy: The mean is zero.
    BestFitConstantPriorMeanStrategy: The constant angle that fits the observations best.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import numpy as np
from ls_bayesian.posterior.interfaces import ParameterToSolutionMap


# ==================================================================================================
class PriorMeanStrategy(ABC):
    """ABC interface for producing the prior mean angle field from the observations.

    Methods:
        build: Produce the prior mean.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def build(
        self,
        forward_map: ParameterToSolutionMap,
        observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.integer]],
        observed_activation_times: np.ndarray[tuple[int], np.dtype[np.float64]],
        num_vertices: int,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Produce the prior mean angle [rad] per vertex, shape `(num_vertices,)`.

        Args:
            forward_map (ParameterToSolutionMap): Forward map from the fiber angle to the
                activation times; only `evaluate_forward` is used.
            observed_vertex_indices (np.ndarray): Indices of the observed vertices.
            observed_activation_times (np.ndarray): Noisy activation times at those vertices.
            num_vertices (int): Number of mesh vertices.

        Returns:
            np.ndarray: The prior mean.
        """


# ==================================================================================================
@dataclass(frozen=True)
class ZeroPriorMeanStrategy(PriorMeanStrategy):
    """The prior mean is zero everywhere: no information from the observations is used."""

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        forward_map: ParameterToSolutionMap,
        observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.integer]],
        observed_activation_times: np.ndarray[tuple[int], np.dtype[np.float64]],
        num_vertices: int,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Build the zero prior mean."""
        return np.zeros(num_vertices)


# ==================================================================================================
@dataclass(frozen=True)
class BestFitConstantPriorMeanStrategy(PriorMeanStrategy):
    r"""The prior mean is the constant angle whose activation times fit the observations best.

    The candidates are `num_candidates` equispaced angles in $[-\pi/2, \pi/2)$, one period of the
    axial angle; the best is the one with the smallest sum of squared differences between its
    activation times and the observations. It costs one forward solve per candidate.

    Attributes:
        num_candidates (int): Number of constant angles tried, at least 2.
    """

    num_candidates: int

    def __post_init__(self) -> None:
        if self.num_candidates < 2:
            raise ValueError(f"num_candidates must be at least 2, got {self.num_candidates}.")

    # ----------------------------------------------------------------------------------------------
    @override
    def build(
        self,
        forward_map: ParameterToSolutionMap,
        observed_vertex_indices: np.ndarray[tuple[int], np.dtype[np.integer]],
        observed_activation_times: np.ndarray[tuple[int], np.dtype[np.float64]],
        num_vertices: int,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Build the constant prior mean of the best-fitting candidate."""
        candidates = np.linspace(-np.pi / 2, np.pi / 2, self.num_candidates, endpoint=False)
        misfits = np.array(
            [
                np.sum(
                    (
                        forward_map.evaluate_forward(np.full(num_vertices, angle))[
                            observed_vertex_indices
                        ]
                        - observed_activation_times
                    )
                    ** 2
                )
                for angle in candidates
            ]
        )
        return np.full(num_vertices, candidates[int(np.argmin(misfits))])

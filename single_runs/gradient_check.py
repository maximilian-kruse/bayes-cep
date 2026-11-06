r"""Finite-difference check of the gradient at states of a stored MCMC chain, by local acceptance.

Where a chain's acceptance rate drops, the cause is either the geometry (the preconditioner does
not fit the posterior there) or the gradient (the forward map is piecewise smooth and single
precision, so the adjoint gradient can be wrong or noisy in some regions, which no preconditioner
repairs). The run tells them apart. It reads the chain, computes the acceptance rate of windows of
consecutive samples (a sample equal to its predecessor is a rejection), takes states from the
windows with the lowest and the highest acceptance, and compares at each state the directional
derivative of the likelihood potential $\Phi$ along a few directions,

$$g^T v \quad\text{against}\quad \frac{\Phi(m + hv) - \Phi(m - hv)}{2h},$$

with $g$ the adjoint gradient the sampler uses. The directions $v$ have unit root mean square per
vertex: the gradient direction, white noise, and draws from the prior (smooth, like the moves of
the chain). Several step widths $h$ separate the truncation error of large steps from the noise of
the single-precision forward solve at small ones. The prior part of the potential is exact and
left out.

Classes:
    GradientCheckRunConfig: The chain, the posterior it samples and the check.
    GradientCheckResults: The arrays the run writes.
    GradientCheckRun: The gradient check run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.storage import open_zarr_samples

from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter, describe_array
from bayes_cep.run.template import Metrics, Run
from single_runs.inference import InferenceProblemConfig, assemble_posterior

LOW_ACCEPTANCE_GROUP = 0
HIGH_ACCEPTANCE_GROUP = 1
_READ_BLOCK_SIZE = 1000  # samples read at a time to find the rejections


# ==================================================================================================
@dataclass(frozen=True)
class GradientCheckRunConfig(RunConfig):
    """The chain, the posterior it samples and the check.

    Attributes:
        problem (InferenceProblemConfig): The posterior the chain samples.
        chain_path (Path): The Zarr store (`samples.zarr`) of a finished MCMC run of `problem`; a
            relative path is relative to the repository. It is a directory and not hashed into the
            metadata of the run.
        burn_in (int): Number of samples discarded from the start of the chain.
        window_size (int): Number of consecutive samples of a window of the acceptance rate.
        num_windows (int): Number of windows of each group, the ones with the lowest and the ones
            with the highest acceptance rate.
        states_per_window (int): Number of states taken from each of these windows, evenly spaced.
        step_widths (tuple[float, ...]): Steps $h$ [rad per vertex] of the central differences.
        num_white_directions (int): Number of white-noise directions.
        num_prior_directions (int): Number of directions drawn from the prior.
        seed (int): Seed of the white-noise directions.
    """

    problem: InferenceProblemConfig
    chain_path: Path
    burn_in: int
    window_size: int
    num_windows: int
    states_per_window: int
    step_widths: tuple[float, ...]
    num_white_directions: int
    num_prior_directions: int
    seed: int

    def __post_init__(self) -> None:
        if self.burn_in < 0 or min(self.window_size, self.num_windows, self.states_per_window) < 1:
            raise ValueError(
                "burn_in must be non-negative, window_size, num_windows and states_per_window "
                f"positive, got {self.burn_in}, {self.window_size}, {self.num_windows} and "
                f"{self.states_per_window}."
            )
        if len(self.step_widths) == 0 or min(self.step_widths) <= 0.0:
            raise ValueError(f"step_widths must be positive and non-empty, got {self.step_widths}.")
        if min(self.num_white_directions, self.num_prior_directions) < 0:
            raise ValueError(
                "The numbers of directions must be non-negative, got "
                f"{self.num_white_directions} and {self.num_prior_directions}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class GradientCheckResults(ArrayRecord):
    """The arrays a gradient check writes; $S$ is the number of states, $D$ the number of
    directions (gradient, white noise, prior draws, in this order), $H$ the number of steps.

    Attributes:
        window_acceptance (np.ndarray): Acceptance rate of each window of the chain after the
            burn-in, shape `(W,)`.
        state_indices (np.ndarray): Sample index of each state in the chain, shape `(S,)`.
        state_groups (np.ndarray): `LOW_ACCEPTANCE_GROUP` or `HIGH_ACCEPTANCE_GROUP` of each state,
            shape `(S,)`.
        state_window_acceptance (np.ndarray): Acceptance rate of the window of each state, shape
            `(S,)`.
        gradient_norm (np.ndarray): Euclidean norm of the likelihood gradient at each state, shape
            `(S,)`.
        analytic_derivative (np.ndarray): $g^Tv$, shape `(S, D)`.
        finite_difference (np.ndarray): The central differences, shape `(S, D, H)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "window_acceptance": "window_acceptance.npy",
        "state_indices": "state_indices.npy",
        "state_groups": "state_groups.npy",
        "state_window_acceptance": "state_window_acceptance.npy",
        "gradient_norm": "gradient_norm.npy",
        "analytic_derivative": "analytic_derivative.npy",
        "finite_difference": "finite_difference.npy",
    }

    window_acceptance: np.ndarray
    state_indices: np.ndarray
    state_groups: np.ndarray
    state_window_acceptance: np.ndarray
    gradient_norm: np.ndarray
    analytic_derivative: np.ndarray
    finite_difference: np.ndarray


_FILES = GradientCheckResults.FILE_NAMES  # the names of the result files, defined once there


# ==================================================================================================
def compute_window_acceptance(chain, burn_in: int, window_size: int) -> np.ndarray:
    """Acceptance rate of consecutive windows of a chain after the burn-in.

    A sample that equals its predecessor is a rejection. The last window is dropped if it is
    shorter than `window_size`.

    Args:
        chain: The samples, shape `(N, num_vertices)`, readable by slices (a Zarr array).
        burn_in (int): Number of samples discarded from the start.
        window_size (int): Number of samples of a window.

    Returns:
        np.ndarray: Acceptance rate per window, shape `(W,)`.

    Raises:
        ValueError: If the chain after the burn-in is shorter than one window.
    """
    num_samples = chain.shape[0]
    num_windows = (num_samples - burn_in) // window_size
    if num_windows < 1:
        raise ValueError(
            f"The chain has {num_samples - burn_in} samples after the burn-in, fewer than the "
            f"window size {window_size}."
        )
    moved = np.empty(num_windows * window_size, dtype=bool)
    start = burn_in
    previous = np.asarray(chain[start - 1 : start]) if start > 0 else None
    for first in range(start, start + moved.size, _READ_BLOCK_SIZE):
        last = min(first + _READ_BLOCK_SIZE, start + moved.size)
        block = np.asarray(chain[first:last])
        if previous is None:  # the first sample of the chain has no predecessor
            flags = np.concatenate([[True], np.any(block[1:] != block[:-1], axis=1)])
        else:
            flags = np.any(block != np.concatenate([previous, block[:-1]]), axis=1)
        moved[first - start : last - start] = flags
        previous = block[-1:]
    return moved.reshape(num_windows, window_size).mean(axis=1)


# ==================================================================================================
class GradientCheckRun(Run[GradientCheckRunConfig]):
    """Compare the adjoint gradient with finite differences at states of low and high acceptance."""

    config_type = GradientCheckRunConfig
    name = "gradient-check"
    outputs = {
        f"results/{_FILES['window_acceptance']}": "Acceptance rate of the windows of the chain.",
        f"results/{_FILES['state_indices']}": "Sample index of each checked state.",
        f"results/{_FILES['state_groups']}": "Low (0) or high (1) acceptance window of each state.",
        f"results/{_FILES['state_window_acceptance']}": "Acceptance rate of the window of each "
        "state.",
        f"results/{_FILES['gradient_norm']}": "Norm of the likelihood gradient at each state.",
        f"results/{_FILES['analytic_derivative']}": "Directional derivative from the adjoint "
        "gradient, per state and direction.",
        f"results/{_FILES['finite_difference']}": "Central differences, per state, direction and "
        "step.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        # The chain is a directory (a Zarr store) and not hashed.
        return self.config.problem.input_files()

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed across the runs of a study."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior(config.problem, steps, logger)
        log_posterior = assembled.log_posterior
        prior = log_posterior.prior
        chain = open_zarr_samples(resolve_repository_path(config.chain_path))
        num_vertices = chain.shape[1]

        with steps.step("Computing the acceptance rate of the windows of the chain"):
            window_acceptance = compute_window_acceptance(chain, config.burn_in, config.window_size)
        logger.info(f"      {describe_array('window acceptance', window_acceptance)}")
        num_windows = min(config.num_windows, window_acceptance.size // 2)
        order = np.argsort(window_acceptance)
        groups = {
            LOW_ACCEPTANCE_GROUP: order[:num_windows],
            HIGH_ACCEPTANCE_GROUP: order[::-1][:num_windows],
        }
        state_indices, state_groups, state_acceptance = [], [], []
        for group, windows in groups.items():
            for window in sorted(windows):
                offsets = (np.arange(config.states_per_window) + 0.5) / config.states_per_window
                first = config.burn_in + int(window) * config.window_size
                state_indices.extend(first + (offsets * config.window_size).astype(int))
                state_groups.extend([group] * config.states_per_window)
                state_acceptance.extend([window_acceptance[window]] * config.states_per_window)
        state_indices = np.asarray(state_indices)
        states = np.stack([np.asarray(chain[int(index)]) for index in state_indices])

        rng = np.random.default_rng(config.seed)
        white = rng.standard_normal((config.num_white_directions, num_vertices))
        smooth = np.stack(
            [
                prior.generate_sample() - prior.mean_vector
                for _ in range(config.num_prior_directions)
            ]
        ).reshape(config.num_prior_directions, num_vertices)
        # Unit root mean square per vertex, so that a step h is h radians per vertex.
        extra = np.concatenate([white, smooth])
        extra *= np.sqrt(num_vertices) / np.linalg.norm(extra, axis=1, keepdims=True)

        num_directions = 1 + extra.shape[0]
        analytic = np.empty((states.shape[0], num_directions))
        finite_difference = np.empty((states.shape[0], num_directions, len(config.step_widths)))
        gradient_norm = np.empty(states.shape[0])
        with steps.step(f"Checking the gradient at {states.shape[0]} states"):
            for row, state in enumerate(states):
                gradient = log_posterior.evaluate_likelihood_gradient(state)
                gradient_norm[row] = np.linalg.norm(gradient)
                gradient_direction = gradient * (np.sqrt(num_vertices) / gradient_norm[row])
                directions = np.concatenate([gradient_direction[None], extra])
                analytic[row] = directions @ gradient
                for column, direction in enumerate(directions):
                    for k, width in enumerate(config.step_widths):
                        forward = log_posterior.evaluate_likelihood_cost(state + width * direction)
                        backward = log_posterior.evaluate_likelihood_cost(state - width * direction)
                        finite_difference[row, column, k] = (forward - backward) / (2.0 * width)
                logger.info(
                    f"      state {row + 1}/{states.shape[0]} (sample {state_indices[row]}, "
                    f"window acceptance {state_acceptance[row]:.2f}): |g| = "
                    f"{gradient_norm[row]:.4g}"
                )

        GradientCheckResults(
            window_acceptance=window_acceptance,
            state_indices=state_indices,
            state_groups=np.asarray(state_groups),
            state_window_acceptance=np.asarray(state_acceptance),
            gradient_norm=gradient_norm,
            analytic_derivative=analytic,
            finite_difference=finite_difference,
        ).save(self.results_dir(run_dir))

        metrics: Metrics = {
            "num_states": int(states.shape[0]),
            "acceptance_low": float(window_acceptance[groups[LOW_ACCEPTANCE_GROUP]].mean()),
            "acceptance_high": float(window_acceptance[groups[HIGH_ACCEPTANCE_GROUP]].mean()),
        }
        with np.errstate(invalid="ignore", divide="ignore"):
            relative_error = np.abs(finite_difference - analytic[:, :, None]) / np.abs(
                analytic[:, :, None]
            )
        group_array = np.asarray(state_groups)
        for group, label in ((LOW_ACCEPTANCE_GROUP, "low"), (HIGH_ACCEPTANCE_GROUP, "high")):
            for k in range(len(config.step_widths)):
                metrics[f"relative_error_median_{label}_step{k}"] = float(
                    np.nanmedian(relative_error[group_array == group, :, k])
                )
            metrics[f"gradient_norm_median_{label}"] = float(
                np.median(gradient_norm[group_array == group])
            )
        return metrics

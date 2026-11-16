r"""Finite-difference check of the likelihood gradient along Laplace eigenvectors.

The adjoint gradient of the forward model is the one a sampler or an optimizer uses; the
`gradient_check` run tests it along a few random directions at chain states. The posterior is slow
and wide along the leading eigenvectors $v_i$ of the Laplace approximation, and these are the
directions in which an error of the gradient would hurt the samplers. This run compares, at stored
states (e.g. the MAP estimate), the directional derivative $g^T v_i$ of the likelihood potential
$\Phi$ with central differences

$$\frac{\Phi(m + h v_i) - \Phi(m - h v_i)}{2h}$$

for several step widths $h$ [rad per used vertex]. The directions are the columns of a stored
eigenvector matrix (e.g. the `hessian_eigenvectors.npy` of a Laplace run), scaled to unit root mean
square over the used vertices.

The forward solver runs in single precision, so the finite differences have a noise floor: below a
step of about 1e-3 rad per vertex they are noise (the error of the potential is about 4e-3, and
$2h\,g^Tv$ must be well above it), and above about 1e-2 their truncation error dominates. Use steps
in this window, judge the agreement by `error_over_scale` (the error relative to the largest
derivative the gradient allows, $\|g\|\sqrt{N}$, not relative to $g^Tv$, which is small along the
leading eigenvectors where the likelihood and prior gradients cancel), and use the Richardson
extrapolation of the two smallest steps, which removes the $h^2$ truncation error.

Classes:
    GradientCheckEigenvectorsRunConfig: The states, the directions and the steps.
    GradientCheckEigenvectorsResults: The arrays the run writes.
    GradientCheckEigenvectorsRun: The run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger

from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from single_runs.inference import InferenceProblemConfig, assemble_posterior

MINIMUM_STEP_WIDTH = 1e-3  # rad per used vertex; below it the single-precision potential is noise


# ==================================================================================================
@dataclass(frozen=True)
class GradientCheckEigenvectorsRunConfig(RunConfig):
    """The states, the directions and the steps of the check.

    Attributes:
        problem (InferenceProblemConfig): The posterior whose likelihood potential is checked.
        state_paths (tuple[Path, ...]): `.npy` files with the states, one angle [rad] per vertex
            each, e.g. the MAP estimate.
        eigenvectors_path (Path): `.npy` file with the eigenvectors as columns, shape
            `(num_vertices, rank)`, descending eigenvalues.
        direction_indices (tuple[int, ...]): Columns of the eigenvectors that are used as
            directions, e.g. leading, middle and trailing ones.
        num_prior_directions (int): Number of further directions, draws from the prior minus its
            mean (smooth, like the moves of a chain), scaled like the eigenvectors; they follow the
            eigenvectors in the arrays.
        include_gradient_direction (bool): Add the direction of the likelihood gradient at each
            state, scaled like the others, as the last direction.
        step_widths (tuple[float, ...]): Steps $h$ [rad per used vertex] of the central
            differences, strictly increasing, at least two, none below `MINIMUM_STEP_WIDTH` (the
            single-precision noise floor). The two smallest are Richardson-extrapolated.
    """

    problem: InferenceProblemConfig
    state_paths: tuple[Path, ...]
    eigenvectors_path: Path
    direction_indices: tuple[int, ...]
    num_prior_directions: int
    include_gradient_direction: bool
    step_widths: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.state_paths:
            raise ValueError("state_paths must not be empty.")
        if self.num_prior_directions < 0:
            raise ValueError(
                f"num_prior_directions must be non-negative, got {self.num_prior_directions}."
            )
        if any(index < 0 for index in self.direction_indices):
            raise ValueError(
                f"direction_indices must be non-negative, got {self.direction_indices}."
            )
        widths = self.step_widths
        if (
            len(widths) < 2
            or any(second <= first for first, second in zip(widths, widths[1:], strict=False))
            or widths[0] < MINIMUM_STEP_WIDTH
        ):
            raise ValueError(
                f"step_widths must be at least two, strictly increasing values of at least "
                f"{MINIMUM_STEP_WIDTH} (the single-precision noise floor), got {widths}."
            )


# ==================================================================================================
@dataclass(frozen=True)
class GradientCheckEigenvectorsResults(ArrayRecord):
    """The arrays of the check; $S$ states, $D$ directions, $H$ steps, $V$ vertices.

    Attributes:
        likelihood_cost (np.ndarray): Likelihood potential at each state, shape `(S,)`.
        gradient (np.ndarray): Likelihood gradient at each state, shape `(S, V)`.
        analytic_derivative (np.ndarray): $g^Tv$, shape `(S, D)`.
        finite_difference (np.ndarray): The central differences, shape `(S, D, H)`.
        directions (np.ndarray): The directions (unit rms over the used vertices), shape
            `(S, D, V)`; the eigenvectors, then the prior draws, then the gradient direction of
            the state if it is included.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "likelihood_cost": "likelihood_cost.npy",
        "gradient": "gradient.npy",
        "analytic_derivative": "analytic_derivative.npy",
        "finite_difference": "finite_difference.npy",
        "directions": "directions.npy",
    }

    likelihood_cost: np.ndarray
    gradient: np.ndarray
    analytic_derivative: np.ndarray
    finite_difference: np.ndarray
    directions: np.ndarray


_FILES = GradientCheckEigenvectorsResults.FILE_NAMES  # the names of the result files


# ==================================================================================================
class GradientCheckEigenvectorsRun(Run[GradientCheckEigenvectorsRunConfig]):
    """Compare the adjoint gradient with finite differences along Laplace eigenvectors."""

    config_type = GradientCheckEigenvectorsRunConfig
    name = "gradient-check-eigenvectors"
    outputs = {
        f"results/{_FILES['likelihood_cost']}": "Likelihood potential at each state.",
        f"results/{_FILES['gradient']}": "Likelihood gradient at each state.",
        f"results/{_FILES['analytic_derivative']}": "Directional derivative from the adjoint "
        "gradient, per state and direction.",
        f"results/{_FILES['finite_difference']}": "Central differences, per state, direction and "
        "step.",
        f"results/{_FILES['directions']}": "The directions of the check.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        return [
            *self.config.problem.input_files(),
            *(resolve_repository_path(path) for path in self.config.state_paths),
            resolve_repository_path(self.config.eigenvectors_path),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed across runs."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior(config.problem, steps, logger)
        log_posterior = assembled.log_posterior
        used = assembled.forward_map.used_vertex_mask
        eigenvectors = np.load(resolve_repository_path(config.eigenvectors_path), mmap_mode="r")
        if (
            eigenvectors.shape[0] != used.size
            or max(config.direction_indices, default=-1) >= (eigenvectors.shape[1])
        ):
            raise ValueError(
                f"The eigenvectors have shape {eigenvectors.shape}; the mesh has {used.size} "
                f"vertices and the largest direction index is {max(config.direction_indices)}."
            )
        fixed_directions = [
            np.asarray(eigenvectors[:, index], dtype=np.float64)
            for index in config.direction_indices
        ]
        prior = log_posterior.prior
        fixed_directions += [
            np.asarray(prior.generate_sample() - prior.mean_vector).reshape(-1)
            for _ in range(config.num_prior_directions)
        ]
        fixed = np.stack(fixed_directions) if fixed_directions else np.empty((0, used.size))
        states = np.stack([np.load(resolve_repository_path(path)) for path in config.state_paths])

        def scale(directions: np.ndarray) -> np.ndarray:
            """Unit root mean square over the used vertices: a step h is h radians per vertex."""
            norms = np.linalg.norm(directions[:, used], axis=1, keepdims=True)
            return directions * np.sqrt(used.sum()) / norms

        fixed = scale(fixed)
        num_states = states.shape[0]
        num_directions = fixed.shape[0] + int(config.include_gradient_direction)
        likelihood_cost = np.empty(num_states)
        gradients = np.empty((num_states, used.size))
        directions = np.empty((num_states, num_directions, used.size))
        analytic = np.empty((num_states, num_directions))
        finite_difference = np.empty((num_states, num_directions, len(config.step_widths)))
        with steps.step(
            f"Checking the gradient at {num_states} states, {num_directions} directions"
        ):
            for row, state in enumerate(states):
                likelihood_cost[row] = log_posterior.evaluate_likelihood_cost(state)
                gradients[row] = log_posterior.evaluate_likelihood_gradient(state)
                directions[row, : fixed.shape[0]] = fixed
                if config.include_gradient_direction:
                    directions[row, -1] = scale(gradients[row][None])[0]
                analytic[row] = directions[row] @ gradients[row]
                for column, direction in enumerate(directions[row]):
                    for k, width in enumerate(config.step_widths):
                        forward = log_posterior.evaluate_likelihood_cost(state + width * direction)
                        backward = log_posterior.evaluate_likelihood_cost(state - width * direction)
                        finite_difference[row, column, k] = (forward - backward) / (2.0 * width)
                logger.info(f"      state {row + 1}/{num_states} done")

        GradientCheckEigenvectorsResults(
            likelihood_cost=likelihood_cost,
            gradient=gradients,
            analytic_derivative=analytic,
            finite_difference=finite_difference,
            directions=directions,
        ).save(self.results_dir(run_dir))

        # The error is judged against the largest derivative the gradient allows, |g| sqrt(N):
        # g^T v is small along the leading eigenvectors (likelihood and prior gradient cancel).
        derivative_scale = np.linalg.norm(gradients, axis=1)[:, None] * np.sqrt(used.sum())
        widths = np.array(config.step_widths)
        # Central differences have the error c h^2 + ...; this removes c from the two smallest.
        extrapolated = (
            widths[1] ** 2 * finite_difference[:, :, 0]
            - widths[0] ** 2 * finite_difference[:, :, 1]
        ) / (widths[1] ** 2 - widths[0] ** 2)
        errors = np.abs(finite_difference - analytic[:, :, None]) / derivative_scale[:, :, None]
        extrapolated_error = np.abs(extrapolated - analytic) / derivative_scale
        metrics: Metrics = {
            "num_states": int(num_states),
            "num_directions": int(num_directions),
            "num_used_vertices": int(used.sum()),
            "gradient_norm_median": float(np.median(np.linalg.norm(gradients, axis=1))),
            "error_over_scale_extrapolated_median": float(np.median(extrapolated_error)),
            "error_over_scale_extrapolated_max": float(np.max(extrapolated_error)),
        }
        for k in range(len(config.step_widths)):
            metrics[f"error_over_scale_median_step{k}"] = float(np.median(errors[:, :, k]))
            metrics[f"error_over_scale_max_step{k}"] = float(np.max(errors[:, :, k]))
        if config.include_gradient_direction:
            # Along the gradient direction g^Tv is large, so the relative error is meaningful.
            with np.errstate(invalid="ignore", divide="ignore"):
                relative = np.abs(extrapolated[:, -1] - analytic[:, -1]) / np.abs(analytic[:, -1])
            metrics["gradient_direction_relative_error_extrapolated_median"] = float(
                np.median(relative)
            )
        return metrics

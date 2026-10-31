r"""Energy error of Hamiltonian trajectories with the calibrated preconditioner.

PMALA with the calibrated preconditioner rejects above a step width of about 0.02 (see
[`rejection_diagnostics`][single_runs.rejection_diagnostics]): one large jump does not follow the
curved, coupled posterior. Hamiltonian Monte Carlo
([`PHMCAlgorithm`][ls_bayesian.mcmc.algorithms.phmc.PHMCAlgorithm], the infinity-mHMC of Beskos et
al., 2017, with the calibrated Gaussian as mass matrix) takes many small steps instead. In the
coordinates of the preconditioner every direction rotates with the period $2\pi$, so the length of
a trajectory is best measured by its total angle $\varepsilon I$: about $\pi/2$ to $\pi$ moves a
Gaussian target to an almost independent state, for $I \approx 1.5/\varepsilon$ gradients.

At states of finished chains (rows of the `pilot_states.npy` of a calibration run) the run draws
velocities from the preconditioner, integrates a trajectory of the total angle `total_angle` for
each step angle $\varepsilon$ and records after every step

- the energy error $\Delta H$ of the trajectory cut there (its acceptance is $1 \wedge
  e^{-\Delta H}$ for the number of steps it has taken),
- the squared distance of the position from the start in the norm of $\overline K^{-1}$, split into
  the blocks of the calibrated eigenvectors (bands, stiffest first, and the rest) as in the
  rejection diagnostics: how far the stiff directions move relative to their posterior width.

Which trajectory length and step angle give the most movement per gradient follows from these
arrays; the number of gradients of a trajectory is its number of steps.

Classes:
    HmcTrajectoryRunConfig: The posterior, the calibration, the states and the trajectories.
    HmcTrajectoryResults: The arrays the run writes.
    HmcTrajectoryRun: The trajectory run.
"""

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast, override

import numpy as np
from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.algorithms.phmc import PHMCAlgorithm
from ls_bayesian.mcmc.model import MCMCModel

from bayes_cep.laplace.calibration import CalibratedGaussianRecord
from bayes_cep.laplace.reference_covariance import PriorReferenceCovariance
from bayes_cep.mcmc.measures import (
    CenteredGaussianMeasure,
    LikelihoodTargetMeasure,
    LowRankGaussianMeasure,
    MeanTiltedTargetMeasure,
    PriorGaussianMeasure,
)
from bayes_cep.run.config import RunConfig
from bayes_cep.run.directories import METRICS_RECORD, ArrayRecord, resolve_repository_path
from bayes_cep.run.progress import StepReporter
from bayes_cep.run.template import Metrics, Run
from single_runs.calibrate import PilotStates
from single_runs.inference import InferenceProblemConfig, assemble_posterior_components
from single_runs.rejection_diagnostics import _BlockSplit


# ==================================================================================================
@dataclass(frozen=True)
class HmcTrajectoryRunConfig(RunConfig):
    """The posterior, the calibration, the states and the trajectories.

    Attributes:
        problem (InferenceProblemConfig): The posterior.
        calibration_path (Path): The `results` directory of a finished calibration run of this
            posterior; a relative path is relative to the repository. Its calibrated Gaussian is
            the mass matrix and its `pilot_states.npy` holds the states.
        state_rows (tuple[int, ...]): Rows of the saved pilot states to use.
        step_widths (tuple[float, ...]): The step angles $\\varepsilon$ to test.
        total_angle (float): Total angle $\\varepsilon I$ of a trajectory; it has
            $I = \\lceil\\text{total\\_angle} / \\varepsilon\\rceil$ steps.
        band_edges (tuple[int, ...]): Indices of the eigenvectors that bound the bands of the
            displacement, as in the rejection diagnostics.
        num_trajectories (int): Number of velocity draws per state.
        seed (int): Seed of the velocities; the draws of a state also depend on its row.
    """

    problem: InferenceProblemConfig
    calibration_path: Path
    state_rows: tuple[int, ...]
    step_widths: tuple[float, ...]
    total_angle: float
    band_edges: tuple[int, ...]
    num_trajectories: int
    seed: int

    def __post_init__(self) -> None:
        if len(self.state_rows) == 0 or any(row < 0 for row in self.state_rows):
            raise ValueError("state_rows must be a non-empty tuple of non-negative rows.")
        if len(self.step_widths) == 0 or any(width <= 0 for width in self.step_widths):
            raise ValueError("step_widths must be a non-empty tuple of positive step angles.")
        if self.total_angle <= 0:
            raise ValueError(f"total_angle must be positive, got {self.total_angle}.")
        edges = self.band_edges
        if (
            len(edges) < 2
            or edges[0] != 0
            or any(a >= b for a, b in zip(edges, edges[1:], strict=False))
        ):
            raise ValueError("band_edges must start with 0 and increase strictly.")
        if self.num_trajectories < 1:
            raise ValueError(f"num_trajectories must be positive, got {self.num_trajectories}.")


# ==================================================================================================
@dataclass(frozen=True)
class HmcTrajectoryResults(ArrayRecord):
    r"""The arrays of a trajectory run; $S$ states, $D$ step angles, $R$ velocity draws, $I$ the
    largest number of steps and $B$ blocks (the bands and the rest).

    Attributes:
        step_widths (np.ndarray): The step angles, shape `(D,)`.
        num_steps (np.ndarray): The number of steps of a trajectory per step angle, shape `(D,)`.
        band_edges (np.ndarray): The band edges, shape `(B,)`.
        state_rows (np.ndarray): The rows of the pilot states, shape `(S,)`.
        energy_changes (np.ndarray): $\\Delta H$ after each step, NaN after the last step of the
            step angle, shape `(S, D, R, I)`.
        displacements (np.ndarray): $\\|P_b(u_l - u_0)\\|^2_{\\overline K^{-1}}$ per block after
            each step, NaN after the last step, shape `(S, D, R, I, B)`.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]] = {
        "step_widths": "step_widths.npy",
        "num_steps": "num_steps.npy",
        "band_edges": "band_edges.npy",
        "state_rows": "state_rows.npy",
        "energy_changes": "energy_changes.npy",
        "displacements": "displacements.npy",
    }

    step_widths: np.ndarray
    num_steps: np.ndarray
    band_edges: np.ndarray
    state_rows: np.ndarray
    energy_changes: np.ndarray
    displacements: np.ndarray


# ==================================================================================================
class HmcTrajectoryRun(Run[HmcTrajectoryRunConfig]):
    """Energy error and movement of Hamiltonian trajectories at states of finished chains."""

    config_type = HmcTrajectoryRunConfig
    name = "hmc-trajectory"
    outputs = {
        f"results/{file_name}": f"The array {field_name}."
        for field_name, file_name in HmcTrajectoryResults.FILE_NAMES.items()
    } | {
        METRICS_RECORD: "Acceptance of the trajectories at the total angles pi/2 and pi per step "
        "angle, the largest energy error, the divergences and the seconds per gradient.",
    }

    # ----------------------------------------------------------------------------------------------
    @override
    def input_files(self) -> list[Path]:
        calibration_dir = resolve_repository_path(self.config.calibration_path)
        return [
            *self.config.problem.input_files(),
            *CalibratedGaussianRecord.file_paths(calibration_dir),
            *PilotStates.file_paths(calibration_dir),
        ]

    # ----------------------------------------------------------------------------------------------
    @override
    def report(self, run_dir: Path) -> None:
        """No plots: the arrays are analysed in a notebook."""

    # ----------------------------------------------------------------------------------------------
    @override
    def _execute(self, run_dir: Path, logger: BaseLogger) -> Metrics:
        config = self.config
        steps = StepReporter(logger)
        assembled = assemble_posterior_components(config.problem, steps, logger)
        prior = assembled.components.prior
        calibration_dir = resolve_repository_path(config.calibration_path)
        with steps.step("Loading the calibrated Gaussian and the states"):
            record = CalibratedGaussianRecord.load(calibration_dir)
            pilot = PilotStates.load(calibration_dir)
            gaussian = record.build_gaussian(PriorReferenceCovariance(prior), config.seed)
        if max(config.state_rows) >= pilot.states.shape[0]:
            raise ValueError(
                f"state_rows reach {max(config.state_rows)}, there are {pilot.states.shape[0]}."
            )

        target = MeanTiltedTargetMeasure(
            LikelihoodTargetMeasure(assembled.components.to_log_posterior()), prior
        )
        reference = CenteredGaussianMeasure(PriorGaussianMeasure(prior))
        approximation = LowRankGaussianMeasure(gaussian)
        algorithm = PHMCAlgorithm(
            MCMCModel(target=target, reference=reference, approximation=approximation),
            step_width=config.step_widths[0],
            num_steps=1,
        )
        with steps.step("Setting up the blocks of directions"):
            split = _BlockSplit(record.eigenvectors, config.band_edges, reference, approximation)

        num_steps = np.array([math.ceil(config.total_angle / w) for w in config.step_widths])
        largest = int(num_steps.max())
        blocks = split.num_blocks
        shape = (len(config.state_rows), len(config.step_widths), config.num_trajectories)
        energy_changes = np.full(shape + (largest,), np.nan)
        displacements = np.full(shape + (largest, blocks), np.nan)
        total_steps = int(num_steps.sum()) * len(config.state_rows) * config.num_trajectories
        logger.info(f"      {total_steps} leapfrog steps in all, a gradient each")

        start = time.perf_counter()
        gradients_before = algorithm.num_gradient_evaluations
        with steps.step("Integrating the trajectories"):
            for position, row in enumerate(config.state_rows):
                state = pilot.states[row]
                rng = np.random.default_rng([config.seed, int(row)])
                for draw in range(config.num_trajectories):
                    velocity = approximation.apply_covariance_factorization(
                        rng.normal(size=approximation.random_vector_size)
                    )
                    for d, width in enumerate(config.step_widths):
                        trajectory = algorithm.integrate(
                            state, velocity, width, int(num_steps[d]), record_path=True
                        )
                        assert trajectory.energy_changes is not None
                        assert trajectory.positions is not None
                        count = int(num_steps[d])
                        energy_changes[position, d, draw, :count] = trajectory.energy_changes
                        for step in range(count):
                            displacements[position, d, draw, step] = split.forms(
                                trajectory.positions[step] - state
                            )[0]
                logger.info(
                    f"      state {position + 1} of {len(config.state_rows)} (row {row}) done, "
                    f"{(time.perf_counter() - start) / 60:.1f} min elapsed"
                )
        gradients = algorithm.num_gradient_evaluations - gradients_before

        results_dir = self.results_dir(run_dir)
        HmcTrajectoryResults(
            step_widths=np.array(config.step_widths),
            num_steps=num_steps,
            band_edges=np.array(config.band_edges),
            state_rows=np.array(config.state_rows),
            energy_changes=energy_changes,
            displacements=displacements,
        ).save(results_dir)

        valid = np.arange(largest)[None, None, None, :] < num_steps[None, :, None, None]
        metrics: Metrics = {
            "num_gradient_evaluations": gradients,
            "seconds_per_gradient": (time.perf_counter() - start) / gradients,
            "num_non_finite": int(np.sum(~np.isfinite(energy_changes) & valid)),
        }
        finite = energy_changes[np.isfinite(energy_changes)]
        metrics["max_abs_energy_change"] = float(np.abs(finite).max()) if finite.size else 0.0
        for d, width in enumerate(config.step_widths):
            for label, angle in (("half_pi", np.pi / 2), ("pi", np.pi)):
                step = min(int(num_steps[d]), math.ceil(angle / width)) - 1
                change = energy_changes[:, d, :, step]
                acceptance = np.where(np.isfinite(change), np.exp(np.minimum(0.0, -change)), 0.0)
                metrics[f"acceptance_h{width:g}_{label}"] = float(np.mean(acceptance))
        return cast(Metrics, metrics)

"""Metadata of a run: when, where and on which inputs it executed.

The environment (code and packages) is collected separately, once per submission, see
[`Environment`][bayes_cep.run.environment.Environment]. The record is a frozen dataclass;
`dataclasses.asdict` gives its JSON form.

Classes:
    RunMetadata: Environment, inputs and machine of one run.
"""

import hashlib
import os
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from bayes_cep.run.directories import REPOSITORY_ROOT
from bayes_cep.run.environment import Environment


# ==================================================================================================
@dataclass(frozen=True)
class RunMetadata:
    """Data for a run.

    Attributes:
        started (str): Start time, ISO 8601 with time zone.
        environment (Environment): The code the run executed.
        input_sha256 (dict[str, str | None]): Content hash of every input file, by path relative
            to the repository (absolute outside of it); `None` for a file that does not exist.
        host (str): Name of the machine.
        cpu_count (int | None): Number of cores of the machine.
        slurm_job_id (str | None): SLURM job id if the run is a cluster task.
        slurm_array_task_id (str | None): SLURM array task id if the run is a cluster task.
    """

    started: str
    environment: Environment
    input_sha256: dict[str, str | None]
    host: str
    cpu_count: int | None
    slurm_job_id: str | None
    slurm_array_task_id: str | None

    @classmethod
    def collect_for_run(cls, input_files: Sequence[Path], environment: Environment) -> Self:
        """Collect the metadata of a run about to start.

        Args:
            input_files (Sequence[Path]): Files the run reads; a missing file is recorded as
                `None`.
            environment (Environment): The environment the run executes in. Collected once by the
                submitter instead of per run, since it costs several git calls and a scan of all
                installed packages.

        Returns:
            Self: The environment, the content hashes of the input files, the machine, and the
                SLURM job identifiers if the run is a cluster task.
        """
        return cls(
            started=datetime.now().astimezone().isoformat(timespec="seconds"),
            environment=environment,
            input_sha256=cls._hash_input_files(input_files),
            host=socket.gethostname(),
            cpu_count=os.cpu_count(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _hash_input_files(input_files: Sequence[Path]) -> dict[str, str | None]:
        """Content hash of each file by repository-relative path; `None` for a missing file."""
        hashes: dict[str, str | None] = {}
        for path in input_files:
            resolved = path.resolve()
            if resolved.is_relative_to(REPOSITORY_ROOT):
                key = str(resolved.relative_to(REPOSITORY_ROOT))
            else:
                key = str(resolved)
            if not resolved.exists():
                hashes[key] = None
                continue
            with resolved.open("rb") as file:
                hashes[key] = hashlib.file_digest(file, "sha256").hexdigest()
        return hashes

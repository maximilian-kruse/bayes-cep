"""Archive of the specification behind an environment, copied into a study directory.

Classes:
    EnvironmentArchive: The specification behind an `Environment`, copied for the study directory.
"""

import hashlib
import shutil
import subprocess
import warnings
from pathlib import Path

from bayes_cep.run.directories import REPOSITORY_ROOT
from bayes_cep.run.environment import LOCK_FILE_NAME, Environment, GitRepository

PROJECT_FILE_NAME = "pyproject.toml"
CONDA_SPECIFICATION_PATTERN = "*_conda_spec.txt"
PATCH_FILE_NAME = "source.patch"
EXPORT_TIMEOUT_SECONDS = 120


# ==================================================================================================
class EnvironmentArchive:
    """The content behind an `Environment`: its specification, copied for the study directory.

    The `Environment` records fingerprints (the hash of `pixi.lock`, the hash of the uncommitted
    changes); the archive stores the files they stand for and checks that they match. Together, the
    archived files rebuild the environment (`pixi install --locked`) without the original machine:

    - `pixi.lock` and `pyproject.toml`: the complete, pinned specification of all environments.
    - `<environment>_<platform>_conda_spec.txt`: the exact conda packages of the pixi environment in
      use, installable with conda or mamba. PyPI and path dependencies are not part of it (they are
      in `pixi.lock` and in the git states of the environment). Skipped, with a warning, if the
      process does not run in a pixi environment or the export fails.
    - `source.patch`: the uncommitted changes of tracked files of this repository, if any (without
      `pixi.lock`, which is archived itself); `git apply` restores them on top of the commit.
      Untracked files are not included.
    """

    def __init__(self, environment: Environment, target_dir: Path) -> None:
        """Archive the specification behind `environment` into `target_dir`.

        Args:
            environment (Environment): The environment collected from the current process.
            target_dir (Path): Directory to write to; created if missing, existing files stay.
        """
        self._environment = environment
        self._target_dir = target_dir

    # ----------------------------------------------------------------------------------------------
    def write_specification(self) -> list[str]:
        """Write the archive.

        A warning is issued if an archived file does not match the hash recorded in the
        environment, i.e. if it changed since the environment was collected.

        Returns:
            list[str]: The names of the files in the target directory.
        """
        self._target_dir.mkdir(parents=True, exist_ok=True)
        for name in (LOCK_FILE_NAME, PROJECT_FILE_NAME):
            if (REPOSITORY_ROOT / name).exists():
                shutil.copy(REPOSITORY_ROOT / name, self._target_dir / name)
        self._check_hash(
            LOCK_FILE_NAME,
            self._environment.pixi_lock_sha256,
            self._read_archived(LOCK_FILE_NAME),
        )
        if self._environment.pixi_environment is None:
            warnings.warn(
                "Not in a pixi environment: no conda specification archived.", stacklevel=2
            )
        else:
            self._export_conda_specification(self._environment.pixi_environment)
        patch = GitRepository(REPOSITORY_ROOT).read_uncommitted_changes()
        if patch:
            (self._target_dir / PATCH_FILE_NAME).write_text(patch, encoding="utf-8")
        self._check_hash(
            PATCH_FILE_NAME, self._environment.git.diff_sha256, self._read_archived(PATCH_FILE_NAME)
        )
        return sorted(path.name for path in self._target_dir.iterdir())

    # ----------------------------------------------------------------------------------------------
    def _read_archived(self, name: str) -> bytes | None:
        """The content of an archived file, or `None` if it was not written."""
        path = self._target_dir / name
        return path.read_bytes() if path.exists() else None

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _check_hash(name: str, recorded_sha256: str | None, content: bytes | None) -> None:
        """Warn if the archived content does not have the hash recorded in the environment."""
        archived_sha256 = None if content is None else hashlib.sha256(content).hexdigest()
        if archived_sha256 != recorded_sha256:
            warnings.warn(
                f"The archived {name} does not match the environment collected before: it "
                "changed in between.",
                stacklevel=2,
            )

    # ----------------------------------------------------------------------------------------------
    def _export_conda_specification(self, pixi_environment: str) -> None:
        """Write the explicit conda specification of a pixi environment into the target directory.

        The specification is optional: if `pixi` is unavailable or the export fails, a warning is
        issued and nothing is written.
        """
        if shutil.which("pixi") is None:
            warnings.warn("pixi not found: no conda specification archived.", stacklevel=2)
            return
        command = [
            "pixi",
            "workspace",
            "export",
            "conda-explicit-spec",
            str(self._target_dir),
            f"--environment={pixi_environment}",
            "--ignore-pypi-errors",
            "--ignore-source-errors",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=EXPORT_TIMEOUT_SECONDS,
            )
            reason = completed.stderr.strip()
        except OSError, subprocess.TimeoutExpired:
            reason = "the export failed to run or timed out."
        if not any(self._target_dir.glob(CONDA_SPECIFICATION_PATTERN)):
            warnings.warn(f"No conda specification archived. {reason}", stacklevel=2)

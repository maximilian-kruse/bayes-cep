"""The directories of runs and studies on disk, and the state of a run.

```
<study root>/                 StudyDirectory
  study/       study.json, environment.json,
               environment/ (pixi.lock, pyproject.toml, conda spec, source.patch)
  runs/<id>/                  RunDirectory: config.json, metadata.json, status.json,
                              run.log, metrics.json and the result files of the run
  summary/     written by `write_run_table`
  jobs/        submitit job folders and logs (SLURM only)
```

A run that was never submitted or started has no directory and counts as pending.

This module owns the layout above, the atomic writing of the JSON records, the repository root
against which configured paths are resolved, and the format of recorded times. The result files
that only a run itself produces are known to the run; `ArrayRecord` is the helper for naming and
storing them.

Constants:
    REPOSITORY_ROOT: Root of the repository.

Classes:
    RunState: The states of a run.
    ArrayRecord: Base class of a set of arrays that is stored as one `.npy` file per field.
    RunDirectory: One run's directory: its log, its JSON records, its configuration and state.
    StudyDirectory: A study's directory: its description and its run directories.

Functions:
    resolve_repository_path: Resolve a configured path against the repository root.
    repository_relative_path: Express a path relative to the repository root if it is inside it.
    module_relative_path: A path given relative to a module file, in its configuration form.
    format_current_time: The current time as an ISO 8601 string with time zone.
    write_json_record: Write a JSON file atomically.
"""

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from bayes_cep.run.config import JsonValue, RunConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RECORD = "config.json"
METADATA_RECORD = "metadata.json"
METRICS_RECORD = "metrics.json"
STATUS_RECORD = "status.json"
LOG_FILE_NAME = "run.log"
RESULTS_DIR_NAME = "results"


# ==================================================================================================
class RunState(StrEnum):
    """The state of a run; also its value in `status.json`.

    `SUBMITTED` is written by the submitter before a job may start, so that a queued run is told
    apart from one that was never submitted. `RUNNING` and `SUBMITTED` are the *active* states.
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_active(self) -> bool:
        """Whether a job may currently be working on the run (or has died while doing so)."""
        return self in (RunState.SUBMITTED, RunState.RUNNING)


# ==================================================================================================
def resolve_repository_path(path: Path) -> Path:
    """Resolve a relative path against the repository root, so runs do not depend on the working
    directory; absolute paths stay as they are."""
    return path if path.is_absolute() else REPOSITORY_ROOT / path


# --------------------------------------------------------------------------------------------------
def repository_relative_path(path: Path) -> Path:
    """Resolve `path` and express it relative to the repository root if it lies inside it.

    This is the form of a path in a configuration: independent of the checkout location, so that
    run ids (hashes of the configuration) are the same in another checkout, a worktree or on the
    cluster. Outside the repository there is no relative form, so the path is absolute.

    Args:
        path (Path): Absolute path, or relative to the working directory.

    Returns:
        Path: Relative to the repository root if inside it, else absolute.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT):
        return resolved.relative_to(REPOSITORY_ROOT)
    return resolved


# --------------------------------------------------------------------------------------------------
def module_relative_path(module_file: str | Path, relative_path: str | Path) -> Path:
    """A path given relative to a module file, in the form a configuration holds it.

    A bare relative path in a configuration is relative to the repository root
    (`resolve_repository_path`), whatever module writes it. A study module outside the repository
    uses this function for the input paths of its configurations (`RAW_DIR = module_relative_path(
    __file__, "raw")`), so that they are relative to the module and not to the repository.

    Args:
        module_file (str | Path): The module file, pass `__file__`.
        relative_path (str | Path): Path relative to the directory of the module file; may contain
            `..`. An absolute path is independent of the module, and is treated the same way:
            made repo-relative if it lies inside the repository, else it stays as it is.

    Returns:
        Path: Relative to the repository root if inside it, else absolute (see
            `repository_relative_path`).
    """
    return repository_relative_path(Path(module_file).resolve().parent / relative_path)


# --------------------------------------------------------------------------------------------------
def format_current_time() -> str:
    """The current time, ISO 8601 with time zone and second resolution."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------------
def write_json_record(path: Path, data: JsonValue) -> None:
    """Write `data` as JSON via a temporary file and a rename, so readers never see a partial file.

    Missing parent folders are created. The temporary file is named after the process, so that
    concurrent writers of the same record do not clobber each other's temporary file.

    Args:
        path (Path): Target file.
        data (JsonValue): Plain JSON data; numpy scalars and the like are not accepted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


# ==================================================================================================
@dataclass(frozen=True)
class ArrayRecord:
    """A set of arrays (and scalars) stored as one `.npy` file per field.

    A subclass is a frozen dataclass that names the file of each of its fields in `FILE_NAMES`
    (field name to file name). Saving, loading and the list of files follow from that mapping, so
    there is no second place where the order or the names of the files could differ. A scalar
    field is stored as a 0-d array and read back as a Python scalar.

    Attributes:
        FILE_NAMES (Mapping[str, str]): File name of each field.
    """

    FILE_NAMES: ClassVar[Mapping[str, str]]

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def file_paths(cls, directory: Path) -> list[Path]:
        """The files of the record in `directory`."""
        return [directory / file_name for file_name in cls.FILE_NAMES.values()]

    # ----------------------------------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        """Write the fields into `directory`, which is created if missing."""
        directory.mkdir(parents=True, exist_ok=True)
        for field_name, file_name in self.FILE_NAMES.items():
            np.save(directory / file_name, getattr(self, field_name))

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, directory: Path) -> Self:
        """Read the fields from `directory`."""
        values = {}
        for field_name, file_name in cls.FILE_NAMES.items():
            array = np.load(directory / file_name)
            values[field_name] = array.item() if array.ndim == 0 else array
        return cls(**values)


# ==================================================================================================
class RunDirectory:
    """The directory of one run; its records are directly in it.

    Attributes:
        path (Path): The directory; it need not exist yet.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the directory `path`."""
        self.path = path

    # ----------------------------------------------------------------------------------------------
    @property
    def log_path(self) -> Path:
        """The log file of the run."""
        return self.path / LOG_FILE_NAME

    # ----------------------------------------------------------------------------------------------
    def clear_for_new_attempt(self) -> None:
        """Remove everything of a previous attempt except the status record.

        The status is kept so that a submitted run stays recognizable as active until its job
        records the start.
        """
        if not self.path.exists():
            return
        for entry in self.path.iterdir():
            if entry.name == STATUS_RECORD:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    # ----------------------------------------------------------------------------------------------
    def record_submitted(self) -> None:
        """Record that a job for the run is queued and may start any time, as `SUBMITTED`."""
        self._write_state(RunState.SUBMITTED)

    # ----------------------------------------------------------------------------------------------
    def record_start(self, config: RunConfig, metadata: dict[str, Any]) -> None:
        """Record the configuration and metadata of a run that starts now, as `RUNNING`.

        The status starts afresh, and the metrics of any earlier attempt are removed, so that
        nothing of an earlier attempt can be mistaken for the result of this one.
        """
        (self.path / METRICS_RECORD).unlink(missing_ok=True)
        write_json_record(self.path / CONFIG_RECORD, config.to_json_dict())
        write_json_record(self.path / METADATA_RECORD, metadata)
        self._write_state(RunState.RUNNING, started=format_current_time())

    # ----------------------------------------------------------------------------------------------
    def record_success(self, metrics: dict[str, Any]) -> None:
        """Record the metrics and the end of a run that finished, as `DONE`."""
        write_json_record(self.path / METRICS_RECORD, metrics)
        self._write_state(
            RunState.DONE,
            started=self._read_state().get("started"),
            finished=format_current_time(),
        )

    # ----------------------------------------------------------------------------------------------
    def record_failure(self, error: BaseException, formatted_traceback: str) -> None:
        """Record that, and why, a run failed or was interrupted, as `FAILED`."""
        self._write_state(
            RunState.FAILED,
            started=self._read_state().get("started"),
            finished=format_current_time(),
            error=f"{type(error).__name__}: {error}",
            traceback=formatted_traceback,
        )

    # ----------------------------------------------------------------------------------------------
    def read_metrics(self) -> dict[str, Any]:
        """The recorded metrics; empty unless the run is `DONE`."""
        if self.read_state() != RunState.DONE:
            return {}
        return json.loads((self.path / METRICS_RECORD).read_text())

    # ----------------------------------------------------------------------------------------------
    def read_state(self) -> RunState:
        """The state of the run; `PENDING` if it has no status file."""
        status = self._read_state()
        return RunState(status["state"]) if status else RunState.PENDING

    # ----------------------------------------------------------------------------------------------
    def _read_state(self) -> dict[str, Any]:
        """The content of `status.json`; empty if there is none."""
        path = self.path / STATUS_RECORD
        return json.loads(path.read_text()) if path.exists() else {}

    # ----------------------------------------------------------------------------------------------
    def _write_state(self, state: RunState, **entries: Any) -> None:
        """Replace `status.json` by `state` and the entries that are set."""
        status = {"state": state.value}
        status.update({name: value for name, value in entries.items() if value is not None})
        write_json_record(self.path / STATUS_RECORD, status)


# ==================================================================================================
class StudyDirectory:
    """The directory of one study.

    Attributes:
        path (Path): The directory.
        study_record_path (Path): Description of the study and its run ids (`study/study.json`).
        environment_path (Path): The environment recorded at creation (`study/environment.json`).
        environment_dir (Path): The archived environment specification (`study/environment/`).
        summary_dir (Path): Where the cross-run summary is written.
        job_dir (Path): Where submitit writes its job folders and logs (SLURM jobs only).
    """

    def __init__(self, path: Path) -> None:
        """Wrap the study directory `path`."""
        self.path = path
        self.study_record_path = path / "study" / "study.json"
        self.environment_path = path / "study" / "environment.json"
        self.environment_dir = path / "study" / "environment"
        self.summary_dir = path / "summary"
        self.job_dir = path / "jobs"

    # ----------------------------------------------------------------------------------------------
    def run_directory(self, run_id: str) -> RunDirectory:
        """The directory of the run with this id."""
        return RunDirectory(self.path / "runs" / run_id)

"""The directories of runs and studies on disk, and the state of a run.

```
<study_name>/                 StudyDirectory
  study/       study.json, runs.json, environment.json, definition.py,
               environment/ (pixi.lock, pyproject.toml, conda spec, source.patch)
  runs/<id>/                  RunDirectory: config.json, metadata.json, status.json,
                              run.log, metrics.json and the result files of the run
  summary/     written by `write_run_table`
  jobs/        submitit job folders and logs (local and SLURM)
```

A run that was never submitted or started has no directory and counts as pending.

This module owns the layout above, the repository root and the atomic reading and writing of the
JSON records. The result files that only a run itself produces are known to the run.

Constants:
    REPOSITORY_ROOT: Root of the repository, against which configured paths are resolved.

Classes:
    RunState: The states of a run.
    RunDirectory: One run's directory: its log, its JSON records, its configuration and state.
    StudyDirectory: A study's directory: its description and its run directories.

Functions:
    write_json_record: Write a JSON file atomically.
    read_json_record: Read a JSON file.
    resolve_repository_path: Resolve a configured path against the repository root.
"""

import json
import os
import shutil
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from bayes_cep.run.config import RunConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RECORD = "config.json"
METADATA_RECORD = "metadata.json"
METRICS_RECORD = "metrics.json"
STATUS_RECORD = "status.json"
LOG_FILE_NAME = "run.log"


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


# ==================================================================================================
def _to_builtin(value: object) -> object:
    """`json.dumps` fallback: numpy scalars (e.g. in metrics) as Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


# --------------------------------------------------------------------------------------------------
def write_json_record(path: Path, data: Any) -> None:
    """Write `data` as JSON via a temporary file and a rename, so readers never see a partial file.

    Missing parent folders are created; numpy scalars are written as Python scalars.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_to_builtin) + "\n")
    os.replace(temporary, path)


# --------------------------------------------------------------------------------------------------
def read_json_record(path: Path) -> Any:
    """Read the JSON file `path`."""
    return json.loads(path.read_text())


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
        self._write_state(RunState.RUNNING, started=_now())

    # ----------------------------------------------------------------------------------------------
    def record_success(self, metrics: dict[str, Any]) -> None:
        """Record the metrics and the end of a run that finished, as `DONE`."""
        write_json_record(self.path / METRICS_RECORD, metrics)
        self._write_state(RunState.DONE, started=self._read_status().get("started"))

    # ----------------------------------------------------------------------------------------------
    def record_failure(self, error: BaseException, formatted_traceback: str) -> None:
        """Record that, and why, a run failed or was interrupted, as `FAILED`."""
        self._write_state(
            RunState.FAILED,
            started=self._read_status().get("started"),
            error=f"{type(error).__name__}: {error}",
            traceback=formatted_traceback,
        )

    # ----------------------------------------------------------------------------------------------
    def read_metrics(self) -> dict[str, Any]:
        """The recorded metrics; empty unless the run is `DONE`."""
        if self.read_state() != RunState.DONE:
            return {}
        return read_json_record(self.path / METRICS_RECORD)

    # ----------------------------------------------------------------------------------------------
    def read_state(self) -> RunState:
        """The state of the run; `PENDING` if it has no status file."""
        status = self._read_status()
        return RunState(status["state"]) if status else RunState.PENDING

    # ----------------------------------------------------------------------------------------------
    def _read_status(self) -> dict[str, Any]:
        """The content of `status.json`; empty if there is none."""
        path = self.path / STATUS_RECORD
        return read_json_record(path) if path.exists() else {}

    # ----------------------------------------------------------------------------------------------
    def _write_state(self, state: RunState, **entries: Any) -> None:
        """Replace `status.json` by `state`, the current time and the entries that are set."""
        status = {"state": state.value, "updated": _now()}
        status.update({name: value for name, value in entries.items() if value is not None})
        write_json_record(self.path / STATUS_RECORD, status)


# --------------------------------------------------------------------------------------------------
def _now() -> str:
    """The current time, ISO 8601 with time zone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ==================================================================================================
class StudyDirectory:
    """The directory of one study.

    Attributes:
        path (Path): The directory.
        description_dir (Path): The records describing the study, in `study/`.
        study_record_path (Path): Human-readable description of the study (`study.json`).
        run_index_path (Path): The list of all runs with their configurations (`runs.json`).
        environment_path (Path): The environment recorded at creation (`environment.json`).
        definition_path (Path): The archived study module.
        environment_dir (Path): The archived environment specification.
        summary_dir (Path): Where the cross-run summary is written.
        job_dir (Path): Where submitit writes its job folders and logs (local and SLURM jobs).
    """

    def __init__(self, path: Path) -> None:
        """Wrap the study directory `path`."""
        self.path = path
        self.description_dir = path / "study"
        self.study_record_path = self.description_dir / "study.json"
        self.run_index_path = self.description_dir / "runs.json"
        self.environment_path = self.description_dir / "environment.json"
        self.definition_path = self.description_dir / "definition.py"
        self.environment_dir = self.description_dir / "environment"
        self.summary_dir = path / "summary"
        self.job_dir = path / "jobs"

    # ----------------------------------------------------------------------------------------------
    def run_directory(self, run_id: str) -> RunDirectory:
        """The directory of the run with this id."""
        return RunDirectory(self.path / "runs" / run_id)

    # ----------------------------------------------------------------------------------------------
    def read_run_index(self) -> list[dict[str, Any]]:
        """The run list recorded at creation: `index`, `id`, `overrides` and `config` per run."""
        return read_json_record(self.run_index_path)

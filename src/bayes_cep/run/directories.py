"""The directories of runs and studies on disk, and the state of a run.

```
<study_name>/                 StudyDirectory
  study/       study.json, runs.json, environment.json, definition.py
  runs/<id>/                  RunDirectory: config.json, metadata.json, status.json,
                              run.log, metrics.json and the result files of the run
  summary/     written by the collector
  slurm/       submitit job folders and logs
```

A run that has not started has no directory and counts as pending.

Constants:
    PENDING, RUNNING, DONE, FAILED: The run states.

Classes:
    RunDirectory: One run's directory: its JSON records and its state.
    StudyDirectory: A study's directory: its description and its run directories.

Functions:
    write_json_atomic: Write a JSON file so that readers never see a partial file.
    read_json: Read a JSON file.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"


# ==================================================================================================
def write_json_atomic(path: Path, data: Any) -> None:
    """Write `data` as JSON, via a temporary file and an atomic rename.

    Args:
        path (Path): Target file; missing parent directories are created.
        data (Any): JSON-serializable data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary_path, path)


# ==================================================================================================
def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text())


# ==================================================================================================
class RunDirectory:
    """The directory of one run.

    Attributes:
        path (Path): The directory; it need not exist yet.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the directory `path`."""
        self.path = path

    # ----------------------------------------------------------------------------------------------
    def write(self, name: str, data: Any) -> None:
        """Write a JSON record `name` of the run, e.g. `"config.json"`, atomically."""
        write_json_atomic(self.path / name, data)

    # ----------------------------------------------------------------------------------------------
    def read(self, name: str) -> Any:
        """Read a JSON record of the run."""
        return read_json(self.path / name)

    # ----------------------------------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        """The recorded metrics; empty if the run has none (pending or failed)."""
        return self.read("metrics.json") if (self.path / "metrics.json").exists() else {}

    # ----------------------------------------------------------------------------------------------
    def state(self) -> str:
        """The state of the run; `PENDING` if it has no status file."""
        if not (self.path / "status.json").exists():
            return PENDING
        return str(self.read("status.json")["state"])

    # ----------------------------------------------------------------------------------------------
    def set_status(self, state: str, **entries: Any) -> None:
        """Record `state`, the current time and any further entries in `status.json`.

        Entries of an existing status file are kept, so the start time survives the final update.
        """
        exists = (self.path / "status.json").exists()
        status = self.read("status.json") if exists else {}
        status.update(
            state=state, updated=datetime.now().astimezone().isoformat(timespec="seconds")
        )
        status.update(entries)
        self.write("status.json", status)


# ==================================================================================================
class StudyDirectory:
    """The directory of one study.

    Attributes:
        path (Path): The directory.
        summary_dir (Path): Where the collector writes.
        slurm_dir (Path): Where submitit writes its job folders and logs.
        definition_path (Path): The archived study module.
    """

    def __init__(self, path: Path) -> None:
        """Wrap the study directory `path`."""
        self.path = path
        self.summary_dir = path / "summary"
        self.slurm_dir = path / "slurm"
        self.definition_path = path / "study" / "definition.py"

    # ----------------------------------------------------------------------------------------------
    def run(self, run_id: str) -> RunDirectory:
        """The directory of the run with this id."""
        return RunDirectory(self.path / "runs" / run_id)

    # ----------------------------------------------------------------------------------------------
    def write(self, name: str, data: Any) -> None:
        """Write a JSON record `name` of the study description (in `study/`), atomically."""
        write_json_atomic(self.path / "study" / name, data)

    # ----------------------------------------------------------------------------------------------
    def read(self, name: str) -> Any:
        """Read a JSON record of the study description."""
        return read_json(self.path / "study" / name)

    # ----------------------------------------------------------------------------------------------
    def run_index(self) -> list[dict[str, Any]]:
        """The run list recorded at creation: `index`, `id`, `overrides` and `config` per run."""
        return self.read("runs.json")

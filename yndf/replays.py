import datetime as dt
import os
import pickle
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


# --------------------------
# Replay step representation
# --------------------------

@dataclass(slots=True, frozen=True)
class StepRecord:
    """
    One recorded step of a NetHack session.

    Fields:
        action: integer action that was taken (keypress).
        observation: observation from the NetHack env (raw).
        info: info dict from the NetHack env (raw).
    """
    action: int
    observation: dict[str, Any] | None
    info: dict[str, Any] | None

def _enum_name_or_value(val: Any) -> Any:
    """Convert Enum to its name, otherwise return value unchanged."""
    return val.name if isinstance(val, Enum) else val


def _sanitize_top_enums(dct: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """
    Copy a dict and convert *top-level* Enum values to their names (do not recurse).
    Returns None if input is None.
    """
    if dct is None:
        return None
    return {k: _enum_name_or_value(v) for k, v in dct.items() if k != "state"}


def _sanitize_step_for_pickle(step: StepRecord) -> StepRecord:
    """
    Return a new StepRecord with top-level Enum values converted to names in any dict fields.
    Original is not mutated.
    """
    return StepRecord(
        action=int(step.action) if step.action is not None else None,
        observation=_sanitize_top_enums(step.observation) or {},
        info=_sanitize_top_enums(step.info) or {},
    )

# --------------------------
# Safe, atomic replay writer
# --------------------------

def save_replay(replay_dir: str | os.PathLike[str], steps: Iterable[StepRecord]) -> str:
    """
    Atomically save a finished replay (list of StepRecord) to a unique file under replay_dir.

    - Converts top-level Enums in dict fields to their names (without touching originals).
    - Writes to a temp file in the same directory, fsyncs, and os.replace()s atomically.
    - Filename includes timestamp + pid + thread id. Safe for many concurrent writers.

    Returns the final path (string).
    """
    path = Path(replay_dir)
    path.mkdir(parents=True, exist_ok=True)

    sanitized = [_sanitize_step_for_pickle(s) for s in steps]

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    pid = os.getpid()
    tid = threading.get_ident()
    final_name = f"replay-{ts}-{pid}-{tid}.pkl"
    final_path = path / final_name

    fd, tmp_str = tempfile.mkstemp(prefix=f".{final_name}.", suffix=".tmp", dir=path)
    tmp_path = Path(tmp_str)

    try:
        with os.fdopen(fd, "wb") as fobj:
            pickle.dump(sanitized, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            fobj.flush()
            os.fsync(fobj.fileno())
        os.replace(tmp_path, final_path)

        # Best-effort fsync on the directory entry
        try:
            dir_fd = os.open(path, os.O_DIRECTORY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    try:
        # -k keep original .pkl; -f overwrite existing .bz2 if present; '--' ends options
        proc = subprocess.Popen(
            ["bzip2", "--", str(final_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # detach from our process group/terminal
            close_fds=True,          # don't leak FDs into the compressor
        )

        # Reap the child in a daemon thread so it doesn't linger as a zombie
        threading.Thread(target=proc.wait, daemon=True).start()

    except FileNotFoundError:
        # bzip2 not installed; silently skip background compression
        pass
    except Exception: # pylint: disable=broad-exception-caught
        # Swallow errors to avoid affecting the caller’s flow
        pass

    return str(final_path)

"""Atomic receipts, OS-lifetime locks, and append-only process attempts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


class LockBusy(RuntimeError):
    pass


class RunLock:
    """The file can remain; the byte-range lock lives only with its process."""
    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        os.set_inheritable(self.stream.fileno(), False)
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self.stream.close()
            raise LockBusy(f"active process owns lock: {self.path}") from error
        return self

    def __exit__(self, *args):
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


def replace_file(source: Path, destination: Path):
    # Windows readers/virus scanners may briefly hold a non-delete-sharing handle.
    # Retrying preserves atomicity; never fall back to truncating the live file.
    for retry in range(30):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if retry == 29:
                raise
            time.sleep(min(0.005 * 2 ** min(retry, 5), 0.1))


def read_json(path: Path):
    """Read an atomic status, tolerating transient Windows sharing conflicts."""
    for retry in range(30):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError):
            if retry == 29:
                raise
            time.sleep(min(0.005 * 2 ** min(retry, 5), 0.1))


class Attempt:
    def __init__(self, directory: Path, latest_status: Path, role: str, mirrors=None, **metadata):
        self.attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        self.path = directory / self.attempt_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.latest_status = latest_status
        self.mirrors = list(mirrors or [])
        self.previous_latest = read_json(latest_status) if latest_status.exists() else None
        self.started = time.perf_counter()
        self.value = {"attempt_id": self.attempt_id, "role": role, "pid": os.getpid(),
                      "interpreter": sys.executable, "started_utc": utc_now(), **metadata}
        atomic_json(self.path / "started.json", self.value)
        self.update("running")

    def update(self, state: str, **extra):
        self.value.update(state=state, updated_utc=utc_now(), **extra)
        atomic_json(self.path / "status.json", self.value)
        atomic_json(self.latest_status, self.value)
        for path in self.mirrors:
            atomic_json(path, self.value)

    def finish(self, exit_status: int, **extra):
        self.update("complete" if exit_status == 0 else "failed", exit_status=exit_status,
                    elapsed_this_attempt_seconds=time.perf_counter() - self.started, finished_utc=utc_now(), **extra)
        atomic_json(self.path / "finished.json", self.value)

    def fail(self, error: BaseException):
        self.finish(1, exception=repr(error), traceback=traceback.format_exc())

    def refuse(self, reason: str):
        # A worker that loses the startup race returns 3. Preserve the old active
        # task's latest status and record only this candidate attempt as refused.
        self.value.update(state="dispatch_refused", exit_status=3, reason=reason, finished_utc=utc_now(),
                          elapsed_this_attempt_seconds=time.perf_counter() - self.started)
        atomic_json(self.path / "status.json", self.value)
        atomic_json(self.path / "finished.json", self.value)
        atomic_json(self.latest_status, self.previous_latest if self.previous_latest is not None else self.value)


def ensure_no_worker(run_dir: Path):
    # Also checked by the worker itself, which closes the parent-death/start race.
    with RunLock(run_dir / "worker.lock"):
        pass


def run_child(command: list[str], cwd: Path, attempt: Attempt, label: str, worker_status=None, dispatch_id=None) -> int:
    stdout = attempt.path / f"{label}.stdout.log"
    stderr = attempt.path / f"{label}.stderr.log"
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        child = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 env=env, creationflags=flags, close_fds=True)
        attempt.update("running", worker_launch_pid=child.pid, worker_pid=None, worker_command=command,
                       worker_stdout=str(stdout), worker_stderr=str(stderr), current_step=label)
        # Windows venv redirectors can have a different PID from the real Python
        # worker. Bind its self-reported PID to this unique dispatch, not old PID files.
        if worker_status is not None:
            deadline = time.monotonic() + 20
            while child.poll() is None and time.monotonic() < deadline:
                if worker_status.exists():
                    value = read_json(worker_status)
                    if value.get("dispatch_id") == dispatch_id:
                        attempt.update("running", worker_pid=value["pid"], worker_attempt=value["attempt_id"])
                        break
                time.sleep(0.05)
        code = child.wait()
    atomic_json(attempt.path / f"{label}.exit.json", {"launch_process_pid": child.pid,
                "actual_python_pid": attempt.value.get("worker_pid"), "command": command,
                "exit_status": code, "finished_utc": utc_now(), "stdout": str(stdout), "stderr": str(stderr)})
    if code == 3:
        attempt.refuse("worker lifetime lock refused the dispatch; an existing worker is left running")
    else:
        attempt.update("running", worker_exit_status=code, worker_pid=None)
    if code not in (0, 3):
        attempt.finish(code, stage="worker_failed", failed_step=label)
    return code

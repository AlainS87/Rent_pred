import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from src.backtest.execution import Attempt, LockBusy, RunLock, atomic_json, ensure_no_worker, read_json, run_child
from src.backtest.protocol import ROOT

HELPER = Path(__file__).with_name("process_helper.py").resolve()


def command(mode, root, *extra):
    return [sys.executable, str(HELPER), mode, "--root", str(root), *extra]


def spawn(arguments):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(arguments, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)


def ready(path):
    deadline = time.monotonic() + 20
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.02)


def cli(module, root):
    arguments = [sys.executable, "-m", module]
    if module.endswith("supervisor"):
        arguments += ["run", "--resume"]
    else:
        arguments += ["--phase", "development", "--origin", "2022-02"]
    arguments += ["--run-dir", str(root), "--manifest-hash", "fixture"]
    return subprocess.run(arguments, cwd=ROOT, capture_output=True, timeout=30)


@pytest.mark.parametrize("name", ["supervisor", "worker"])
def test_process_lock_contention_does_not_overwrite_running_status(checkpoint_path, name):
    root = checkpoint_path
    atomic_json(root / "frozen_protocol.json", {})
    atomic_json(root / "status.json", {"state": "running", "sentinel": "original"})
    atomic_json(root / "worker_status.json", {"state": "running", "sentinel": "original worker"})
    original = (root / "status.json").read_bytes(), (root / "worker_status.json").read_bytes()
    process = spawn(command("hold", root, "--name", name))
    try:
        ready(root / f"{name}.ready.json")
        module = "src.backtest.supervisor" if name == "supervisor" else "src.backtest.worker"
        result = cli(module, root)
        assert result.returncode == 3, result.stderr.decode(errors="replace")
        assert original == ((root / "status.json").read_bytes(), (root / "worker_status.json").read_bytes())
        assert not (root / "attempts").exists()
    finally:
        (root / "release").touch()
        process.communicate(timeout=30)


def test_crash_releases_lifetime_lock_without_pid_heuristics(checkpoint_path):
    root = checkpoint_path
    atomic_json(root / "old_pid.json", {"pid": os.getpid()})  # live but unrelated PID is irrelevant
    process = spawn(command("hold", root, "--name", "supervisor", "--crash"))
    try:
        ready(root / "supervisor.ready.json")
        with pytest.raises(LockBusy):
            with RunLock(root / "supervisor.lock"):
                pass
    finally:
        (root / "release").touch()
        process.communicate(timeout=30)
    assert process.returncode == 17
    with RunLock(root / "supervisor.lock"):
        pass


def test_dead_parent_with_live_orphan_worker_refuses_redispatch(checkpoint_path):
    root = checkpoint_path
    atomic_json(root / "frozen_protocol.json", {})
    atomic_json(root / "status.json", {"state": "running", "sentinel": "do not replace"})
    original = (root / "status.json").read_bytes()
    parent = spawn(command("orphan", root))
    try:
        parent.communicate(timeout=30)
        assert parent.returncode == 19
        ready(root / "worker.ready.json")
        with RunLock(root / "supervisor.lock"):
            with pytest.raises(LockBusy):
                ensure_no_worker(root)
        result = cli("src.backtest.supervisor", root)
        assert result.returncode == 3
        assert (root / "status.json").read_bytes() == original
        assert not (root / "attempts").exists()
    finally:
        (root / "release").touch()
        deadline = time.monotonic() + 20
        while True:
            try:
                ensure_no_worker(root)
                break
            except LockBusy:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)


def test_nonzero_worker_exit_is_failed_and_attempt_history_is_preserved(checkpoint_path):
    root = checkpoint_path
    attempt = Attempt(root / "attempts", root / "status.json", "supervisor_test")
    code = run_child(command("fail", root), ROOT, attempt, "intentional_failure")
    assert code == 7
    first = (attempt.path / "finished.json").read_bytes()
    assert json.loads(first)["state"] == "failed"
    assert json.loads(first)["exit_status"] == 7
    assert "intentional stderr" in (attempt.path / "intentional_failure.stderr.log").read_text()
    assert json.loads((attempt.path / "intentional_failure.exit.json").read_text())["exit_status"] == 7
    resumed = Attempt(root / "attempts", root / "status.json", "supervisor_test", resumed=True)
    resumed.finish(0)
    assert resumed.path != attempt.path
    assert (attempt.path / "finished.json").read_bytes() == first
    assert json.loads((root / "status.json").read_text())["state"] == "complete"


def test_atomic_status_and_complete_are_readable_during_process_writes(checkpoint_path):
    root = checkpoint_path
    process = spawn(command("atomic", root))
    reads = 0
    while process.poll() is None:
        path = root / "atomic_status.json"
        if path.exists():
            value = read_json(path)
            assert 0 <= value["sequence"] <= 99
            assert value["payload"] == "x" * 2000
            reads += 1
        time.sleep(0.001)
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, stderr.decode(errors="replace")
    assert reads > 1
    assert json.loads((root / "atomic_complete.json").read_text()) == {"exit_status": 0, "sequence": 99}


def test_worker_start_race_is_refusal_not_training_failure_and_restores_prior_status(checkpoint_path):
    root = checkpoint_path
    atomic_json(root / "frozen_protocol.json", {})
    prior = {"state": "running", "sentinel": "existing task status"}
    atomic_json(root / "status.json", prior)
    atomic_json(root / "worker_status.json", {"sentinel": "existing worker"})
    with RunLock(root / "worker.lock"):
        attempt = Attempt(root / "attempts/candidate", root / "status.json", "candidate_supervisor")
        arguments = [sys.executable, "-m", "src.backtest.worker", "--run-dir", str(root),
                     "--manifest-hash", "fixture", "--phase", "development", "--origin", "2022-02"]
        assert run_child(arguments, ROOT, attempt, "contended_worker") == 3
    assert read_json(root / "status.json") == prior
    assert read_json(root / "worker_status.json") == {"sentinel": "existing worker"}
    assert read_json(attempt.path / "finished.json")["state"] == "dispatch_refused"

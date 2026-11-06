"""Short-lived subprocess fixtures. Never imports or fits rental models."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.backtest.execution import Attempt, RunLock, atomic_json


def wait_for(path, timeout=20):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hold", "orphan", "fail", "atomic"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", default="worker")
    parser.add_argument("--crash", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    if args.mode == "hold":
        with RunLock(root / f"{args.name}.lock"):
            atomic_json(root / f"{args.name}.ready.json", {"pid": os.getpid()})
            wait_for(root / "release", timeout=25)
            if args.crash:
                os._exit(17)
        return 0
    if args.mode == "orphan":
        with RunLock(root / "supervisor.lock"):
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "hold", "--root", str(root)],
                             cwd=str(Path(__file__).resolve().parents[3]), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True, creationflags=flags)
            wait_for(root / "worker.ready.json")
            os._exit(19)
    if args.mode == "fail":
        attempt = Attempt(root / "child_attempts", root / "child_status.json", "test_child")
        attempt.finish(7, exception="intentional process fixture failure")
        print("intentional stdout", flush=True)
        print("intentional stderr", file=sys.stderr, flush=True)
        return 7
    for sequence in range(100):
        atomic_json(root / "atomic_status.json", {"sequence": sequence, "payload": "x" * 2000})
        time.sleep(0.002)
    atomic_json(root / "atomic_complete.json", {"exit_status": 0, "sequence": 99})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

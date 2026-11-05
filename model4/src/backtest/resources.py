"""Low-overhead runtime/Windows memory observations, without affecting other jobs."""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
import threading
import time


def memory_mib() -> dict:
    if os.name != "nt":
        return {}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
            "PeakPagefileUsage", "PrivateUsage"]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"rss_mib": counters.WorkingSetSize / 2**20, "process_peak_rss_mib": counters.PeakWorkingSetSize / 2**20,
            "private_mib": counters.PrivateUsage / 2**20}


class ResourceMeter:
    def __enter__(self):
        self.start = time.perf_counter()
        self.stop = threading.Event()
        self.samples = [memory_mib()]
        def observe():
            while not self.stop.wait(0.1):
                self.samples.append(memory_mib())
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.samples.append(memory_mib())
        self.result = {"elapsed_seconds": time.perf_counter() - self.start,
                       "start": self.samples[0], "end": self.samples[-1],
                       "observed_peak_rss_mib": max((s.get("rss_mib", 0) for s in self.samples), default=0),
                       "sampling_interval_seconds": 0.1}


def runtime_metadata() -> dict:
    packages = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib", "catboost", "pytest"]
    return {"python": platform.python_version(), "executable": os.sys.executable,
            "platform": platform.platform(), "logical_cpus": os.cpu_count(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}

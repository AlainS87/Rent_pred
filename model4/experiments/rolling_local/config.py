"""Standard-library-only configuration validation, safe for CLI help."""
import hashlib
import json
from pathlib import Path


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    for key in ("raw_data", "feature_data", "origins", "budget", "resources"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")
    origins = config["origins"]
    if not isinstance(origins, list) or not origins or len(origins) != len(set(origins)):
        raise ValueError("Origins must be a nonempty, unique, ordered list")
    import re
    if any(not isinstance(v, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", v) for v in origins):
        raise ValueError("Origins must use YYYY-MM")
    if config.get("inner_months", 3) != 3 or config.get("forecast_months", 16) != 16:
        raise ValueError("This fixed protocol uses three inner months and sixteen forecast months")
    wall, reserve = config["budget"].get("wall_seconds"), config["budget"].get("report_reserve_seconds")
    if not isinstance(wall, int) or not isinstance(reserve, int) or not 0 < reserve < wall:
        raise ValueError("Budget requires positive wall_seconds and smaller report reserve")
    threads = config["resources"].get("cpu_threads")
    if not isinstance(threads, int) or not 1 <= threads <= 8 or config["resources"].get("device", "cpu") != "cpu":
        raise ValueError("The local protocol is CPU-only, with 1..8 assigned logical CPUs")
    for key in ("raw_data", "feature_data"):
        config[key] = str((path.parent / config[key]).resolve())
    for origin, rows in config.get("expected_outer_rows", {}).items():
        if origin not in origins or not isinstance(rows, int) or rows <= 0:
            raise ValueError("Invalid expected outer row count")
    config["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config

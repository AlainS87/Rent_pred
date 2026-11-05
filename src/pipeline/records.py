"""Logs, the champion pointer, and the generated design note."""

import json
from pathlib import Path

from src.pipeline.paths import MODELS, rel

LOG_COMMANDS = {"train", "evolve"}


def model_dir(model: str) -> Path:
    return MODELS / model


def logs_dir(model: str) -> Path:
    return model_dir(model) / "docs" / "logs"


def design_path(model: str) -> Path:
    return model_dir(model) / "docs" / "design.md"


def champion_path(model: str) -> Path:
    return model_dir(model) / "docs" / "champion.json"


def read_logs(model: str) -> list[dict]:
    folder = logs_dir(model)
    if not folder.exists():
        return []
    records = []
    for path in sorted(folder.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def write_log(model: str, record: dict) -> Path:
    folder = logs_dir(model)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record['run_id']}.json"
    payload = {key: value for key, value in record.items() if key != "trial"}
    _dump(path, payload)
    return path


def load_champion(model: str) -> dict | None:
    path = champion_path(model)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_champion(model: str, champion: dict) -> None:
    path = champion_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    _dump(path, champion)


def refresh_design(model: str) -> None:
    """Rewrite design.md from the model note, the rule text, and the logs."""
    import importlib

    module = importlib.import_module(f"models.{model}.model")
    from src.pipeline.rules import rules_text

    logs = read_logs(model)
    champion = load_champion(model)
    text = "\n".join(
        [
            f"# {module.TITLE}",
            "",
            module.MODEL_NOTE.strip(),
            "",
            rules_text().strip(),
            "",
            "## Current champion",
            "",
            "<!-- champion:start -->",
            _champion_section(champion).strip(),
            "<!-- champion:end -->",
            "",
            "## Lineage",
            "",
            "<!-- lineage:start -->",
            _lineage_section(logs).strip(),
            "<!-- lineage:end -->",
            "",
            "## Predictions",
            "",
            "<!-- predictions:start -->",
            _prediction_section(logs).strip(),
            "<!-- predictions:end -->",
            "",
        ]
    )
    path = design_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _champion_section(champion: dict | None) -> str:
    if champion is None:
        return "No champion yet.\n"
    genome = champion["genome"]
    cap = champion.get("max_rows")
    cap_line = str(cap) if cap else "full training table"
    confirm = champion.get("confirm_rmse")
    confirm_line = "not scored" if confirm is None else f"{confirm:.6f}"
    confirm_log = champion.get("confirm_log") or "not scored"
    return "\n".join(
        [
            f"- Run: `{champion['run_id']}`",
            f"- Log: `{champion['log_path']}`",
            f"- Feature hash: `{champion['feature_hash']}`",
            f"- Genome hash: `{champion['genome_hash']}`",
            f"- Features: {_columns(genome['features'])}",
            f"- Preprocess: {_preprocess(genome['preprocess'])}",
            f"- Architecture: `{json.dumps(genome['architecture'], sort_keys=True)}`",
            f"- Train RMSE: {champion['train_rmse']:.6f}",
            f"- Search RMSE: {champion['search_rmse']:.6f}",
            f"- Confirm RMSE: {confirm_line}",
            f"- Confirm log: `{confirm_log}`",
            f"- Row cap: {cap_line}",
            "",
        ]
    )


def _lineage_section(logs: list[dict]) -> str:
    fit_logs = [record for record in logs if record.get("command") in LOG_COMMANDS]
    if not fit_logs:
        return "No training runs yet.\n"
    blocks = [
        "Regenerated from `docs/logs`. Do not edit the metrics in this section by hand.",
        "",
    ]
    for record in fit_logs:
        genome = record["genome"]
        decision = "accepted" if record.get("accepted") else "rejected"
        reason = record.get("reason", "")
        if not reason.lower().startswith(decision):
            reason = f"{decision}. {reason}"
        lines = [
            f"### `{record['run_id']}`",
            "",
            f"- Command: {record['command']}",
            f"- Parent: `{record.get('parent_run_id')}`",
            f"- Generation: {record.get('generation')}",
            f"- Mutation: `{json.dumps(record.get('mutation'), sort_keys=True)}`",
            f"- Feature hash: `{record['feature_hash']}`",
            f"- Genome hash: `{record['genome_hash']}`",
            f"- Features: {_columns(genome['features'])}",
            f"- Preprocess: {_preprocess(genome['preprocess'])}",
            f"- Architecture: `{json.dumps(genome['architecture'], sort_keys=True)}`",
            f"- Train RMSE: {record['train_rmse']:.6f}",
        ]
        if record.get("search_rmse") is not None:
            lines.append(f"- Search RMSE: {record['search_rmse']:.6f}")
        lines.extend(
            [
                f"- Decision: {reason}",
                f"- Log: `{record['log_path']}`",
                "",
            ]
        )
        blocks.extend(lines)
    return "\n".join(blocks)


def _prediction_section(logs: list[dict]) -> str:
    rows = [record for record in logs if record.get("command") == "predict"]
    if not rows:
        return "No prediction runs yet.\n"
    blocks = []
    for record in rows:
        blocks.extend(
            [
                f"### `{record['run_id']}`",
                "",
                f"- Champion run: `{record.get('champion_run_id')}`",
                f"- Rows: {record.get('n')}",
                f"- Mean: {record.get('prediction_mean')}",
                f"- Std: {record.get('prediction_std')}",
                f"- Min: {record.get('prediction_min')}",
                f"- Max: {record.get('prediction_max')}",
                f"- Output: `{record.get('output')}`",
                f"- Refit on all labeled months: {record.get('refit_all')}",
                f"- Log: `{record['log_path']}`",
                "",
            ]
        )
    return "\n".join(blocks)


def _columns(columns: list[str]) -> str:
    return ", ".join(f"`{column}`" for column in columns)


def _preprocess(spec: dict) -> str:
    parts = [f"{name}={spec[name]}" for name in ("numeric_impute", "scale", "categorical", "rare_min_count")]
    return ", ".join(parts)


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log_path_for(model: str, run_id: str) -> str:
    return rel(logs_dir(model) / f"{run_id}.json")

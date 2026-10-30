"""Read logs and propose one-axis changes to features, preprocess, and architecture."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import FAMILY_COLUMNS, ORIGINAL_COLUMNS
from src.pipeline.data import TimeSplit, load_joined, time_split
from src.pipeline.fit import fit_genome, load_run, predict_frame, rmse, save_run
from src.pipeline.genome import apply_epochs, feature_hash, genome_hash, load_default_genome, seed_genome
from src.pipeline.records import (
    design_path,
    load_champion,
    log_path_for,
    read_logs,
    refresh_design,
    write_champion,
    write_log,
)
from src.pipeline.rules import OVERFIT_GAP, PERMUTATION_ROWS, PREPROCESS_KNOBS, SEARCH_GAIN

FAMILIES = {"original": list(ORIGINAL_COLUMNS), **{name: list(cols) for name, cols in FAMILY_COLUMNS.items()}}
AXIS_RANK = {"features": 0, "preprocess": 1, "architecture": 2}


def evolve_model(model: str, pool: list[str], generations: int, val_months: int, max_rows: int | None, epochs: int | None, seed: int) -> dict:
    if generations < 1:
        raise SystemExit("--generations must be at least 1")
    frame = load_joined("train", pool)
    split = time_split(frame, val_months, max_rows)
    logs = read_logs(model)
    seen = _seen_hashes(logs)
    champion, trial = _starting_champion(
        model, pool, epochs, seed, split, logs, seen, max_rows, val_months
    )
    print(
        f"champion {champion['run_id']} search RMSE {champion['search_rmse']:.4f} "
        f"features {len(champion['genome']['features'])}"
    )
    for generation in range(1, generations + 1):
        proposals = propose(
            model,
            champion,
            trial,
            split,
            pool,
            generation,
            logs,
            seen,
            seed,
        )
        fresh = [(genome, mutation) for genome, mutation in proposals if genome_hash(genome) not in seen]
        if not fresh:
            print(f"generation {generation}: every proposed mutant is already in the logs")
            break
        fitted = []
        for genome, mutation in fresh:
            fitted.append(
                _fit_record(
                    model,
                    genome,
                    split,
                    seed,
                    command="evolve",
                    parent_run_id=champion["run_id"],
                    generation=generation,
                    mutation=mutation,
                    pool=pool,
                    max_rows=max_rows,
                    val_months=val_months,
                    accepted=False,
                    reason="scored this generation",
                )
            )
            seen.add(fitted[-1]["genome_hash"])
            logs.append(fitted[-1])
        best = min(fitted, key=_rank)
        limit = champion["search_rmse"] * (1.0 - SEARCH_GAIN)
        won = best["search_rmse"] < limit
        for record in fitted:
            if won and record["run_id"] == best["run_id"]:
                record["accepted"] = True
                record["reason"] = (
                    f"accepted because search RMSE {record['search_rmse']:.6f} is at least "
                    f"{SEARCH_GAIN:.0%} below the parent search RMSE {champion['search_rmse']:.6f}"
                )
            elif record["run_id"] == best["run_id"]:
                record["accepted"] = False
                record["reason"] = (
                    f"rejected because search RMSE {record['search_rmse']:.6f} is not at least "
                    f"{SEARCH_GAIN:.0%} below the parent search RMSE {champion['search_rmse']:.6f}"
                )
            else:
                record["accepted"] = False
                record["reason"] = (
                    f"rejected because search RMSE {record['search_rmse']:.6f} is worse than "
                    f"the best mutant in this generation ({best['search_rmse']:.6f})"
                )
            write_log(model, record)
            _rewrite_metrics(model, record)
        refresh_design(model)
        if not won:
            print(f"generation {generation}: no mutant beat the champion by {SEARCH_GAIN:.0%}")
            break
        champion = _champion_from_record(best, max_rows)
        write_champion(model, champion)
        trial = best["trial"]
        print(f"generation {generation}: accepted {champion['run_id']} search RMSE {champion['search_rmse']:.4f}")
    confirm_champion(model, champion, split, trial)
    refresh_design(model)
    return champion


def train_model(
    model: str,
    features: list[str],
    val_months: int,
    max_rows: int | None,
    epochs: int | None,
    seed: int,
    genome_file: str = "",
    locked: dict | None = None,
    commit_best: bool = True,
    reason: str | None = None,
    mutation: dict | None = None,
    parent_run_id: str | None = None,
) -> dict:
    """Fit one frozen genome. Replace the best snapshot only when ``commit_best`` is set."""
    default = load_default_genome(model)
    source = locked or default
    genome = {
        "features": list(features),
        "preprocess": dict(source["preprocess"]),
        "architecture": json.loads(json.dumps(source["architecture"])),
    }
    genome = apply_epochs(genome, epochs, model)
    if genome_file:
        overlay = json.loads(Path(genome_file).read_text(encoding="utf-8"))
        if "preprocess" in overlay:
            genome["preprocess"].update(overlay["preprocess"])
        if "architecture" in overlay:
            genome["architecture"].update(overlay["architecture"])
        from src.pipeline.genome import _require_genome

        _require_genome(genome)
    frame = load_joined("train", features)
    split = time_split(frame, val_months, max_rows)
    record = _fit_record(
        model,
        genome,
        split,
        seed,
        command="train",
        parent_run_id=parent_run_id,
        generation=None,
        mutation=mutation or {"axis": "train"},
        pool=features,
        max_rows=max_rows,
        val_months=val_months,
        accepted=True,
        reason=reason or "trained as a frozen genome and set as the champion",
    )
    write_log(model, record)
    champion = _champion_from_record(record, max_rows)
    champion = confirm_champion(model, champion, split, record["trial"], commit=commit_best)
    if commit_best:
        from src.pipeline.best import publish_best

        publish_best(model, record, champion)
    print(f"trained {champion['run_id']} search RMSE {champion['search_rmse']:.4f}")
    return champion


def confirm_champion(model: str, champion: dict, split: TimeSplit, trial: dict | None, commit: bool = True) -> dict:
    """Score the confirm months once. The score is stored and not used to accept a genome."""
    if trial is None:
        preprocessor, bundle, _genome = load_run(model, champion["genome"]["features"], champion["run_id"])
    else:
        preprocessor = trial["preprocessor"]
        bundle = trial["bundle"]
    columns = champion["genome"]["features"]
    prediction = predict_frame(model, preprocessor, bundle, split.confirm, columns)
    score = rmse(split.confirm["MONTHLY_RENT"].to_numpy(dtype=np.float64), prediction)
    from src.pipeline.paths import rel

    run_id = _new_run_id(model, champion["genome_hash"], {row["run_id"] for row in read_logs(model)})
    record = {
        "accepted": None,
        "command": "test",
        "confirm_rmse": score,
        "design_path": rel(design_path(model)),
        "feature_hash": champion["feature_hash"],
        "genome": champion["genome"],
        "genome_hash": champion["genome_hash"],
        "log_path": log_path_for(model, run_id),
        "parent_run_id": champion["run_id"],
        "reason": "confirm RMSE is recorded once and is not an acceptance signal",
        "run_id": run_id,
        "search_rmse": champion["search_rmse"],
        "train_rmse": champion["train_rmse"],
    }
    write_log(model, record)
    champion["confirm_rmse"] = score
    champion["confirm_log"] = record["log_path"]
    if commit:
        write_champion(model, champion)
    print(f"confirm RMSE {score:.4f}")
    return champion


def test_model(model: str, val_months: int | None, max_rows: int | None, seed: int) -> dict:
    del seed
    champion = load_champion(model)
    if champion is None:
        raise SystemExit(f"models/{model}/docs/champion.json does not exist; train or evolve first")
    months = val_months if val_months is not None else champion.get("val_months", 3)
    cap = champion.get("max_rows") if max_rows is None else max_rows
    frame = load_joined("train", champion["genome"]["features"])
    split = time_split(frame, months, cap)
    confirm_champion(model, champion, split, None)
    refresh_design(model)
    return champion


def propose(model, champion, trial, split: TimeSplit, pool, generation, logs, seen, seed):
    genome = champion["genome"]
    parent = champion["run_id"]
    proposals = []
    feature = _feature_mutation(model, genome, trial, split, pool, generation, logs, parent, seed)
    if feature is not None:
        proposals.append(feature)
    preprocess = _preprocess_mutation(genome, parent, logs, seen)
    if preprocess is not None:
        proposals.append(preprocess)
    architecture = _architecture_mutation(model, genome, champion["train_rmse"], champion["search_rmse"])
    if architecture is not None:
        proposals.append(architecture)
    return proposals


def _feature_mutation(model, genome, trial, split, pool, generation, logs, parent, seed):
    tried = [
        record["mutation"]
        for record in logs
        if record.get("parent_run_id") == parent and (record.get("mutation") or {}).get("axis") == "features"
    ]
    tried_column_adds = [item for item in tried if item.get("change") == "add" and "column" in item]
    tried_column_drops = [item for item in tried if item.get("change") == "drop" and "column" in item]
    odd = generation % 2 == 1
    current = list(genome["features"])
    if odd and not tried_column_adds:
        added = _add_column(genome, trial, split, pool)
        if added is not None:
            return added
    if (not odd) and not tried_column_drops:
        dropped = _drop_column(model, genome, trial, split, seed)
        if dropped is not None:
            return dropped
    single_tried = bool(tried_column_adds if odd else tried_column_drops)
    if not single_tried:
        return None
    if odd:
        return _add_family(genome, trial, split, pool)
    return _drop_family(model, genome, trial, split, pool, seed)


def _add_column(genome, trial, split, pool):
    unused = [column for column in pool if column not in genome["features"]]
    if not unused:
        return None
    residual = split.search["MONTHLY_RENT"].to_numpy(dtype=np.float64) - trial["search_pred"]
    index = {column: position for position, column in enumerate(pool)}
    chosen = max(unused, key=lambda column: (_abs_corr(split.search[column], residual), -index[column]))
    return _replace_features(genome, genome["features"] + [chosen], {"axis": "features", "change": "add", "column": chosen})


def _drop_column(model, genome, trial, split, seed):
    columns = list(genome["features"])
    if len(columns) <= 1:
        return None
    scores = _importance(model, genome, trial, split, columns, seed)
    index = {column: position for position, column in enumerate(columns)}
    chosen = min(columns, key=lambda column: (scores[column], -index[column]))
    kept = [column for column in columns if column != chosen]
    return _replace_features(genome, kept, {"axis": "features", "change": "drop", "column": chosen})


def _add_family(genome, trial, split, pool):
    pool_set = set(pool)
    current = set(genome["features"])
    residual = split.search["MONTHLY_RENT"].to_numpy(dtype=np.float64) - trial["search_pred"]
    candidates = []
    for name, columns in FAMILIES.items():
        if not columns or not set(columns) <= pool_set:
            continue
        if any(column in current for column in columns):
            continue
        score = float(np.mean([_abs_corr(split.search[column], residual) for column in columns]))
        candidates.append((score, name, columns))
    if not candidates:
        return None
    _score, name, columns = max(candidates, key=lambda item: (item[0], item[1]))
    return _replace_features(
        genome,
        list(genome["features"]) + list(columns),
        {"axis": "features", "change": "add", "family": name},
    )


def _drop_family(model, genome, trial, split, pool, seed):
    pool_set = set(pool)
    current = list(genome["features"])
    current_set = set(current)
    scores = _importance(model, genome, trial, split, current, seed)
    candidates = []
    for name, columns in FAMILIES.items():
        if not columns or not set(columns) <= pool_set:
            continue
        if not set(columns) <= current_set:
            continue
        if len(current_set - set(columns)) < 1:
            continue
        score = float(np.mean([scores[column] for column in columns]))
        candidates.append((score, name, columns))
    if not candidates:
        return None
    _score, name, columns = min(candidates, key=lambda item: (item[0], item[1]))
    drop = set(columns)
    kept = [column for column in current if column not in drop]
    return _replace_features(genome, kept, {"axis": "features", "change": "drop", "family": name})


def _replace_features(genome, features, mutation):
    if not features or list(features) == list(genome["features"]):
        return None
    copied = json.loads(json.dumps(genome))
    copied["features"] = list(features)
    return copied, mutation


def _preprocess_mutation(genome, parent, logs, seen):
    tried = {
        record["mutation"].get("knob")
        for record in logs
        if record.get("parent_run_id") == parent and (record.get("mutation") or {}).get("axis") == "preprocess"
    }
    for knob, choices in PREPROCESS_KNOBS.items():
        if knob in tried:
            continue
        current = genome["preprocess"][knob]
        if knob == "rare_min_count":
            current = int(current)
        alternate = next(value for value in choices if value != current)
        copied = json.loads(json.dumps(genome))
        copied["preprocess"][knob] = alternate
        if genome_hash(copied) in seen:
            continue
        return copied, {"axis": "preprocess", "knob": knob, "from": current, "to": alternate}
    return None


def _architecture_mutation(model, genome, train_rmse, search_rmse):
    from src.pipeline.fit import load_model_module

    mode = "regularize" if _overfit(train_rmse, search_rmse) else "capacity"
    updated = load_model_module(model).mutate_architecture(genome["architecture"], mode)
    if updated is None:
        return None
    copied = json.loads(json.dumps(genome))
    copied["architecture"] = updated
    return copied, {
        "axis": "architecture",
        "change": mode,
        "from": genome["architecture"],
        "to": updated,
    }


def _importance(model, genome, trial, split, columns, seed):
    count = min(PERMUTATION_ROWS, len(split.search))
    positions = np.unique(np.rint(np.linspace(0, len(split.search) - 1, count)).astype(int))
    sample = split.search.iloc[positions].reset_index(drop=True)
    target = sample["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    base = rmse(target, predict_frame(model, trial["preprocessor"], trial["bundle"], sample, genome["features"]))
    scores = {}
    for index, column in enumerate(columns):
        shuffled = sample.copy()
        generator = np.random.default_rng(seed + index)
        shuffled[column] = generator.permutation(shuffled[column].to_numpy())
        score = rmse(target, predict_frame(model, trial["preprocessor"], trial["bundle"], shuffled, genome["features"]))
        scores[column] = score - base
    return scores


def _abs_corr(series: pd.Series, residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    else:
        codes, _uniques = pd.factorize(series.astype("string"))
        values = codes.astype(np.float64)
        values[codes < 0] = np.nan
    mask = np.isfinite(values) & np.isfinite(residual)
    if int(mask.sum()) < 3:
        return 0.0
    if float(np.std(values[mask])) == 0 or float(np.std(residual[mask])) == 0:
        return 0.0
    return abs(float(np.corrcoef(values[mask], residual[mask])[0, 1]))


def _overfit(train_rmse: float, search_rmse: float) -> bool:
    if train_rmse <= 0:
        return False
    return (search_rmse - train_rmse) / train_rmse > OVERFIT_GAP


def _starting_champion(model, pool, epochs, seed, split, logs, seen, max_rows, val_months):
    stored = load_champion(model)
    if stored is not None and _same_run(stored, model, pool, max_rows, val_months, epochs):
        genome = apply_epochs(stored["genome"], epochs, model)
    else:
        genome = apply_epochs(seed_genome(load_default_genome(model), pool), epochs, model)
    digest = genome_hash(genome)
    if digest in seen and _weights_exist(model, genome, logs):
        record = next(
            row
            for row in logs
            if row.get("genome_hash") == digest and row.get("command") in {"train", "evolve"}
        )
        preprocessor, bundle, _genome = load_run(model, genome["features"], record["run_id"])
        prediction = predict_frame(model, preprocessor, bundle, split.search, genome["features"])
        trial = {
            "preprocessor": preprocessor,
            "bundle": bundle,
            "search_pred": prediction,
            "train_rmse": record["train_rmse"],
            "search_rmse": record["search_rmse"],
        }
        champion = _champion_from_record(record, record.get("max_rows"))
        if stored is not None and stored.get("run_id") == champion["run_id"]:
            champion["confirm_rmse"] = stored.get("confirm_rmse")
            champion["confirm_log"] = stored.get("confirm_log")
        write_champion(model, champion)
        return champion, trial
    record = _fit_record(
        model,
        genome,
        split,
        seed,
        command="evolve",
        parent_run_id=None,
        generation=0,
        mutation={"axis": "seed"},
        pool=pool,
        max_rows=max_rows,
        val_months=val_months,
        accepted=True,
        reason="starting genome",
    )
    seen.add(record["genome_hash"])
    logs.append(record)
    champion = _champion_from_record(record, max_rows)
    write_champion(model, champion)
    refresh_design(model)
    return champion, record["trial"]


def _fit_record(model, genome, split, seed, command, parent_run_id, generation, mutation, pool, max_rows, val_months, accepted, reason):
    print(f"fit {mutation.get('axis')} {mutation.get('change', mutation.get('knob', ''))}".rstrip())
    trial = fit_genome(model, genome, split.fit, split.search, seed)
    existing = {row["run_id"] for row in read_logs(model)}
    run_id = _new_run_id(model, trial["genome_hash"], existing)
    from src.pipeline.paths import rel

    record = {
        "accepted": accepted,
        "command": command,
        "confirm_months": split.confirm_months,
        "design_path": rel(design_path(model)),
        "feature_hash": trial["feature_hash"],
        "fit_months": [split.fit_months[0], split.fit_months[-1]],
        "generation": generation,
        "genome": json.loads(json.dumps(genome)),
        "genome_hash": trial["genome_hash"],
        "log_path": log_path_for(model, run_id),
        "max_rows": max_rows,
        "mutation": mutation,
        "parent_run_id": parent_run_id,
        "pool": list(pool),
        "reason": reason,
        "rows": {"confirm": len(split.confirm), "fit": len(split.fit), "search": len(split.search)},
        "run_id": run_id,
        "search_months": split.search_months,
        "search_rmse": trial["search_rmse"],
        "train_rmse": trial["train_rmse"],
        "val_months": val_months if val_months is not None else len(split.confirm_months),
    }
    save_run(model, genome, trial, record)
    record["trial"] = trial
    write_log(model, record)
    print(f"  {run_id} train RMSE {trial['train_rmse']:.4f} search RMSE {trial['search_rmse']:.4f}")
    return record


def _rewrite_metrics(model, record):
    """Keep the run directory's metrics file on the decision written into the log."""
    from src.pipeline.fit import run_folder

    folder = run_folder(model, record["genome"]["features"], record["run_id"])
    path = folder / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["accepted"] = record["accepted"]
    metrics["reason"] = record["reason"]
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _champion_from_record(record: dict, max_rows) -> dict:
    from src.pipeline.paths import rel

    return {
        "confirm_log": record.get("confirm_log"),
        "confirm_rmse": record.get("confirm_rmse"),
        "feature_hash": record["feature_hash"],
        "genome": record["genome"],
        "genome_hash": record["genome_hash"],
        "log_path": record["log_path"],
        "max_rows": record.get("max_rows", max_rows),
        "run_id": record["run_id"],
        "search_rmse": record["search_rmse"],
        "train_rmse": record["train_rmse"],
        "val_months": record.get("val_months"),
        "design_path": record.get("design_path") or rel(design_path(record.get("model", "mlp"))),
    }


def _same_run(stored: dict, model: str, pool: list[str], max_rows, val_months, epochs) -> bool:
    """Resume a champion only when the pool, row cap, and epoch cap match.

    A smoke champion must not become the starting point of a full run.
    """
    if not set(stored["genome"]["features"]) <= set(pool):
        return False
    if stored.get("max_rows") != max_rows or stored.get("val_months") != val_months:
        return False
    default = load_default_genome(model)["architecture"]
    arch = stored["genome"]["architecture"]
    if model == "mlp":
        wanted = int(default.get("epochs", 20) if epochs is None else epochs)
        return int(arch.get("epochs", wanted)) == wanted
    if model in {"xgb", "catboost"}:
        wanted = int(default.get("n_estimators", 200) if epochs is None else epochs)
        return int(arch.get("n_estimators", wanted)) == wanted
    return epochs is None


def _seen_hashes(logs: list[dict]) -> set[str]:
    return {record["genome_hash"] for record in logs if record.get("command") in {"train", "evolve"} and record.get("genome_hash")}


def _weights_exist(model, genome, logs) -> bool:
    from src.pipeline.fit import load_model_module, run_folder

    name = load_model_module(model).WEIGHT_NAME
    for record in logs:
        if record.get("genome_hash") != genome_hash(genome):
            continue
        if record.get("command") not in {"train", "evolve"}:
            continue
        path = run_folder(model, genome["features"], record["run_id"]) / name
        if path.exists():
            return True
    return False


def _new_run_id(model: str, digest: str, existing: set[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    base = f"{stamp}_{model}_{digest[:8]}"
    run_id = base
    number = 2
    while run_id in existing:
        run_id = f"{base}_{number}"
        number += 1
    return run_id


def _rank(record: dict):
    return (record["search_rmse"], AXIS_RANK.get(record["mutation"]["axis"], 9), record["genome_hash"])


def predict_model(model: str, out: str, refit_all: bool, seed: int) -> dict:
    """Write Id,Predicted from the champion. This does not change the genome."""
    import joblib
    from pathlib import Path

    from src.pipeline.fit import load_model_module, run_folder
    from src.pipeline.genome import write_features_file
    from src.pipeline.paths import ROOT, rel

    champion = load_champion(model)
    from src.pipeline.finalize import load_final_weights

    final_weights = None if refit_all else load_final_weights(model)
    if final_weights is not None:
        preprocessor, bundle, genome, used_run = final_weights
        columns = list(genome["features"])
        feature_hash = champion["feature_hash"] if champion is not None else used_run
        genome_hash = champion["genome_hash"] if champion is not None else used_run
    elif champion is None:
        raise SystemExit(f"models/{model}/docs/champion.json does not exist; train or evolve first")
    else:
        genome = json.loads(json.dumps(champion["genome"]))
        columns = list(genome["features"])
        feature_hash = champion["feature_hash"]
        genome_hash = champion["genome_hash"]
        if refit_all:
            frame = load_joined("train", columns)
            trial = fit_genome(model, genome, frame, frame, seed)
            preprocessor = trial["preprocessor"]
            bundle = trial["bundle"]
            refit_id = _new_run_id(model, champion["genome_hash"] + "refit", {row["run_id"] for row in read_logs(model)})
            folder = run_folder(model, columns, refit_id)
            folder.mkdir(parents=True, exist_ok=True)
            write_features_file(folder.parent.parent, columns)
            payload = json.loads(json.dumps(genome))
            payload["refit_all"] = True
            (folder / "genome.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            joblib.dump(preprocessor, folder / "preprocessor.joblib")
            load_model_module(model).save_model(bundle, folder)
            used_run = refit_id
        else:
            preprocessor, bundle, _genome = load_run(model, columns, champion["run_id"])
            used_run = champion["run_id"]
    from src.pipeline.genome import feature_hash as digest_features
    from src.pipeline.genome import genome_hash as digest_genome

    feature_hash = digest_features(columns)
    genome_hash = digest_genome(genome)
    test = load_joined("test", columns)
    prediction = predict_frame(model, preprocessor, bundle, test, columns)
    output = Path(out) if out else ROOT / "outputs" / "predictions" / f"{model}_{feature_hash}.csv"
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"Id": np.arange(len(prediction)), "Predicted": prediction})
    table.to_csv(output, index=False)
    existing = {row["run_id"] for row in read_logs(model)}
    run_id = _new_run_id(model, "predict00", existing)
    record = {
        "champion_run_id": used_run,
        "command": "predict",
        "design_path": rel(design_path(model)),
        "feature_hash": feature_hash,
        "genome": genome,
        "genome_hash": genome_hash,
        "log_path": log_path_for(model, run_id),
        "n": int(len(prediction)),
        "output": rel(output),
        "prediction_max": float(np.max(prediction)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_std": float(np.std(prediction)),
        "refit_all": bool(refit_all),
        "run_id": run_id,
    }
    write_log(model, record)
    refresh_design(model)
    print(f"wrote {len(prediction)} predictions to {rel(output)}")
    return record

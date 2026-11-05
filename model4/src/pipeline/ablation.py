"""Search auxiliary columns one round at a time with the model's own architecture.

Set A is the current best feature list. Each missing auxiliary column is
trained as set B = set A + that column for the same epoch budget. Ridge and
the MLP call a column promising when its search RMSE is at least 1% below
set A. XGBoost and CatBoost call it promising when the search RMSE is strictly
lower. Those probes do not change ``best/``. Set C is set A plus every
promising column, trained once. It replaces the best snapshot only when its
search RMSE is lower.
The loop stops when a round finds no promising column, or when set C does
not beat the previous best.
"""

import json
from pathlib import Path

from src.features.build_features import FEATURE_COLUMNS
from src.pipeline.best import load_best, seed_best_from_champion
from src.pipeline.evolve import predict_model, train_model
from src.pipeline.paths import MODELS
from src.pipeline.records import read_logs, refresh_design, write_log
from src.pipeline.rules import SEARCH_GAIN

ANY_DROP_MODELS = {"xgb", "catboost"}


def ablate_model(model: str, seed: int = 5228, epochs: int | None = None) -> dict:
    seed_best_from_champion(model)
    if load_best(model) is None:
        raise SystemExit(f"models/{model}/best does not exist; train a model first")
    folder = MODELS / model / "ablation"
    folder.mkdir(parents=True, exist_ok=True)
    rounds = []
    installed = []
    round_index = 0
    while True:
        round_index += 1
        best = load_best(model)
        set_a = list(best["champion"]["genome"]["features"])
        locked = {
            "architecture": best["champion"]["genome"]["architecture"],
            "preprocess": best["champion"]["genome"]["preprocess"],
        }
        base_rmse = float(best["search_rmse"])
        base_run = best["run_id"]
        candidates = [column for column in FEATURE_COLUMNS if column not in set_a]
        print(
            f"round {round_index} {model}: set A has {len(set_a)} columns, "
            f"search RMSE {base_rmse:.4f}, {len(candidates)} candidates"
        )
        promising = []
        for index, column in enumerate(candidates, start=1):
            print(f"  {index}/{len(candidates)} set B + {column}")
            trained = train_model(
                model,
                set_a + [column],
                best["champion"].get("val_months") or 3,
                None,
                epochs,
                seed,
                locked=locked,
                commit_best=False,
                parent_run_id=base_run,
                mutation={"axis": "features", "change": "add", "column": column, "round": round_index},
                reason=f"round {round_index} probe of {column} against {base_run}",
            )
            is_promising = _is_promising(model, trained["search_rmse"], base_rmse)
            if is_promising:
                promising.append(column)
            _mark_probe(model, trained["run_id"], base_run, base_rmse, column, is_promising)
            _write_round(
                folder,
                {
                    "base_run_id": base_run,
                    "base_search_rmse": base_rmse,
                    "column": column,
                    "confirm_rmse": trained.get("confirm_rmse"),
                    "promising": is_promising,
                    "round": round_index,
                    "run_id": trained["run_id"],
                    "search_rmse": trained["search_rmse"],
                    "stage": "probe",
                },
            )
            print(
                f"    search {trained['search_rmse']:.4f} "
                f"{'promising' if is_promising else 'skip'}"
            )
        refresh_design(model)
        if not promising:
            rounds.append({"promising": [], "replaced": False, "round": round_index})
            print(f"round {round_index} stopped: no promising columns")
            break
        print(f"round {round_index} set C with {len(promising)} columns")
        trained_c = train_model(
            model,
            set_a + promising,
            best["champion"].get("val_months") or 3,
            None,
            epochs,
            seed,
            locked=locked,
            commit_best=True,
            parent_run_id=base_run,
            mutation={"axis": "features", "change": "set_c", "columns": promising, "round": round_index},
            reason=f"round {round_index} set C against {base_run}",
        )
        finished = load_best(model)
        replaced = finished is not None and finished["run_id"] != base_run
        _write_round(
            folder,
            {
                "base_run_id": base_run,
                "base_search_rmse": base_rmse,
                "columns": promising,
                "confirm_rmse": trained_c.get("confirm_rmse"),
                "replaced": replaced,
                "round": round_index,
                "run_id": trained_c["run_id"],
                "search_rmse": trained_c["search_rmse"],
                "stage": "set_c",
            },
        )
        rounds.append(
            {
                "promising": promising,
                "replaced": replaced,
                "round": round_index,
                "search_rmse": trained_c["search_rmse"],
            }
        )
        if not replaced:
            print(
                f"round {round_index} stopped: set C search RMSE {trained_c['search_rmse']:.4f} "
                f"is not below {base_rmse:.4f}"
            )
            break
        installed.append(finished["run_id"])
        predict_model(model, f"artifacts/predictions/{model}_ablation.csv", False, seed)
        print(f"round {round_index} replaced best with {finished['run_id']}")
    summary = {"installed": installed, "rounds": rounds}
    (folder / "rounds.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def recombine_logged_probes(model: str, seed: int = 5228) -> dict:
    """Train set C from round-1 probe files. Does not refit the probes."""
    if model not in ANY_DROP_MODELS:
        raise SystemExit(f"{model} still uses the 1% probe rule")
    best = load_best(model)
    if best is None:
        raise SystemExit(f"models/{model}/best does not exist")
    folder = MODELS / model / "ablation"
    base_run = best["run_id"]
    base_rmse = float(best["search_rmse"])
    probes = []
    for path in sorted(folder.glob("round_1_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("stage") != "probe" or row.get("base_run_id") != base_run:
            continue
        probes.append(row)
    order = {column: index for index, column in enumerate(FEATURE_COLUMNS)}
    promising = [row["column"] for row in probes if row["search_rmse"] < row["base_search_rmse"]]
    promising.sort(key=lambda column: order.get(column, len(order)))
    set_a = list(best["champion"]["genome"]["features"])
    locked = {
        "architecture": best["champion"]["genome"]["architecture"],
        "preprocess": best["champion"]["genome"]["preprocess"],
    }
    print(f"{model} set C from {len(promising)} logged drops against {base_run}")
    trained_c = train_model(
        model,
        set_a + promising,
        best["champion"].get("val_months") or 3,
        None,
        None,
        seed,
        locked=locked,
        commit_best=True,
        parent_run_id=base_run,
        mutation={"axis": "features", "change": "set_c", "columns": promising, "round": 1},
        reason=(
            f"round 1 set C against {base_run}; each column had a lower search RMSE "
            "and the 1% bar was not required"
        ),
    )
    finished = load_best(model)
    replaced = finished is not None and finished["run_id"] != base_run
    _write_round(
        folder,
        {
            "base_run_id": base_run,
            "base_search_rmse": base_rmse,
            "columns": promising,
            "confirm_rmse": trained_c.get("confirm_rmse"),
            "replaced": replaced,
            "round": 1,
            "rule": "any_drop",
            "run_id": trained_c["run_id"],
            "search_rmse": trained_c["search_rmse"],
            "stage": "set_c",
        },
    )
    summary = {
        "installed": [finished["run_id"]] if replaced else [],
        "rounds": [
            {
                "promising": promising,
                "replaced": replaced,
                "round": 1,
                "rule": "any_drop",
                "search_rmse": trained_c["search_rmse"],
            }
        ],
    }
    (folder / "rounds.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"set C search RMSE {trained_c['search_rmse']:.4f} "
        f"{'replaced best' if replaced else 'left best in place'}"
    )
    refresh_design(model)
    return summary


def _is_promising(model: str, search_rmse: float, base_rmse: float) -> bool:
    gain = 0.0 if model in ANY_DROP_MODELS else SEARCH_GAIN
    return search_rmse < base_rmse * (1.0 - gain)


def _mark_probe(model: str, run_id: str, base_run: str, base_rmse: float, column: str, promising: bool) -> None:
    record = next(
        row for row in read_logs(model) if row.get("run_id") == run_id and row.get("command") == "train"
    )
    record["accepted"] = promising
    record["parent_run_id"] = base_run
    score = record["search_rmse"]
    if model in ANY_DROP_MODELS:
        record["reason"] = (
            f"{'promising' if promising else 'not promising'} because search RMSE {score:.6f} "
            f"is {'below' if promising else 'not below'} {base_rmse:.6f} from {base_run} after adding {column}"
        )
    elif promising:
        record["reason"] = (
            f"promising because search RMSE {score:.6f} is at least "
            f"{SEARCH_GAIN:.0%} below {base_rmse:.6f} from {base_run} after adding {column}"
        )
    else:
        record["reason"] = (
            f"not promising because search RMSE {score:.6f} is not at least "
            f"{SEARCH_GAIN:.0%} below {base_rmse:.6f} from {base_run} after adding {column}"
        )
    write_log(model, record)


def _write_round(folder: Path, payload: dict) -> None:
    if payload["stage"] == "probe":
        name = f"round_{payload['round']}_{payload['column']}.json"
    else:
        name = f"round_{payload['round']}_set_c.json"
    (folder / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

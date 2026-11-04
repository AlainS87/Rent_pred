"""Retrain the multistage stack with the core-data preprocessing of the 454.29 CatBoost.

The saved three-layer RNN is loaded and not refit. Per-town CatBoost, the
per-town mixture, and the shared MLP are trained from scratch in this
directory. Validation months 2024-10..12 stop CatBoost and the MLP. The
January-March 2025 score and the six-month pooled score are computed after
that choice.

    python -u "mean_residual_model/aligned_preprocess/run.py"
    python -u "mean_residual_model/aligned_preprocess/run.py" --resume
    python -u "mean_residual_model/aligned_preprocess/run.py" --predict

Training-row CatBoost gaps that enter the MLP come from the full town model
fit on those same rows. They are in-sample, not out of fold. The mixture gaps
on those rows are in-sample as well. ``mu`` and both gaps stay in rent per
square metre; floor area is a separate MLP input.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mean_residual_model.reginal_mean_model.rnn.model import predict_saved_rnn3
from mean_residual_model.reginal_mean_model.sequence_store import town_slug
from mean_residual_model.run import SEED, _check_months, _write_kaggle
from models.catboost.model import fit_model, load_model, predict_model, save_model
from models.catboost.preprocess import fit_preprocessor, transform_frame
from src.features.keys import block_key
from src.features.source_fields import flat_type_key
from src.pipeline.data import load_joined, time_split
from src.pipeline.fit import rmse

HERE = Path(__file__).resolve().parent
STACK_PATH = ROOT / "mean_residual_model" / "multi_stage_model(lstm+catboost+mlp)" / "run.py"
PREDICTION_PATH = ROOT / "outputs" / "predictions" / "mean_residual_aligned_preprocess.csv"

CAT_FEATURES = [
    "latitude",
    "longitude",
    "building_key",
    "street_key",
    "flat_type_ord",
    "flat_type_key",
    "flat_model_ord",
    "flat_model_key",
    "approval_year",
    "approval_month",
    "approval_month_index",
    "lease_age",
    "lease_age_invalid",
    "nearest_open_mrt_code",
    "dist_nearest_open_mrt_km",
    "log_dist_nearest_open_mrt_km",
    "nearest_mall_name",
    "dist_nearest_mall_km",
    "nearest_primary_name",
    "dist_nearest_primary_km",
    "nearest_secondary_name",
    "dist_nearest_secondary_km",
]
CAT_CATEGORICAL = [
    "building_key",
    "street_key",
    "flat_type_key",
    "flat_model_key",
    "nearest_open_mrt_code",
    "nearest_mall_name",
    "nearest_primary_name",
    "nearest_secondary_name",
]
MLP_FIELDS = [
    "mu",
    "t",
    "gmm_gap",
    "floor_area",
    "building_age",
    "month_index",
    "approval_year",
    "approval_month",
    "lease_age",
    "lease_age_invalid",
    "latitude",
    "longitude",
    "latitude_raw",
    "longitude_raw",
    "flat_type_ord",
    "flat_model_ord",
    "max_floor",
    "year_completed",
    "dist_mrt",
    "dist_mall",
    "dist_primary",
    "dist_secondary",
    "log_dist_mrt",
    "n_mrt_1km",
    "n_mall_1km",
    "n_primary_1km",
    "n_secondary_1km",
    "n_blocks_500m",
]
LOAD_COLUMNS = [
    "RENT_APPROVAL_DATE",
    "TOWN",
    "BLOCK",
    "STREET",
    "FLAT_TYPE",
    "building_age_years",
    "max_floor",
    "year_completed",
    "flat_age_from_lease",
    "approval_month_index",
    "floor_area_sqm",
    "lease_commence_year",
    "latitude",
    "longitude",
    "street_key",
    "flat_type_ord",
    "flat_model_ord",
    "flat_model_key",
    "nearest_open_mrt_code",
    "dist_nearest_open_mrt_km",
    "log_dist_nearest_open_mrt_km",
    "nearest_mall_name",
    "dist_nearest_mall_km",
    "nearest_primary_name",
    "dist_nearest_primary_km",
    "nearest_secondary_name",
    "dist_nearest_secondary_km",
    "n_open_mrt_within_1km",
    "n_mall_within_1km",
    "n_primary_within_1km",
    "n_secondary_within_1km",
    "n_hdb_blocks_within_500m",
]
PREPROCESS = {
    "categorical": "native",
    "numeric_impute": "median",
    "rare_min_count": 30,
    "scale": "none",
}
CAT_ARCH = {
    "early_stopping_rounds": 50,
    "learning_rate": 0.05,
    "max_depth": 8,
    "n_estimators": 1000,
}
PLAIN_FIELDS = {
    "floor_area": "floor_area_sqm",
    "building_age": "building_age_years",
    "month_index": "approval_month_index",
    "approval_year": "approval_year",
    "approval_month": "approval_month",
    "lease_age": "lease_age",
    "lease_age_invalid": "lease_age_invalid",
    "flat_type_ord": "flat_type_ord",
    "flat_model_ord": "flat_model_ord",
    "max_floor": "max_floor",
    "year_completed": "year_completed",
    "dist_mrt": "dist_nearest_open_mrt_km",
    "dist_mall": "dist_nearest_mall_km",
    "dist_primary": "dist_nearest_primary_km",
    "dist_secondary": "dist_nearest_secondary_km",
    "log_dist_mrt": "log_dist_nearest_open_mrt_km",
    "n_mrt_1km": "n_open_mrt_within_1km",
    "n_mall_1km": "n_mall_within_1km",
    "n_primary_1km": "n_primary_within_1km",
    "n_secondary_1km": "n_secondary_within_1km",
    "n_blocks_500m": "n_hdb_blocks_within_500m",
}
PROTECTED_DIRS = [
    ROOT / "mean_residual_model" / "reginal_mean_model" / "rnn" / "h16_l3",
    ROOT / "mean_residual_model" / "refit",
    ROOT / "mean_residual_model" / "multi_stage_model(lstm+catboost+mlp)",
    ROOT / "models",
]
PROTECTED_FILES = [
    ROOT / "outputs" / "predictions" / "mean_residual_multistage.csv",
    ROOT / "outputs" / "predictions" / "mean_residual_multistage_refit.csv",
]


def _load_stack():
    """Load the existing stack as a file. The folder name is not a package."""
    if not STACK_PATH.is_file():
        raise SystemExit(f"missing stack script: {STACK_PATH}")
    spec = importlib.util.spec_from_file_location("multistage_stack_aligned", STACK_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {STACK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_aligned(frame: pd.DataFrame, invalid_rows: int) -> pd.DataFrame:
    """Approval year, calendar month, BLOCK|STREET, and lease age at approval.

    A lease year later than the approval year keeps the row. Lease age is
    null on that row and ``lease_age_invalid`` is 1. Every other row uses
    approval year minus lease commencement year.
    """
    text = frame["RENT_APPROVAL_DATE"].astype("string").str.strip()
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce")
    month = pd.to_numeric(text.str.slice(5, 7), errors="coerce")
    if year.isna().any() or month.isna().any() or (month < 1).any() or (month > 12).any():
        raise SystemExit("RENT_APPROVAL_DATE is not YYYY-MM")
    lease = pd.to_numeric(frame["lease_commence_year"], errors="coerce")
    if lease.isna().any():
        raise SystemExit("lease commencement year is missing")
    invalid = lease > year
    if int(invalid.sum()) != invalid_rows:
        raise SystemExit(f"expected {invalid_rows} invalid lease rows, found {int(invalid.sum())}")
    age = (year - lease).astype("float64")
    age = age.mask(invalid, np.nan)
    stored_age = pd.to_numeric(frame["flat_age_from_lease"], errors="coerce")
    if not np.allclose(age.loc[~invalid].to_numpy(dtype=np.float64), stored_age.loc[~invalid].to_numpy(dtype=np.float64)):
        raise SystemExit("lease age does not match flat_age_from_lease on valid rows")
    stored_index = frame["approval_month_index"].to_numpy(dtype=np.int64)
    fresh_index = ((year - 2021) * 12 + (month - 1)).to_numpy(dtype=np.int64)
    if not np.array_equal(stored_index, fresh_index):
        raise SystemExit("approval_month_index is not January 2021 = 0")
    out = frame.copy()
    out["approval_year"] = year.astype("float64")
    out["approval_month"] = month.astype("float64")
    out["lease_age"] = age.to_numpy(dtype=np.float64)
    out["lease_age_invalid"] = invalid.astype("float64")
    out["building_key"] = block_key(out["BLOCK"], out["STREET"])
    out["flat_type_key"] = flat_type_key(out["FLAT_TYPE"])
    if out["building_key"].isna().any() or out["flat_type_key"].isna().any():
        raise SystemExit("building key or flat type is missing")
    if not out["building_key"].astype(str).str.contains("|", regex=False).all():
        raise SystemExit("building key is not BLOCK|STREET")
    return out


def _fingerprint() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for root in PROTECTED_DIRS:
        if not root.exists():
            raise SystemExit(f"missing protected directory: {root}")
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            stat = path.stat()
            found[str(path.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    for path in PROTECTED_FILES:
        if not path.is_file():
            raise SystemExit(f"missing protected file: {path}")
        stat = path.stat()
        found[str(path.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    return found


def _changed_protected(before: dict[str, list[int]]) -> list[str]:
    after = _fingerprint()
    changed = sorted(set(before) ^ set(after))
    for name in sorted(set(before) & set(after)):
        if before[name] != after[name]:
            changed.append(name)
    return sorted(set(changed))


def _empty_store(periods, test) -> dict:
    def block(n_rows: int) -> dict:
        data = {name: np.zeros(n_rows, dtype=np.float64) for name in MLP_FIELDS}
        for name in ("mu", "t", "gmm_gap"):
            data[name] = np.full(n_rows, np.nan, dtype=np.float64)
        return data

    store = {name: block(len(part)) for name, part in periods.items()}
    store["test"] = block(len(test))
    for name, part in periods.items():
        _copy_plain(store[name], part)
    _copy_plain(store["test"], test)
    return store


def _copy_plain(block, frame: pd.DataFrame) -> None:
    for dest, source in PLAIN_FIELDS.items():
        block[dest][:] = frame[source].to_numpy(dtype=np.float64)


def _matrix(block, fit=False, center=None, spread=None):
    values = np.column_stack([block[name] for name in MLP_FIELDS]).astype(np.float64)
    if fit:
        center = np.nanmedian(values, axis=0)
        if not np.isfinite(center).all():
            bad = [name for name, value in zip(MLP_FIELDS, center) if not np.isfinite(value)]
            raise RuntimeError(f"MLP median is not finite for {bad}")
        filled = np.where(np.isfinite(values), values, center)
        spread = np.std(filled, axis=0)
        spread[spread == 0] = 1.0
        return (filled - center) / spread, center, spread
    filled = np.where(np.isfinite(values), values, center)
    return (filled - center) / spread


def _mlp_predict(network, block, x_center, x_spread, y_center, y_spread) -> np.ndarray:
    values = _matrix(block, center=x_center, spread=x_spread)
    network.eval()
    with torch.no_grad():
        scaled = network(torch.tensor(values, dtype=torch.float32)).numpy()
    return scaled * y_spread + y_center


def _fit_mlp(network_cls, store, rents):
    train_x, center, spread = _matrix(store["train"], fit=True)
    torch.manual_seed(SEED)
    y = rents["train"]
    y_center = float(np.mean(y))
    y_spread = float(np.std(y)) or 1.0
    network = network_cls(train_x.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    features = torch.tensor(train_x, dtype=torch.float32)
    target = torch.tensor((y - y_center) / y_spread, dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    best_state = None
    best_rmse = float("inf")
    best_epoch = 0
    stall = 0
    for epoch in range(1, 101):
        network.train()
        order = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], 1024):
            batch = order[start : start + 1024]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(features[batch]), target[batch])
            loss.backward()
            optimizer.step()
        prediction = _mlp_predict(network, store["search"], center, spread, y_center, y_spread)
        score = rmse(rents["search"], prediction)
        print(f"mlp epoch {epoch} validation rmse {score:.4f}", flush=True)
        if score < best_rmse - 1e-6:
            best_rmse = score
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            best_epoch = epoch
            stall = 0
        else:
            stall += 1
            if stall >= 10:
                break
    if best_state is None:
        raise RuntimeError("MLP produced no weights")
    network.load_state_dict(best_state)
    network.eval()
    return network, best_epoch, center, spread, y_center, y_spread


def _save_mlp(network, epoch, x_center, x_spread, y_center, y_spread) -> None:
    folder = HERE / "mlp"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state": {key: value.detach().cpu() for key, value in network.state_dict().items()},
            "x_center": x_center,
            "x_spread": x_spread,
            "y_center": y_center,
            "y_spread": y_spread,
            "mlp_fields": list(MLP_FIELDS),
            "cat_features": list(CAT_FEATURES),
        },
        folder / "model.pt",
    )


def _catboost(fit, y_fit, search, y_search):
    preparer = fit_preprocessor(PREPROCESS, fit, y_fit, CAT_FEATURES)
    if preparer.categorical_columns_ != [name for name in CAT_FEATURES if name in CAT_CATEGORICAL]:
        raise RuntimeError(f"categorical columns are {preparer.categorical_columns_}")
    return fit_model(
        transform_frame(preparer, fit),
        y_fit,
        transform_frame(preparer, search),
        y_search,
        CAT_ARCH,
        SEED,
    ), preparer


def _town_dir(town: str) -> Path:
    return HERE / "catboost" / town_slug(town)


def _gmm_path(town: str) -> Path:
    return HERE / "gmm" / f"{town_slug(town)}.joblib"


def _town_ready(town: str) -> bool:
    folder = _town_dir(town)
    spec = folder / "features.json"
    if not (folder / "model.joblib").is_file() or not (folder / "preprocessor.joblib").is_file():
        return False
    if not _gmm_path(town).is_file() or not spec.is_file():
        return False
    saved = json.loads(spec.read_text(encoding="utf-8"))
    return saved == CAT_FEATURES


def _predict_catboost(bundle, preparer, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.zeros(0, dtype=np.float64)
    return predict_model(bundle, transform_frame(preparer, frame))


def _assert_feature_roles(fit: pd.DataFrame, y_fit: np.ndarray) -> None:
    preparer = fit_preprocessor(PREPROCESS, fit, y_fit, CAT_FEATURES)
    expected = [name for name in CAT_FEATURES if name in CAT_CATEGORICAL]
    if preparer.categorical_columns_ != expected:
        raise SystemExit(f"categorical columns are {preparer.categorical_columns_}, expected {expected}")
    numeric = [name for name in CAT_FEATURES if name not in CAT_CATEGORICAL]
    if preparer.numeric_columns_ != numeric:
        raise SystemExit(f"numeric columns are {preparer.numeric_columns_}, expected {numeric}")
    matrix = transform_frame(preparer, fit)
    if not np.isfinite(matrix["lease_age"].to_numpy(dtype=np.float64)).all():
        raise SystemExit("lease age was not imputed for CatBoost")
    invalid_pos = np.flatnonzero(fit["lease_age_invalid"].to_numpy(dtype=np.float64) == 1.0)
    if len(invalid_pos) != 1:
        raise SystemExit(f"preprocessor check expected one invalid lease row, found {len(invalid_pos)}")
    if float(matrix["lease_age_invalid"].iloc[int(invalid_pos[0])]) != 1.0:
        raise SystemExit("the invalid lease indicator was not kept")
    known_types = {"1 room", "2 room", "3 room", "4 room", "5 room", "executive"}
    levels = set(matrix["flat_type_key"].cat.categories)
    if not levels or not levels <= known_types:
        raise SystemExit(f"flat type categories are {sorted(levels)}")
    if any("|" not in str(level) for level in matrix["building_key"].cat.categories):
        raise SystemExit("a building category is not BLOCK|STREET")
    banned = {"fee_amount", "FEE", "FURNISHED", "furnished_bin", "block_key", "month_sin", "month_cos"}
    if banned & set(matrix.columns):
        raise SystemExit(f"banned columns reached CatBoost: {banned & set(matrix.columns)}")


def _fit_or_load_town(stack, town, periods, residual, test, store, resume: bool) -> dict:
    masks = {name: part["TOWN"].astype(str).to_numpy() == town for name, part in periods.items()}
    test_mask = test["TOWN"].astype(str).to_numpy() == town
    fit = periods["train"].loc[masks["train"]].reset_index(drop=True)
    search = periods["search"].loc[masks["search"]].reset_index(drop=True)
    confirm = periods["confirm"].loc[masks["confirm"]].reset_index(drop=True)
    test_town = test.loc[test_mask].reset_index(drop=True)
    y_fit = residual["train"][masks["train"]]
    y_search = residual["search"][masks["search"]]
    if len(fit) == 0 or len(search) == 0 or len(confirm) == 0:
        raise RuntimeError(f"{town} is missing a time split")
    stack._scale_coords(fit, [search, confirm, test_town])
    stack._write_coords(store, "train", masks["train"], fit)
    stack._write_coords(store, "search", masks["search"], search)
    stack._write_coords(store, "confirm", masks["confirm"], confirm)
    stack._write_coords(store, "test", test_mask, test_town)

    folder = _town_dir(town)
    loaded = False
    if resume and _town_ready(town):
        print(f"  load saved catboost and mixture {town}", flush=True)
        bundle = load_model(folder)
        preparer = joblib.load(folder / "preprocessor.joblib")
        mixture = joblib.load(_gmm_path(town))
        best_iteration = None
        loaded = True
    else:
        started = time.perf_counter()
        bundle, preparer = _catboost(fit, y_fit, search, y_search)
        try:
            best_iteration = int(bundle.get_best_iteration())
        except Exception:
            best_iteration = None
        folder.mkdir(parents=True, exist_ok=True)
        save_model(bundle, folder)
        joblib.dump(preparer, folder / "preprocessor.joblib")
        (folder / "features.json").write_text(json.dumps(CAT_FEATURES, indent=2) + "\n", encoding="utf-8")
        mixture = stack._fit_mixture(fit, y_fit)
        if mixture.get("kind") != "conditional_location_flat_per_sqm":
            raise RuntimeError(f"{town} mixture kind is {mixture.get('kind')}")
        _gmm_path(town).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(mixture, _gmm_path(town))
        elapsed = time.perf_counter() - started
        print(
            f"  mixture k {mixture['n_components']} best_iteration {best_iteration} "
            f"seconds {elapsed:.1f}",
            flush=True,
        )

    if mixture.get("kind") != "conditional_location_flat_per_sqm":
        raise RuntimeError(f"{town} mixture kind is {mixture.get('kind')}")
    for name, frame, mask in (
        ("train", fit, masks["train"]),
        ("search", search, masks["search"]),
        ("confirm", confirm, masks["confirm"]),
        ("test", test_town, test_mask),
    ):
        if int(np.asarray(mask).sum()) == 0:
            continue
        store[name]["t"][mask] = _predict_catboost(bundle, preparer, frame)
        stack._write_mixture(store, name, mask, frame, mixture)
    return {
        "town": town,
        "n_components": int(mixture["n_components"]),
        "n_fit": int(len(fit)),
        "n_search": int(len(search)),
        "n_confirm": int(len(confirm)),
        "best_iteration": best_iteration,
        "loaded": loaded,
    }


def _coverage(frame: pd.DataFrame, towns: list[str]) -> None:
    labels = frame["TOWN"].astype(str).to_numpy()
    covered = np.zeros(len(frame), dtype=bool)
    for town in towns:
        mask = labels == town
        if covered[mask].any():
            raise SystemExit(f"{town} overlaps another town")
        covered[mask] = True
    if not covered.all():
        missing = sorted(set(labels[~covered]))
        raise SystemExit(f"rows are outside the town list: {missing}")


def _prepare():
    print("loading tables", flush=True)
    train = _add_aligned(load_joined("train", LOAD_COLUMNS), 1)
    test = _add_aligned(load_joined("test", LOAD_COLUMNS), 0)
    if len(train) != 150000 or len(test) != 50000:
        raise SystemExit(f"row counts are train {len(train)} test {len(test)}")
    if train["building_key"].nunique() != 9406:
        raise SystemExit(f"train buildings {train['building_key'].nunique()}, expected 9406")
    flat_types = set(train["flat_type_key"].astype(str).unique())
    if flat_types != {"1 room", "2 room", "3 room", "4 room", "5 room", "executive"}:
        raise SystemExit(f"train flat types are {sorted(flat_types)}")
    split = time_split(train, 3, None)
    _check_months(split)
    counts = (len(split.fit), len(split.search), len(split.confirm))
    if counts != (132802, 8274, 8924):
        raise SystemExit(f"split sizes are {counts}")
    if int(split.fit["lease_age"].isna().sum()) != 1:
        raise SystemExit("the invalid lease row is not in the training months")
    if int(split.search["lease_age"].isna().sum()) or int(split.confirm["lease_age"].isna().sum()):
        raise SystemExit("a validation or test row lost its lease age")
    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    return split, periods, test


def _levels(stack, periods, test):
    print("loading saved RNN", flush=True)
    level = {name: predict_saved_rnn3(part) for name, part in periods.items()}
    test_level = predict_saved_rnn3(test)
    for name, values in level.items():
        if len(values) != len(periods[name]) or not np.isfinite(values).all():
            raise SystemExit(f"RNN prediction for {name} is not finite")
    if len(test_level) != len(test) or not np.isfinite(test_level).all():
        raise SystemExit("RNN prediction for the private test is not finite")
    rents = {name: part["MONTHLY_RENT"].to_numpy(dtype=np.float64) for name, part in periods.items()}
    residual = {}
    for name, part in periods.items():
        units, area = stack._unit_and_area(part)
        if not np.isfinite(area).all() or np.any(area <= 0):
            raise SystemExit(f"{name} floor area is not positive")
        residual[name] = units - level[name]
        if not np.isfinite(residual[name]).all():
            raise SystemExit(f"{name} unit gap is not finite")
    return level, test_level, rents, residual


def _score(network, store, rents, x_center, x_spread, y_center, y_spread) -> dict:
    search_pred = _mlp_predict(network, store["search"], x_center, x_spread, y_center, y_spread)
    confirm_pred = _mlp_predict(network, store["confirm"], x_center, x_spread, y_center, y_spread)
    train_pred = _mlp_predict(network, store["train"], x_center, x_spread, y_center, y_spread)
    pooled_y = np.concatenate([rents["search"], rents["confirm"]])
    pooled_p = np.concatenate([search_pred, confirm_pred])
    if len(pooled_y) != 17198:
        raise SystemExit(f"pooled rows are {len(pooled_y)}, expected 17198")
    return {
        "train_rmse": rmse(rents["train"], train_pred),
        "validation_rmse": rmse(rents["search"], search_pred),
        "test_rmse": rmse(rents["confirm"], confirm_pred),
        "pooled_oct2024_mar2025_rmse": rmse(pooled_y, pooled_p),
        "search_pred": search_pred,
        "confirm_pred": confirm_pred,
    }


def main(resume: bool) -> None:
    banned_inputs = {"fee_amount", "FEE", "FURNISHED", "furnished_bin", "block_key"}
    if banned_inputs & set(CAT_FEATURES) or banned_inputs & set(MLP_FIELDS):
        raise SystemExit("fee, furnished, or block-only key is still a model input")
    if "flat_type_ord" not in MLP_FIELDS or "building_key" in MLP_FIELDS:
        raise SystemExit("the MLP feature list drifted")
    print("checking protected files", flush=True)
    before = _fingerprint()
    print(f"protected files {len(before)}", flush=True)
    (HERE / "protected_before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    stack = _load_stack()
    split, periods, test = _prepare()
    level, test_level, rents, residual = _levels(stack, periods, test)
    towns = stack._towns(periods)
    print(f"towns {len(towns)}", flush=True)
    missing = sorted(set(test["TOWN"].astype(str)) - set(towns))
    if missing:
        raise SystemExit(f"private-test towns absent from the fit months: {missing}")
    for name, part in periods.items():
        _coverage(part, towns)
    _coverage(test, towns)
    bedok = periods["train"].loc[periods["train"]["TOWN"].astype(str).to_numpy() == "bedok"].reset_index(drop=True)
    bedok_y = residual["train"][periods["train"]["TOWN"].astype(str).to_numpy() == "bedok"]
    _assert_feature_roles(bedok, bedok_y)
    store = _empty_store(periods, test)
    for name in periods:
        store[name]["mu"][:] = level[name]
    store["test"]["mu"][:] = test_level
    town_components = []
    for index, town in enumerate(towns, start=1):
        print(f"[{index}/{len(towns)}] catboost and mixture {town}", flush=True)
        try:
            record = _fit_or_load_town(stack, town, periods, residual, test, store, resume)
        except Exception:
            print(f"failed on {town}", flush=True)
            raise
        town_components.append(record)
        (HERE / "progress.json").write_text(
            json.dumps({"finished_towns": [item["town"] for item in town_components]}, indent=2) + "\n",
            encoding="utf-8",
        )
    for name in ("train", "search", "confirm", "test"):
        for field in ("mu", "t", "gmm_gap", "latitude", "longitude", "latitude_raw", "longitude_raw"):
            if not np.isfinite(store[name][field]).all():
                raise SystemExit(f"{name} {field} is not finite")
    if int((~np.isfinite(store["train"]["lease_age"])).sum()) != 1:
        raise SystemExit("training lease age should be null for exactly one row")
    print("fitting shared MLP", flush=True)
    network, epoch, x_center, x_spread, y_center, y_spread = _fit_mlp(stack._MLP, store, rents)
    _save_mlp(network, epoch, x_center, x_spread, y_center, y_spread)
    scored = _score(network, store, rents, x_center, x_spread, y_center, y_spread)
    test_pred = _mlp_predict(network, store["test"], x_center, x_spread, y_center, y_spread)
    changed = _changed_protected(before)
    if changed:
        raise SystemExit(f"protected files changed: {changed[:20]}")
    scores = {
        "train_rmse": scored["train_rmse"],
        "validation_rmse": scored["validation_rmse"],
        "validation_months": ["2024-10", "2024-11", "2024-12"],
        "validation_rows": 8274,
        "test_rmse": scored["test_rmse"],
        "test_months": ["2025-01", "2025-02", "2025-03"],
        "test_rows": 8924,
        "pooled_oct2024_mar2025_rmse": scored["pooled_oct2024_mar2025_rmse"],
        "pooled_rows": 17198,
        "pooled_was_stopping_criterion": False,
        "best_epoch": epoch,
        "mlp_stopped_on": "validation_2024-10_to_2024-12",
        "catboost_stopped_on": "validation_2024-10_to_2024-12",
        "catboost_features_on_train_rows": "in_sample_full_town_model",
        "mixture_gaps_on_train_rows": "in_sample",
        "train_rmse_uses_in_sample_gaps": True,
        "mlp_fields": list(MLP_FIELDS),
        "cat_features": list(CAT_FEATURES),
        "cat_categorical": list(CAT_CATEGORICAL),
        "removed_from_mlp": ["fee_amount", "lease_commence_year", "flat_age_from_lease"],
        "lease_age": "approval_year - lease_commence_year; null and indicator 1 when lease year is later; row kept",
        "building_key": "normalized BLOCK|STREET via src.features.keys.block_key",
        "flat_type_key": "src.features.source_fields.flat_type_key; flat_type_ord kept for CatBoost, mixture, and MLP",
        "mu_and_gaps_multiplied_by_area": False,
        "rnn_checkpoint": "mean_residual_model/reginal_mean_model/rnn/h16_l3",
        "rnn_retrained": False,
        "seed": SEED,
        "previous_validation_rmse": 448.2357450773328,
        "previous_test_rmse": 462.8737151230945,
        "teammate_pooled_catboost_rmse": 454.29,
        "pooled_beats_454_29": bool(scored["pooled_oct2024_mar2025_rmse"] < 454.29),
        "town_components": town_components,
        "prediction_csv": PREDICTION_PATH.relative_to(ROOT).as_posix(),
        "protected_files_unchanged": True,
        "fit_months_end": split.fit_months[-1],
    }
    (HERE / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    _write_kaggle(PREDICTION_PATH, test_pred)
    print(
        f"validation 2024-10..12 RMSE {scores['validation_rmse']:.4f} "
        f"(stopping split, {scores['validation_rows']} rows), "
        f"test 2025-01..03 RMSE {scores['test_rmse']:.4f} ({scores['test_rows']} rows), "
        f"pooled 2024-10..2025-03 RMSE {scores['pooled_oct2024_mar2025_rmse']:.4f} "
        f"({scores['pooled_rows']} rows, not a stopping criterion), epoch {epoch}",
        flush=True,
    )
    print(f"train RMSE {scores['train_rmse']:.4f} uses in-sample CatBoost and mixture gaps", flush=True)
    print(f"wrote {(HERE / 'metrics.json').relative_to(ROOT).as_posix()}", flush=True)
    print(f"wrote {PREDICTION_PATH.relative_to(ROOT).as_posix()}", flush=True)
    print("protected weights and prediction csvs unchanged", flush=True)


def predict() -> None:
    """Write the private-test file from the saved town models and MLP."""
    stack = _load_stack()
    _split, periods, test = _prepare()
    _level, test_level, _rents, _residual = _levels(stack, periods, test)
    towns = stack._towns(periods)
    missing = sorted(set(test["TOWN"].astype(str)) - set(towns))
    if missing:
        raise SystemExit(f"private-test towns absent from the fit months: {missing}")
    store = _empty_store(periods, test)
    store["test"]["mu"][:] = test_level
    for town in towns:
        print(f"predict {town}", flush=True)
        if not _town_ready(town):
            raise SystemExit(f"missing saved town artifacts for {town}")
        masks = {name: part["TOWN"].astype(str).to_numpy() == town for name, part in periods.items()}
        test_mask = test["TOWN"].astype(str).to_numpy() == town
        fit = periods["train"].loc[masks["train"]].reset_index(drop=True)
        search = periods["search"].loc[masks["search"]].reset_index(drop=True)
        confirm = periods["confirm"].loc[masks["confirm"]].reset_index(drop=True)
        test_town = test.loc[test_mask].reset_index(drop=True)
        stack._scale_coords(fit, [search, confirm, test_town])
        stack._write_coords(store, "test", test_mask, test_town)
        folder = _town_dir(town)
        bundle = load_model(folder)
        preparer = joblib.load(folder / "preprocessor.joblib")
        mixture = joblib.load(_gmm_path(town))
        if int(test_mask.sum()):
            store["test"]["t"][test_mask] = _predict_catboost(bundle, preparer, test_town)
            stack._write_mixture(store, "test", test_mask, test_town, mixture)
    checkpoint = torch.load(HERE / "mlp" / "model.pt", map_location="cpu", weights_only=False)
    if list(checkpoint["mlp_fields"]) != MLP_FIELDS:
        raise SystemExit("saved MLP fields do not match this script")
    network = stack._MLP(len(MLP_FIELDS))
    network.load_state_dict(checkpoint["state"])
    network.eval()
    test_pred = _mlp_predict(
        network,
        store["test"],
        checkpoint["x_center"],
        checkpoint["x_spread"],
        checkpoint["y_center"],
        checkpoint["y_spread"],
    )
    _write_kaggle(PREDICTION_PATH, test_pred)
    print(f"wrote {PREDICTION_PATH.relative_to(ROOT).as_posix()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a town whose saved CatBoost, preprocessor, and mixture match this feature list.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Write the private-test CSV from saved weights. Does not refit.",
    )
    args = parser.parse_args()
    if args.predict:
        predict()
    else:
        main(resume=args.resume)

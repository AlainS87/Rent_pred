"""One CatBoost in place of the shared MLP. The RNN, town CatBoost, and mixture stay frozen.

The tree predicts MONTHLY_RENT. Its inputs are the core fields from the
teammates' preprocessing, the auxiliary families this repository can build,
and three separate unit-price columns: the saved RNN town-month mean, the
saved town CatBoost gap, and the saved mixture gap. Those three are not
multiplied by floor area.

Training months run through 2024-09. October-December 2024 is the only split
that stops iterations. January-March 2025 is scored after that choice. The
pooled October 2024-March 2025 RMSE is not a stopping criterion.

    python -u "mean_residual_model/catboost_head/run.py"

Training-row CatBoost gaps are predictions of each town's saved full model
on the rows it was fit on. They are in-sample, not out of fold. Mixture gaps
on those rows are in-sample as well.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mean_residual_model.reginal_mean_model.rnn.model import predict_saved_rnn3
from mean_residual_model.reginal_mean_model.sequence_store import town_slug
from mean_residual_model.run import SEED, _check_months, _write_kaggle
from models.catboost.model import _category_columns, _fill_missing_categories, _prepare_pair
from models.catboost.preprocess import fit_preprocessor, transform_frame
from src.features.build_features import DATA, FEATURE_COLUMNS
from src.features.group_means import _month_start, _smoothed, _window_totals, _with_month
from src.features.keys import block_key
from src.features.source_fields import flat_type_key
from src.features.spatial import (
    TRUNK_OPEN_PREFIXES,
    _prepare_mrt,
    _subset,
    count_within,
    nearest,
)
from src.pipeline.data import load_joined, time_split
from src.pipeline.fit import rmse

HERE = Path(__file__).resolve().parent
STACK_PATH = ROOT / "mean_residual_model" / "multi_stage_model(lstm+catboost+mlp)" / "run.py"
PREDICTION_PATH = ROOT / "outputs" / "predictions" / "mean_residual_catboost_head.csv"
DROPPED_AUX = ("fee_amount", "furnished_bin")
OPEN_POOL_COLUMNS = (
    "dist_nearest_open_mrt_km",
    "nearest_open_mrt_code",
    "nearest_open_mrt_name",
    "nearest_open_mrt_line",
    "nearest_open_mrt_is_interchange",
    "n_open_mrt_within_0_5km",
    "n_open_mrt_within_1km",
    "n_open_mrt_within_2km",
    "log_dist_nearest_open_mrt_km",
)
CORE_COLUMNS = (
    "approval_year",
    "approval_month",
    "approval_month_index",
    "building_key",
    "flat_type_key",
    "lease_age",
    "lease_age_invalid",
)
GAP_COLUMNS = ("rnn_unit_mean", "catboost_unit_gap", "mixture_unit_gap")
CATEGORICAL_COLUMNS = (
    "building_key",
    "flat_type_key",
    "postal_code",
    "subzone",
    "planning_area",
    "region",
    "nearest_open_mrt_code",
    "nearest_open_mrt_name",
    "nearest_open_mrt_line",
    "nearest_primary_name",
    "nearest_mall_name",
    "nearest_ns_station_code",
    "nearest_ns_station_name",
    "nearest_ew_station_code",
    "nearest_ew_station_name",
    "nearest_ne_station_code",
    "nearest_ne_station_name",
    "nearest_cc_station_code",
    "nearest_cc_station_name",
    "nearest_dt_station_code",
    "nearest_dt_station_name",
    "nearest_te_station_code",
    "nearest_te_station_name",
    "nearest_secondary_name",
    "nearest_school_name",
    "town_key",
    "block_key",
    "street_key",
    "flat_model_key",
)
PREPROCESS = {
    "categorical": "native",
    "numeric_impute": "median",
    "rare_min_count": 30,
    "scale": "none",
}
# The auxiliary section says the trees share one recorded setting and does not
# state its numbers. Depth and learning rate are the stated fallback. The
# iteration cap and the 50-round patience are the town CatBoost budget, so
# validation loss can stop the tree before the cap.
CAT_ARCH = {
    "early_stopping_rounds": 50,
    "learning_rate": 0.05,
    "max_depth": 8,
    "n_estimators": 2000,
}
HYPHEN_FLAT_TYPES = {"1-room", "2-room", "3-room", "4-room", "5-room", "executive"}
PROTECTED_DIRS = [
    ROOT / "mean_residual_model" / "reginal_mean_model" / "rnn" / "h16_l3",
    ROOT / "mean_residual_model" / "multi_stage_model(lstm+catboost+mlp)",
    ROOT / "mean_residual_model" / "aligned_preprocess",
    ROOT / "mean_residual_model" / "refit",
]
PROTECTED_FILES = [
    ROOT / "outputs" / "predictions" / "mean_residual_multistage.csv",
    ROOT / "outputs" / "predictions" / "mean_residual_multistage_refit.csv",
    ROOT / "outputs" / "predictions" / "mean_residual_aligned_preprocess.csv",
]
FAMILIES_INCLUDED = [
    "core derived: approval year, calendar month, month index, BLOCK|STREET building id, hyphenated flat type, lease age and invalid indicator",
    "source fields except constant fee_amount and furnished_bin: town, block, street, flat-type rank, flat-model rank and name, floor area, lease commencement year",
    "blocks and age: coordinates, postal code, maximum floor, completion year, subzone, planning area, region, building age, lease-versus-completion gap, flat age from lease, central-region flag, age times central",
    "MRT line distances, planned stations, and future-line distances: stored trunk-prefix columns",
    "MRT open distance, code, name, line, interchange flag, 0.5/1/2 km counts, and log distance: recomputed for this run on every snapshot-open station code, including LRT and open non-trunk prefixes",
    "schools and malls: nearest primary, secondary, all-school, and mall distances and names, plus the stored radius counts",
    "density and CBD: other HDB blocks within 500 m, town centroid, Raffles Place",
    "COE: lagged premiums, changes, bid ratios, category-A three-month mean, category-C premium",
    "stocks: lagged bank basket and property basket only",
    "group means: five stored previous-three-month means reused; nearest-MRT mean recomputed with the same shrinkage on the snapshot-open station code",
    "three frozen unit columns, not multiplied by floor area: RNN town-month mean, town CatBoost gap, mixture gap",
]
NOT_BUILT = [
    "the 1,878-column candidate universe, the 492-column compact table, and the 1,595 month-panel candidates: that builder is not in the repository",
    "individual stock symbols, OHLC, volume, and stock PCA",
    "adjusted-price sensitivity",
    "school type, gender, session, address, transport text, and separate JC/CI blocks beyond the stored all-schools nearest distance and count",
    "deduplicated MRT entities: distances use station code rows, which is what the existing nearest function does",
    "cutoff-frozen macro: the stored COE and stock columns are the lagged month T-1 values",
    "sine and cosine of calendar month",
]


def _features() -> list[str]:
    auxiliary = [
        column
        for column in FEATURE_COLUMNS
        if column not in DROPPED_AUX and column not in CORE_COLUMNS
    ]
    names = [*CORE_COLUMNS, *auxiliary, *GAP_COLUMNS]
    if len(names) != len(set(names)):
        raise SystemExit("feature list repeats a column")
    banned = {"fee_amount", "FEE", "FURNISHED", "furnished_bin", "MONTHLY_RENT", "RENT_APPROVAL_DATE"}
    if banned & set(names):
        raise SystemExit(f"banned column in the feature list: {banned & set(names)}")
    return names


def _load_columns() -> list[str]:
    extra = ["RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE"]
    auxiliary = [column for column in FEATURE_COLUMNS if column not in DROPPED_AUX]
    return list(dict.fromkeys([*extra, *auxiliary]))


def _load_stack():
    if not STACK_PATH.is_file():
        raise SystemExit(f"missing stack script: {STACK_PATH}")
    spec = importlib.util.spec_from_file_location("multistage_stack_frozen", STACK_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {STACK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hyphen_flat_type(values: pd.Series) -> pd.Series:
    """The report standardizes one- to five-room labels to the hyphenated form."""
    text = flat_type_key(values)
    return text.str.replace(r"^([1-5]) room$", r"\1-room", regex=True)


def _add_core(frame: pd.DataFrame, invalid_rows: int) -> pd.DataFrame:
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
    age = (year - lease).astype("float64").mask(invalid, np.nan)
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
    out["flat_type_key"] = _hyphen_flat_type(out["FLAT_TYPE"])
    if out["building_key"].isna().any() or out["flat_type_key"].isna().any():
        raise SystemExit("building key or flat type is missing")
    labels = set(out["flat_type_key"].astype(str).unique())
    if labels != HYPHEN_FLAT_TYPES:
        raise SystemExit(f"flat types are {sorted(labels)}")
    return out


def _fingerprint() -> dict[str, list[int]]:
    roots = list(PROTECTED_DIRS)
    roots.extend(sorted((ROOT / "models").glob("*/best")))
    roots.extend(sorted((ROOT / "models").glob("*/final")))
    found: dict[str, list[int]] = {}
    for root in roots:
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


def _block_origin(blocks: pd.DataFrame) -> pd.DataFrame:
    origin = pd.DataFrame(
        {
            "block_key": block_key(blocks["BLOCK"], blocks["STREET"]),
            "latitude": pd.to_numeric(blocks["LATITUDE"], errors="coerce"),
            "longitude": pd.to_numeric(blocks["LONGITUDE"], errors="coerce"),
        }
    )
    if not origin["block_key"].is_unique:
        raise SystemExit("block file block_key is not unique")
    if origin["latitude"].isna().any() or origin["longitude"].isna().any():
        raise SystemExit("block coordinates are missing")
    return origin


def _open_pool_table(blocks: pd.DataFrame, mrt: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Nearest open station per block. ``trunk`` matches the stored columns."""
    origin = _block_origin(blocks)
    stations = _prepare_mrt(mrt)
    if mode == "trunk":
        pool = _subset(stations, TRUNK_OPEN_PREFIXES, "open")
    elif mode == "snapshot_open":
        pool = stations[stations["status"].eq("open")].copy()
    else:
        raise SystemExit(f"unknown open-pool mode: {mode}")
    if pool.empty:
        raise SystemExit(f"{mode} station pool is empty")
    code_counts = stations.groupby("name_key")["code"].nunique()
    interchange_names = set(code_counts[code_counts >= 2].index)
    lat = origin["latitude"].to_numpy()
    lon = origin["longitude"].to_numpy()
    dist, index = nearest(
        lat,
        lon,
        pool["LATITUDE"].to_numpy(),
        pool["LONGITUDE"].to_numpy(),
        pool["code"].to_numpy(),
    )
    chosen = pool.iloc[index].reset_index(drop=True)
    prefix = chosen["line_prefix"].astype("string")
    line = prefix.mask(prefix.isin(["EW", "CG"]), "EW")
    out = origin[["block_key"]].copy()
    out["dist_nearest_open_mrt_km"] = dist
    out["nearest_open_mrt_code"] = chosen["code"].astype("string").to_numpy()
    out["nearest_open_mrt_name"] = chosen["name_key"].astype("string").to_numpy()
    out["nearest_open_mrt_line"] = line.astype("string").to_numpy()
    out["nearest_open_mrt_is_interchange"] = chosen["name_key"].isin(interchange_names).astype(np.int64).to_numpy()
    out["log_dist_nearest_open_mrt_km"] = np.log(dist + 0.001)
    pool_lat = pool["LATITUDE"].to_numpy()
    pool_lon = pool["LONGITUDE"].to_numpy()
    out["n_open_mrt_within_0_5km"] = count_within(lat, lon, pool_lat, pool_lon, 0.5)
    out["n_open_mrt_within_1km"] = count_within(lat, lon, pool_lat, pool_lon, 1.0)
    out["n_open_mrt_within_2km"] = count_within(lat, lon, pool_lat, pool_lon, 2.0)
    out.attrs["pool_rows"] = int(len(pool))
    out.attrs["prefixes"] = [str(value) for value in sorted(pool["line_prefix"].dropna().astype(str).unique())]
    return out


def _mapped(frame: pd.DataFrame, table: pd.DataFrame, column: str) -> pd.Series:
    lookup = table.set_index("block_key")[column]
    mapped = frame["building_key"].map(lookup)
    if mapped.isna().any():
        raise SystemExit(f"{column} missed {int(mapped.isna().sum())} rental rows")
    return mapped


def _verify_trunk(frame: pd.DataFrame, trunk: pd.DataFrame) -> None:
    for column in OPEN_POOL_COLUMNS:
        stored = frame[column]
        fresh = _mapped(frame, trunk, column)
        if column in {"nearest_open_mrt_code", "nearest_open_mrt_name", "nearest_open_mrt_line"}:
            mismatch = int((fresh.astype("string") != stored.astype("string")).sum())
            if mismatch:
                raise SystemExit(f"trunk rebuild disagrees on {column} for {mismatch} rows")
            continue
        diff = float(np.nanmax(np.abs(fresh.to_numpy(dtype=np.float64) - stored.to_numpy(dtype=np.float64))))
        if diff > 1e-4:
            raise SystemExit(f"trunk rebuild of {column} differs by {diff}")


def _shrunk_group_mean(dates, keys, history_dates, history_keys, history_rents) -> np.ndarray:
    history = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": np.asarray(history_dates),
            "MONTHLY_RENT": np.asarray(history_rents, dtype=np.float64),
            "group_key": pd.Series(history_keys, dtype="string"),
        }
    )
    history = _with_month(history)
    months = _month_start(pd.Series(np.asarray(dates)))
    query = pd.Series(keys, dtype="string")
    global_sum, global_n = _window_totals(months, None, history, None)
    group_sum, group_n = _window_totals(months, query, history, "group_key")
    return _smoothed(group_sum, group_n, global_sum, global_n)


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    both_nan = np.isnan(left) & np.isnan(right)
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return float("inf")
    if both_nan.all():
        return 0.0
    return float(np.max(np.abs(left[~both_nan] - right[~both_nan])))


def _prepare_open_pool(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    print("checking snapshot-open station distances against the stored trunk columns", flush=True)
    blocks = pd.read_csv(DATA / "auxiliary" / "sg-hdb-block.csv", dtype={"BLOCK": str, "STREET": str, "TOWN": str})
    mrt = pd.read_csv(DATA / "auxiliary" / "sg-mrt-stations.csv")
    trunk = _open_pool_table(blocks, mrt, "trunk")
    opened = _open_pool_table(blocks, mrt, "snapshot_open")
    _verify_trunk(train, trunk)
    _verify_trunk(test, trunk)
    stored_codes = train["nearest_open_mrt_code"]
    rebuilt_old = _shrunk_group_mean(
        train["RENT_APPROVAL_DATE"],
        stored_codes,
        train["RENT_APPROVAL_DATE"],
        stored_codes,
        train["MONTHLY_RENT"],
    )
    stored_mean = train["mean_rent_nearest_mrt"].to_numpy(dtype=np.float64)
    mean_diff = _max_abs_diff(rebuilt_old, stored_mean)
    if mean_diff > 0.05:
        raise SystemExit(f"rebuilt trunk MRT group mean differs from the stored column by {mean_diff}")
    fresh_codes = _mapped(train, opened, "nearest_open_mrt_code").astype("string")
    changed = int((fresh_codes != stored_codes.astype("string")).sum())
    january = train["RENT_APPROVAL_DATE"].astype("string").eq("2021-01")
    print(
        f"open stations {opened.attrs['pool_rows']} versus trunk {trunk.attrs['pool_rows']}; "
        f"train rows whose nearest code changes {changed}; "
        f"trunk group-mean max abs diff {mean_diff:.6f}",
        flush=True,
    )
    return {
        "opened": opened,
        "open_station_codes": int(opened.attrs["pool_rows"]),
        "trunk_station_codes": int(trunk.attrs["pool_rows"]),
        "open_prefixes": list(opened.attrs["prefixes"]),
        "train_rows_nearest_code_changed": changed,
        "trunk_group_mean_max_abs_diff": mean_diff,
        "january_rows": int(january.sum()),
    }


def _apply_open_pool(frame: pd.DataFrame, opened: pd.DataFrame) -> None:
    for column in OPEN_POOL_COLUMNS:
        mapped = _mapped(frame, opened, column)
        if column in {"nearest_open_mrt_code", "nearest_open_mrt_name", "nearest_open_mrt_line"}:
            frame[column] = mapped.astype("string").to_numpy()
        else:
            frame[column] = mapped.to_numpy(dtype=np.float64)


def _apply_open_means(train: pd.DataFrame, test: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> None:
    """Replace only the nearest-MRT mean. The other five stored means stay put.

    ``train``, ``test``, and each split already carry the snapshot-open station
    code. The split is not a contiguous slice of the labelled file, so each
    piece is scored from its own dates and codes against the full labelled history.
    """
    history_dates = train["RENT_APPROVAL_DATE"]
    history_keys = train["nearest_open_mrt_code"]
    history_rents = train["MONTHLY_RENT"]
    new_train = _shrunk_group_mean(history_dates, history_keys, history_dates, history_keys, history_rents)
    january = train["RENT_APPROVAL_DATE"].astype("string").eq("2021-01").to_numpy()
    if not np.isnan(new_train[january]).all():
        raise SystemExit("January 2021 nearest-MRT mean is not null")
    train["mean_rent_nearest_mrt"] = new_train
    labelled = len(train)
    covered = 0
    for part in parts.values():
        covered += len(part)
        part["mean_rent_nearest_mrt"] = _shrunk_group_mean(
            part["RENT_APPROVAL_DATE"],
            part["nearest_open_mrt_code"],
            history_dates,
            history_keys,
            history_rents,
        )
    if covered != labelled:
        raise SystemExit(f"split parts cover {covered} rows, expected {labelled}")
    test["mean_rent_nearest_mrt"] = _shrunk_group_mean(
        test["RENT_APPROVAL_DATE"],
        test["nearest_open_mrt_code"],
        history_dates,
        history_keys,
        history_rents,
    )


def _frozen_gaps(stack, periods, test) -> None:
    print("loading saved RNN", flush=True)
    level = {name: predict_saved_rnn3(part) for name, part in periods.items()}
    test_level = predict_saved_rnn3(test)
    towns = stack._towns(periods)
    missing = sorted(set(test["TOWN"].astype(str)) - set(towns))
    if missing:
        raise SystemExit(f"private-test towns absent from the fit months: {missing}")
    gaps = {
        name: {
            "t": np.full(len(part), np.nan, dtype=np.float64),
            "gmm_gap": np.full(len(part), np.nan, dtype=np.float64),
        }
        for name, part in periods.items()
    }
    gaps["test"] = {
        "t": np.full(len(test), np.nan, dtype=np.float64),
        "gmm_gap": np.full(len(test), np.nan, dtype=np.float64),
    }
    for index, town in enumerate(towns, start=1):
        print(f"[{index}/{len(towns)}] frozen catboost and mixture {town}", flush=True)
        masks = {name: part["TOWN"].astype(str).to_numpy() == town for name, part in periods.items()}
        test_mask = test["TOWN"].astype(str).to_numpy() == town
        fit = periods["train"].loc[masks["train"]].reset_index(drop=True).copy()
        search = periods["search"].loc[masks["search"]].reset_index(drop=True).copy()
        confirm = periods["confirm"].loc[masks["confirm"]].reset_index(drop=True).copy()
        test_town = test.loc[test_mask].reset_index(drop=True).copy()
        stack._scale_coords(fit, [search, confirm, test_town])
        stack._apply_frozen_catboost(town, fit, search, confirm, test_town, masks, test_mask, gaps)
        mixture = joblib.load(stack.HERE / "gmm" / f"{town_slug(town)}.joblib")
        if mixture.get("kind") != "conditional_location_flat_per_sqm":
            raise SystemExit(f"{town} mixture is not the saved per-square-metre mixture")
        for name, part, mask in (
            ("train", fit, masks["train"]),
            ("search", search, masks["search"]),
            ("confirm", confirm, masks["confirm"]),
            ("test", test_town, test_mask),
        ):
            if int(np.asarray(mask).sum()) == 0:
                continue
            stack._write_mixture(gaps, name, mask, part, mixture)
    for name, values in level.items():
        periods[name]["rnn_unit_mean"] = values
        periods[name]["catboost_unit_gap"] = gaps[name]["t"]
        periods[name]["mixture_unit_gap"] = gaps[name]["gmm_gap"]
    test["rnn_unit_mean"] = test_level
    test["catboost_unit_gap"] = gaps["test"]["t"]
    test["mixture_unit_gap"] = gaps["test"]["gmm_gap"]
    for name, part in (*periods.items(), ("test", test)):
        for column in GAP_COLUMNS:
            values = part[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise SystemExit(f"{name} {column} is not finite")
        area = part["floor_area_sqm"].to_numpy(dtype=np.float64)
        if np.any(area <= 0):
            raise SystemExit(f"{name} floor area is not positive")
        # A unit gap is a few SGD per square metre. Multiplying by area would
        # push the median absolute gap into the hundreds.
        if float(np.median(np.abs(part["catboost_unit_gap"]))) > 30:
            raise SystemExit(f"{name} CatBoost gap looks like it was multiplied by floor area")
        median_mu = float(np.median(part["rnn_unit_mean"]))
        if median_mu > 200:
            raise SystemExit(f"{name} RNN output looks like monthly rent, not rent per square metre")


def _fit_head(x_fit, y_fit, x_search, y_search):
    x_fit, y_fit = _prepare_pair(x_fit, y_fit)
    x_search, y_search = _prepare_pair(x_search, y_search)
    categories = _category_columns(x_fit)
    x_fit = _fill_missing_categories(x_fit)
    x_search = _fill_missing_categories(x_search)
    params = {
        "depth": int(CAT_ARCH["max_depth"]),
        "learning_rate": float(CAT_ARCH["learning_rate"]),
        "iterations": int(CAT_ARCH["n_estimators"]),
        "early_stopping_rounds": int(CAT_ARCH["early_stopping_rounds"]),
        "loss_function": "RMSE",
        "random_seed": SEED,
        "thread_count": -1,
        "allow_writing_files": False,
    }
    model = CatBoostRegressor(**params)
    model.fit(
        x_fit,
        y_fit,
        eval_set=(x_search, y_search),
        cat_features=categories,
        use_best_model=True,
        verbose=100,
    )
    return model


def _check_kinds(frame: pd.DataFrame, features: list[str]) -> None:
    from src.pipeline.preprocess import _is_categorical

    for column in features:
        if column in GAP_COLUMNS:
            continue
        detected = _is_categorical(column, frame[column])
        expected = column in CATEGORICAL_COLUMNS
        if detected != expected:
            raise SystemExit(
                f"{column} categorical detected {detected}, expected {expected}, dtype {frame[column].dtype}"
            )


def _assert_roles(preparer, fit: pd.DataFrame, features: list[str]) -> None:
    expected = [column for column in features if column in CATEGORICAL_COLUMNS]
    if preparer.categorical_columns_ != expected:
        raise SystemExit(f"categorical columns are {preparer.categorical_columns_}")
    numeric = [column for column in features if column not in CATEGORICAL_COLUMNS]
    if preparer.numeric_columns_ != numeric:
        raise SystemExit(f"numeric columns are {preparer.numeric_columns_}")
    matrix = transform_frame(preparer, fit)
    for column in GAP_COLUMNS:
        if not np.allclose(matrix[column].to_numpy(dtype=np.float64), fit[column].to_numpy(dtype=np.float64)):
            raise SystemExit(f"{column} was scaled or imputed")
    invalid_pos = np.flatnonzero(fit["lease_age_invalid"].to_numpy(dtype=np.float64) == 1.0)
    if len(invalid_pos) != 1:
        raise SystemExit(f"expected one invalid lease row in the training months, found {len(invalid_pos)}")
    if float(matrix["lease_age_invalid"].iloc[int(invalid_pos[0])]) != 1.0:
        raise SystemExit("the invalid lease indicator was not kept")
    if not np.isfinite(matrix["lease_age"].to_numpy(dtype=np.float64)).all():
        raise SystemExit("lease age was not imputed")
    levels = set(matrix["flat_type_key"].cat.categories)
    if levels != HYPHEN_FLAT_TYPES:
        raise SystemExit(f"flat type categories are {sorted(levels)}")
    if any("|" not in str(level) for level in matrix["building_key"].cat.categories):
        raise SystemExit("a building category is not BLOCK|STREET")


def main() -> None:
    features = _features()
    print("checking protected files", flush=True)
    before = _fingerprint()
    print(f"protected files {len(before)}", flush=True)
    print("loading tables", flush=True)
    train = _add_core(load_joined("train", _load_columns()), 1)
    test = _add_core(load_joined("test", _load_columns()), 0)
    if len(train) != 150000 or len(test) != 50000:
        raise SystemExit(f"row counts are train {len(train)} test {len(test)}")
    if train["building_key"].nunique() != 9406:
        raise SystemExit(f"train buildings {train['building_key'].nunique()}, expected 9406")
    _check_kinds(train, features)
    _check_kinds(test, features)
    pool_info = _prepare_open_pool(train, test)
    train.attrs["opened"] = pool_info.pop("opened")
    split = time_split(train, 3, None)
    _check_months(split)
    counts = (len(split.fit), len(split.search), len(split.confirm))
    if counts != (132802, 8274, 8924):
        raise SystemExit(f"split sizes are {counts}")
    if int(split.fit["lease_age"].isna().sum()) != 1:
        raise SystemExit("the invalid lease row is not in the training months")
    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    stack = _load_stack()
    _frozen_gaps(stack, periods, test)
    print("applying snapshot-open station columns", flush=True)
    _apply_open_pool(train, train.attrs["opened"])
    _apply_open_pool(test, train.attrs["opened"])
    for part in periods.values():
        _apply_open_pool(part, train.attrs["opened"])
    _apply_open_means(train, test, periods)
    # The split copies were taken before the full-table code replacement, so
    # the means above use the replaced codes. Check the training-month window
    # still excludes that month.
    if int(periods["train"]["lease_age_invalid"].sum()) != 1:
        raise SystemExit("the invalid lease indicator was lost")
    preparer = fit_preprocessor(PREPROCESS, periods["train"], periods["train"]["MONTHLY_RENT"], features)
    _assert_roles(preparer, periods["train"], features)
    print(
        f"fitting one CatBoost on {len(features)} columns, depth {CAT_ARCH['max_depth']}, "
        f"learning rate {CAT_ARCH['learning_rate']}, seed {SEED}",
        flush=True,
    )
    started = time.perf_counter()
    model = _fit_head(
        transform_frame(preparer, periods["train"]),
        periods["train"]["MONTHLY_RENT"],
        transform_frame(preparer, periods["search"]),
        periods["search"]["MONTHLY_RENT"],
    )
    elapsed = time.perf_counter() - started
    best_index = int(model.get_best_iteration())
    tree_count = int(model.tree_count_)
    print(f"best iteration index {best_index}, trees {tree_count}, seconds {elapsed:.1f}", flush=True)
    search_pred = model.predict(_fill_missing_categories(transform_frame(preparer, periods["search"])))
    confirm_pred = model.predict(_fill_missing_categories(transform_frame(preparer, periods["confirm"])))
    train_pred = model.predict(_fill_missing_categories(transform_frame(preparer, periods["train"])))
    test_pred = model.predict(_fill_missing_categories(transform_frame(preparer, test)))
    search_y = periods["search"]["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    confirm_y = periods["confirm"]["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    train_y = periods["train"]["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    pooled_y = np.concatenate([search_y, confirm_y])
    pooled_p = np.concatenate([search_pred, confirm_pred])
    if len(pooled_y) != 17198:
        raise SystemExit(f"pooled rows are {len(pooled_y)}, expected 17198")
    scores = {
        "validation_rmse": rmse(search_y, search_pred),
        "validation_months": ["2024-10", "2024-11", "2024-12"],
        "validation_rows": 8274,
        "test_rmse": rmse(confirm_y, confirm_pred),
        "test_months": ["2025-01", "2025-02", "2025-03"],
        "test_rows": 8924,
        "pooled_oct2024_mar2025_rmse": rmse(pooled_y, pooled_p),
        "pooled_rows": 17198,
        "pooled_was_stopping_criterion": False,
        "train_rmse": rmse(train_y, train_pred),
        "train_rmse_uses_in_sample_gaps": True,
        "catboost_unit_gap_on_training_rows": "in_sample_full_town_model",
        "mixture_unit_gap_on_training_rows": "in_sample",
        "best_iteration_index": best_index,
        "tree_count": tree_count,
        "iterations_cap": CAT_ARCH["n_estimators"],
        "early_stopping_rounds": CAT_ARCH["early_stopping_rounds"],
        "learning_rate": CAT_ARCH["learning_rate"],
        "depth": CAT_ARCH["max_depth"],
        "seed": SEED,
        "stopping_split": "validation_2024-10_to_2024-12",
        "target": "MONTHLY_RENT",
        "mu_and_gaps_multiplied_by_area": False,
        "rnn_retrained": False,
        "town_catboost_retrained": False,
        "mixture_retrained": False,
        "mlp_replaced": True,
        "rnn_checkpoint": "mean_residual_model/reginal_mean_model/rnn/h16_l3",
        "frozen_catboost_dir": "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/catboost",
        "frozen_mixture_dir": "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/gmm",
        "previous_mlp_validation_rmse": 448.2357450773328,
        "previous_mlp_test_rmse": 462.8737151230945,
        "teammate_private_test_best_rmse": 432.50,
        "private_test_scored": False,
        "feature_count": len(features),
        "features": features,
        "categorical_columns": list(preparer.categorical_columns_),
        "families_included": FAMILIES_INCLUDED,
        "rebuilt_candidates_not_built": NOT_BUILT,
        "open_pool": {key: value for key, value in pool_info.items() if key != "opened"},
        "dropped_constant_columns": list(DROPPED_AUX),
        "fit_seconds": elapsed,
        "prediction_csv": PREDICTION_PATH.relative_to(ROOT).as_posix(),
    }
    changed = _changed_protected(before)
    if changed:
        raise SystemExit(f"protected files changed: {changed[:20]}")
    scores["protected_files_unchanged"] = True
    HERE.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, HERE / "model.joblib")
    joblib.dump(preparer, HERE / "preprocessor.joblib")
    (HERE / "features.json").write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    (HERE / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    _write_kaggle(PREDICTION_PATH, test_pred)
    print(
        f"validation RMSE {scores['validation_rmse']:.4f} "
        f"(stopping split, {scores['validation_rows']} rows), "
        f"test RMSE {scores['test_rmse']:.4f} ({scores['test_rows']} rows), "
        f"pooled RMSE {scores['pooled_oct2024_mar2025_rmse']:.4f} "
        f"({scores['pooled_rows']} rows, not a stopping criterion), "
        f"trees {tree_count}",
        flush=True,
    )
    print(
        f"train RMSE {scores['train_rmse']:.4f} uses in-sample CatBoost and mixture gaps",
        flush=True,
    )
    print(f"wrote {(HERE / 'metrics.json').relative_to(ROOT).as_posix()}", flush=True)
    print(f"wrote {PREDICTION_PATH.relative_to(ROOT).as_posix()}", flush=True)
    print("protected weights and prediction csvs unchanged", flush=True)


if __name__ == "__main__":
    main()

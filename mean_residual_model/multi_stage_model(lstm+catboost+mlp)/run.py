"""Town LSTM mean, per-town CatBoost, and one numeric MLP.

    python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py"
    python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --frozen-catboost
    python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --residuals-only
    python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --predict
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mean_residual_model.reginal_mean_model.rnn.model import predict_saved_rnn3
from mean_residual_model.reginal_mean_model.sequence_store import town_slug
from mean_residual_model.run import SEED, _check_months, _write_kaggle
from mean_residual_model.regional import with_month
from models.catboost.model import fit_model, load_model, predict_model, save_model
from models.catboost.preprocess import fit_preprocessor, transform_frame
from src.pipeline.data import load_joined, time_split
from src.pipeline.fit import rmse

HERE = Path(__file__).resolve().parent
CAT_FEATURES = [
    "latitude",
    "longitude",
    "block_key",
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
]
MLP_ADDED = [
    "log_dist_nearest_open_mrt_km",
    "n_open_mrt_within_1km",
    "n_mall_within_1km",
    "n_primary_within_1km",
    "n_secondary_within_1km",
    "n_hdb_blocks_within_500m",
]
LOAD_COLUMNS = [
    "RENT_APPROVAL_DATE",
    "TOWN",
    "building_age_years",
    "max_floor",
    "year_completed",
    "flat_age_from_lease",
    "approval_month_index",
    "floor_area_sqm",
    "fee_amount",
    "lease_commence_year",
    *CAT_FEATURES,
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
STRENGTH = 20
START_COMPONENTS = 20
MIN_COMPONENT_ROWS = 30


def _unit_and_area(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    area = part["floor_area_sqm"].to_numpy(dtype=np.float64)
    rent = part["MONTHLY_RENT"].to_numpy(dtype=np.float64)
    return rent / area, area


def main(frozen_catboost: bool = False) -> None:
    frame = load_joined("train", LOAD_COLUMNS)
    split = time_split(frame, 3, None)
    _check_months(split)
    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    level = {name: predict_saved_rnn3(part) for name, part in periods.items()}
    rents = {name: part["MONTHLY_RENT"].to_numpy(dtype=np.float64) for name, part in periods.items()}
    units = {}
    areas = {}
    for name, part in periods.items():
        units[name], areas[name] = _unit_and_area(part)
    residual = {name: units[name] - level[name] for name in periods}
    test = load_joined("test", LOAD_COLUMNS)
    test_level = predict_saved_rnn3(test)
    towns = _towns(periods)
    print(f"towns {len(towns)}")

    store = _empty_store(periods, test)
    for name in periods:
        store[name]["mu"][:] = level[name]
    store["test"]["mu"][:] = test_level
    town_components = []
    for town in towns:
        print(f"{'frozen catboost' if frozen_catboost else 'catboost'} and mixture {town}", flush=True)
        town_components.append(_fit_town(town, periods, residual, test, store, frozen_catboost))

    network, epoch, center, spread = _fit_mlp(store, rents)
    _save_mlp(network, epoch, center, spread, store)
    search_pred = _mlp_predict(network, store["search"], center, spread)
    confirm_pred = _mlp_predict(network, store["confirm"], center, spread)
    test_pred = _mlp_predict(network, store["test"], center, spread)
    scores = {
        "train_rmse": rmse(rents["train"], _mlp_predict(network, store["train"], center, spread)),
        "search_rmse": rmse(rents["search"], search_pred),
        "confirm_rmse": rmse(rents["confirm"], confirm_pred),
        "best_epoch": epoch,
        "gmm": "conditional_location_flat_per_sqm",
        "gmm_min_component_rows": MIN_COMPONENT_ROWS,
        "gmm_start_components": START_COMPONENTS,
        "town_components": town_components,
        "catboost": "frozen_full_model" if frozen_catboost else "refit_oof",
        "mean_model": "rnn_h16_l3_per_sqm",
        "target": "monthly_rent",
        "mlp_stopped_on_search": True,
    }
    (HERE / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    output = ROOT / "outputs" / "predictions" / "mean_residual_multistage.csv"
    _write_kaggle(output, test_pred)
    print(
        f"stack RMSE train {scores['train_rmse']:.4f}, "
        f"search {scores['search_rmse']:.4f}, confirm {scores['confirm_rmse']:.4f}, epoch {epoch}"
    )
    print(f"wrote {output.relative_to(ROOT).as_posix()}")


def fit_residuals() -> None:
    """Fit per-town CatBoost and the mixture on the RNN per-square-metre gap."""
    frame = load_joined("train", LOAD_COLUMNS)
    split = time_split(frame, 3, None)
    _check_months(split)
    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    level = {name: predict_saved_rnn3(part) for name, part in periods.items()}
    rents = {name: part["MONTHLY_RENT"].to_numpy(dtype=np.float64) for name, part in periods.items()}
    units = {}
    areas = {}
    for name, part in periods.items():
        units[name], areas[name] = _unit_and_area(part)
    residual = {name: units[name] - level[name] for name in periods}
    test = load_joined("test", LOAD_COLUMNS)
    towns = _towns(periods)
    print(f"towns {len(towns)}")
    store = _empty_store(periods, test)
    for name in periods:
        store[name]["mu"][:] = level[name]
    town_components = []
    for town in towns:
        print(f"catboost and mixture {town}", flush=True)
        town_components.append(_fit_town(town, periods, residual, test, store, False))
    scores = {
        "mean_model": "rnn_h16_l3_per_sqm",
        "target": "rent_per_sqm_minus_region_mean",
        "town_components": town_components,
        "mlp_retrained": False,
    }
    for split_name in ("search", "confirm"):
        for key, label in (("t", "catboost"), ("gmm_gap", "gmm")):
            unit = level[split_name] + store[split_name][key]
            scores[f"{label}_{split_name}_unit_rmse"] = rmse(units[split_name], unit)
            scores[f"{label}_{split_name}_rent_rmse"] = rmse(rents[split_name], unit * areas[split_name])
    path = HERE / "residuals_metrics.json"
    path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(
        f"search unit RMSE catboost {scores['catboost_search_unit_rmse']:.4f}, "
        f"gmm {scores['gmm_search_unit_rmse']:.4f}"
    )
    print(
        f"search rent RMSE catboost {scores['catboost_search_rent_rmse']:.4f}, "
        f"gmm {scores['gmm_search_rent_rmse']:.4f}"
    )
    print(f"wrote {path.relative_to(ROOT).as_posix()}")


def fit_mlp_saved() -> None:
    """Train the MLP on saved CatBoost and mixture gaps. The output is monthly rent."""
    frame = load_joined("train", LOAD_COLUMNS)
    split = time_split(frame, 3, None)
    _check_months(split)
    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    level = {name: predict_saved_rnn3(part) for name, part in periods.items()}
    rents = {name: part["MONTHLY_RENT"].to_numpy(dtype=np.float64) for name, part in periods.items()}
    test = load_joined("test", LOAD_COLUMNS)
    test_level = predict_saved_rnn3(test)
    towns = _towns(periods)
    print(f"towns {len(towns)}")
    store = _empty_store(periods, test)
    for name in periods:
        store[name]["mu"][:] = level[name]
    store["test"]["mu"][:] = test_level
    for town in towns:
        print(f"load {town}", flush=True)
        masks = {name: part["TOWN"].astype(str).to_numpy() == town for name, part in periods.items()}
        test_mask = test["TOWN"].astype(str).to_numpy() == town
        fit = periods["train"].loc[masks["train"]].reset_index(drop=True)
        search = periods["search"].loc[masks["search"]].reset_index(drop=True)
        confirm = periods["confirm"].loc[masks["confirm"]].reset_index(drop=True)
        test_town = test.loc[test_mask].reset_index(drop=True)
        _scale_coords(fit, [search, confirm, test_town])
        _write_coords(store, "train", masks["train"], fit)
        _write_coords(store, "search", masks["search"], search)
        _write_coords(store, "confirm", masks["confirm"], confirm)
        _write_coords(store, "test", test_mask, test_town)
        _apply_frozen_catboost(town, fit, search, confirm, test_town, masks, test_mask, store)
        mixture = joblib.load(HERE / "gmm" / f"{town_slug(town)}.joblib")
        if mixture.get("kind") != "conditional_location_flat_per_sqm":
            raise SystemExit(f"{town} mixture is not a per-square-metre location mixture")
        for name, part, mask in (
            ("train", fit, masks["train"]),
            ("search", search, masks["search"]),
            ("confirm", confirm, masks["confirm"]),
            ("test", test_town, test_mask),
        ):
            if int(np.asarray(mask).sum()) == 0:
                continue
            _write_mixture(store, name, mask, part, mixture)
    network, epoch, center, spread = _fit_mlp(store, rents)
    _save_mlp(network, epoch, center, spread, store)
    search_pred = _mlp_predict(network, store["search"], center, spread)
    confirm_pred = _mlp_predict(network, store["confirm"], center, spread)
    test_pred = _mlp_predict(network, store["test"], center, spread)
    scores = {
        "train_rmse": rmse(rents["train"], _mlp_predict(network, store["train"], center, spread)),
        "search_rmse": rmse(rents["search"], search_pred),
        "confirm_rmse": rmse(rents["confirm"], confirm_pred),
        "best_epoch": epoch,
        "catboost": "frozen_full_model",
        "gmm": "conditional_location_flat_per_sqm",
        "mean_model": "rnn_h16_l3_per_sqm",
        "mlp_input": "mu_t_gmm_area_separate_raw_coordinates",
        "mlp_added": MLP_ADDED,
        "mlp_stopped_on_search": True,
        "target": "monthly_rent",
    }
    (HERE / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    output = ROOT / "outputs" / "predictions" / "mean_residual_multistage.csv"
    _write_kaggle(output, test_pred)
    print(
        f"stack RMSE train {scores['train_rmse']:.4f}, "
        f"search {scores['search_rmse']:.4f}, confirm {scores['confirm_rmse']:.4f}, epoch {epoch}"
    )
    print(f"wrote {output.relative_to(ROOT).as_posix()}")


def _fit_town(town: str, periods, residual, test, store, frozen_catboost: bool = False) -> dict:
    masks = {name: part["TOWN"].astype(str).to_numpy() == town for name, part in periods.items()}
    test_mask = test["TOWN"].astype(str).to_numpy() == town
    fit = periods["train"].loc[masks["train"]].reset_index(drop=True)
    search = periods["search"].loc[masks["search"]].reset_index(drop=True)
    confirm = periods["confirm"].loc[masks["confirm"]].reset_index(drop=True)
    test_town = test.loc[test_mask].reset_index(drop=True)
    y_fit = residual["train"][masks["train"]]
    y_search = residual["search"][masks["search"]]
    _scale_coords(fit, [search, confirm, test_town])
    _write_coords(store, "train", masks["train"], fit)
    _write_coords(store, "search", masks["search"], search)
    _write_coords(store, "confirm", masks["confirm"], confirm)
    _write_coords(store, "test", test_mask, test_town)

    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    if frozen_catboost:
        _apply_frozen_catboost(town, fit, search, confirm, test_town, masks, test_mask, store)
    else:
        oof = np.zeros(len(fit), dtype=np.float64)
        for train_pos, hold_pos in folds.split(fit):
            bundle, preparer = _catboost(fit.iloc[train_pos], y_fit[train_pos], search, y_search)
            oof[hold_pos] = predict_model(bundle, transform_frame(preparer, fit.iloc[hold_pos]))
        store["train"]["t"][masks["train"]] = oof
        full_bundle, full_preparer = _catboost(fit, y_fit, search, y_search)
        folder = HERE / "catboost" / town_slug(town)
        folder.mkdir(parents=True, exist_ok=True)
        save_model(full_bundle, folder)
        joblib.dump(full_preparer, folder / "preprocessor.joblib")
        store["search"]["t"][masks["search"]] = predict_model(full_bundle, transform_frame(full_preparer, search))
        store["confirm"]["t"][masks["confirm"]] = predict_model(full_bundle, transform_frame(full_preparer, confirm))
        if len(test_town):
            store["test"]["t"][test_mask] = predict_model(full_bundle, transform_frame(full_preparer, test_town))

    mixture = _fit_mixture(fit, y_fit)
    print(f"  mixture k {mixture['n_components']} sizes {mixture['hard_sizes']}", flush=True)
    gmm_path = HERE / "gmm" / f"{town_slug(town)}.joblib"
    gmm_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mixture, gmm_path)
    for name, frame, mask, target in (
        ("train", fit, masks["train"], y_fit),
        ("search", search, masks["search"], None),
        ("confirm", confirm, masks["confirm"], None),
        ("test", test_town, test_mask, None),
    ):
        if int(np.asarray(mask).sum()) == 0:
            continue
        _write_mixture(store, name, mask, frame, mixture)

    return {"n_components": mixture["n_components"], "town": town}


def _apply_frozen_catboost(town, fit, search, confirm, test_town, masks, test_mask, store) -> None:
    """Predict with the saved full model. The 5-fold models were not stored."""
    folder = HERE / "catboost" / town_slug(town)
    model_path = folder / "model.joblib"
    preparer_path = folder / "preprocessor.joblib"
    if not model_path.is_file() or not preparer_path.is_file():
        raise SystemExit(f"missing frozen CatBoost for {town}: {folder}")
    bundle = load_model(folder)
    preparer = joblib.load(preparer_path)
    pieces = (
        ("train", fit, masks["train"]),
        ("search", search, masks["search"]),
        ("confirm", confirm, masks["confirm"]),
        ("test", test_town, test_mask),
    )
    for name, frame, mask in pieces:
        if int(np.asarray(mask).sum()) == 0:
            continue
        store[name]["t"][mask] = predict_model(bundle, transform_frame(preparer, frame))


def _catboost(fit, y_fit, search, y_search):
    preparer = fit_preprocessor(PREPROCESS, fit, y_fit, CAT_FEATURES)
    return fit_model(
        transform_frame(preparer, fit),
        y_fit,
        transform_frame(preparer, search),
        y_search,
        CAT_ARCH,
        SEED,
    ), preparer


def _fit_mixture(fit: pd.DataFrame, residual: np.ndarray) -> dict:
    """Fit a mixture on scaled location, flat ranks, and the fit-month residual."""
    ordinal = _ordinal_stats(fit)
    scaled = np.column_stack(
        [
            fit["latitude"].to_numpy(dtype=np.float64),
            fit["longitude"].to_numpy(dtype=np.float64),
            *_scaled_ordinals(fit, ordinal),
            np.asarray(residual, dtype=np.float64),
        ]
    )
    count = START_COMPONENTS
    mixture = None
    hard_sizes = None
    while count >= 1:
        if len(fit) < count:
            count -= 1
            continue
        candidate = GaussianMixture(
            n_components=count,
            covariance_type="full",
            random_state=SEED,
            reg_covar=1e-6,
        )
        candidate.fit(scaled)
        labels = candidate.predict(scaled)
        sizes = np.bincount(labels, minlength=count)
        if count == 1 or np.all(sizes >= MIN_COMPONENT_ROWS):
            mixture = candidate
            hard_sizes = [int(size) for size in sizes]
            break
        count -= 1
    if mixture is None or hard_sizes is None:
        raise RuntimeError("location mixture did not fit")
    return {
        "gmm": mixture,
        "hard_sizes": hard_sizes,
        "kind": "conditional_location_flat_per_sqm",
        "n_components": int(mixture.n_components),
        "ordinal": ordinal,
    }


def _ordinal_stats(fit: pd.DataFrame) -> dict:
    stats = {}
    for column in ("flat_type_ord", "flat_model_ord"):
        values = fit[column].astype(float)
        center = float(values.mean())
        spread = float(values.std(ddof=0)) or 1.0
        stats[column] = {"center": center, "spread": spread}
    return stats


def _scaled_ordinals(frame: pd.DataFrame, stats: dict) -> list[np.ndarray]:
    scaled = []
    for column in ("flat_type_ord", "flat_model_ord"):
        values = frame[column].astype(float).to_numpy(dtype=np.float64)
        center = stats[column]["center"]
        spread = stats[column]["spread"]
        scaled.append((values - center) / spread)
    return scaled


def _conditional(gmm: GaussianMixture, coords: np.ndarray) -> np.ndarray:
    """Conditional residual mean given scaled location and flat ranks."""
    width = coords.shape[1]
    log_terms = []
    heights = []
    for index in range(gmm.n_components):
        mean = gmm.means_[index]
        cov = gmm.covariances_[index]
        location_cov = cov[:width, :width] + 1e-8 * np.eye(width)
        solved = np.linalg.solve(location_cov, (coords - mean[:width]).T).T
        heights.append(mean[width] + solved @ cov[width, :width])
        sign, logdet = np.linalg.slogdet(location_cov)
        quad = np.sum((coords - mean[:width]) * solved, axis=1)
        log_density = -0.5 * (width * np.log(2.0 * np.pi) + logdet + quad)
        if sign <= 0:
            log_density = np.full(len(coords), -1e30)
        log_terms.append(np.log(float(gmm.weights_[index]) + 1e-300) + log_density)
    stacked = np.column_stack(log_terms)
    mix = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    mix_sum = mix.sum(axis=1)
    responsibility = mix / mix_sum[:, None]
    return np.sum(responsibility * np.column_stack(heights), axis=1)


def _write_mixture(store, name: str, mask, frame: pd.DataFrame, mixture: dict) -> None:
    coords = np.column_stack(
        [
            frame["latitude"].to_numpy(dtype=np.float64),
            frame["longitude"].to_numpy(dtype=np.float64),
            *_scaled_ordinals(frame, mixture["ordinal"]),
        ]
    )
    store[name]["gmm_gap"][mask] = _conditional(mixture["gmm"], coords)


def _write_encodings(store, masks, test_mask, fit, y_fit, search, confirm, test_town, oof_folds) -> None:
    for column, key in (("FLAT_TYPE", "flat_type"), ("FLAT_MODEL", "flat_model")):
        labels = _labels(fit[column], column)
        oof = np.zeros(len(fit), dtype=np.float64)
        for train_pos, hold_pos in oof_folds.split(fit):
            oof[hold_pos] = _shrunk(labels.iloc[train_pos], y_fit[train_pos], labels.iloc[hold_pos])
        store["train"][key][masks["train"]] = oof
        store["search"][key][masks["search"]] = _shrunk(labels, y_fit, _labels(search[column], column))
        store["confirm"][key][masks["confirm"]] = _shrunk(labels, y_fit, _labels(confirm[column], column))
        if len(test_town):
            store["test"][key][test_mask] = _shrunk(labels, y_fit, _labels(test_town[column], column))


def _shrunk(train_labels: pd.Series, train_y: np.ndarray, apply_labels: pd.Series) -> np.ndarray:
    global_mean = float(np.mean(train_y))
    frame = pd.DataFrame({"label": train_labels.to_numpy(), "target": train_y})
    grouped = frame.groupby("label")["target"].agg(["mean", "size"])
    shrunk = (grouped["size"] * grouped["mean"] + STRENGTH * global_mean) / (grouped["size"] + STRENGTH)
    return apply_labels.map(shrunk).fillna(global_mean).to_numpy(dtype=np.float64)


def _labels(series: pd.Series, column: str) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    if column == "FLAT_TYPE":
        text = text.str.replace("-", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return text.fillna("__MISSING__")


def _fit_mlp(store, rents):
    train_x, center, spread = _matrix(store["train"], fit=True)
    torch.manual_seed(SEED)
    y = rents["train"]
    y_center = float(np.mean(y))
    y_spread = float(np.std(y)) or 1.0
    network = _MLP(train_x.shape[1])
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
        prediction = _mlp_predict(network, store["search"], y_center, y_spread, center, spread)
        score = rmse(rents["search"], prediction)
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
    return network, best_epoch, (center, y_center), (spread, y_spread)


def _fit_mlp_epochs(store, rents, epochs: int):
    """Train the shared MLP for a frozen epoch count. There is no search score."""
    if epochs < 1:
        raise SystemExit(f"MLP epoch count must be at least 1, got {epochs}")
    train_x, center, spread = _matrix(store["train"], fit=True)
    torch.manual_seed(SEED)
    y = rents["train"]
    y_center = float(np.mean(y))
    y_spread = float(np.std(y)) or 1.0
    network = _MLP(train_x.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    features = torch.tensor(train_x, dtype=torch.float32)
    target = torch.tensor((y - y_center) / y_spread, dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    for epoch in range(1, epochs + 1):
        network.train()
        order = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], 1024):
            batch = order[start : start + 1024]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(features[batch]), target[batch])
            loss.backward()
            optimizer.step()
        print(f"mlp epoch {epoch}/{epochs}", flush=True)
    network.eval()
    return network, epochs, (center, y_center), (spread, y_spread)


class _MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _mlp_predict(network, block, y_center, y_spread, center=None, spread=None) -> np.ndarray:
    if isinstance(y_center, tuple):
        center, y_center = y_center
        spread, y_spread = y_spread
    values = _matrix(block, center=center, spread=spread)
    network.eval()
    with torch.no_grad():
        scaled = network(torch.tensor(values, dtype=torch.float32)).numpy()
    return scaled * y_spread + y_center


def _matrix(block, fit=False, center=None, spread=None):
    values = np.column_stack(
        [
            block["mu"],
            block["t"],
            block["gmm_gap"],
            block["floor_area"],
            block["fee"],
            block["lease"],
            block["age"],
            block["month_index"],
            block["latitude"],
            block["longitude"],
            block["latitude_raw"],
            block["longitude_raw"],
            block["flat_type_ord"],
            block["flat_model_ord"],
            block["max_floor"],
            block["year_completed"],
            block["flat_age"],
            block["dist_mrt"],
            block["dist_mall"],
            block["dist_primary"],
            block["dist_secondary"],
            block["log_dist_mrt"],
            block["n_mrt_1km"],
            block["n_mall_1km"],
            block["n_primary_1km"],
            block["n_secondary_1km"],
            block["n_blocks_500m"],
        ]
    ).astype(np.float64)
    if fit:
        center = np.nanmedian(values, axis=0)
        filled = np.where(np.isfinite(values), values, center)
        spread = np.std(filled, axis=0)
        spread[spread == 0] = 1.0
        return (filled - center) / spread, center, spread
    filled = np.where(np.isfinite(values), values, center)
    return (filled - center) / spread


def _save_mlp(network, epoch, center, spread, store) -> None:
    folder = HERE / "mlp"
    folder.mkdir(parents=True, exist_ok=True)
    x_center, y_center = center
    x_spread, y_spread = spread
    torch.save(
        {
            "epoch": epoch,
            "state": {key: value.detach().cpu() for key, value in network.state_dict().items()},
            "x_center": x_center,
            "x_spread": x_spread,
            "y_center": y_center,
            "y_spread": y_spread,
        },
        folder / "model.pt",
    )
    del store


def _empty_store(periods, test) -> dict:
    def block(n_rows: int) -> dict:
        return {
            "age": np.zeros(n_rows, dtype=np.float64),
            "dist_mall": np.zeros(n_rows, dtype=np.float64),
            "dist_mrt": np.zeros(n_rows, dtype=np.float64),
            "dist_primary": np.zeros(n_rows, dtype=np.float64),
            "dist_secondary": np.zeros(n_rows, dtype=np.float64),
            "fee": np.zeros(n_rows, dtype=np.float64),
            "flat_age": np.zeros(n_rows, dtype=np.float64),
            "flat_model_ord": np.zeros(n_rows, dtype=np.float64),
            "flat_type_ord": np.zeros(n_rows, dtype=np.float64),
            "floor_area": np.zeros(n_rows, dtype=np.float64),
            "gmm_gap": np.zeros(n_rows, dtype=np.float64),
            "latitude": np.zeros(n_rows, dtype=np.float64),
            "latitude_raw": np.zeros(n_rows, dtype=np.float64),
            "lease": np.zeros(n_rows, dtype=np.float64),
            "log_dist_mrt": np.zeros(n_rows, dtype=np.float64),
            "longitude": np.zeros(n_rows, dtype=np.float64),
            "longitude_raw": np.zeros(n_rows, dtype=np.float64),
            "max_floor": np.zeros(n_rows, dtype=np.float64),
            "n_blocks_500m": np.zeros(n_rows, dtype=np.float64),
            "n_mall_1km": np.zeros(n_rows, dtype=np.float64),
            "n_mrt_1km": np.zeros(n_rows, dtype=np.float64),
            "n_primary_1km": np.zeros(n_rows, dtype=np.float64),
            "n_secondary_1km": np.zeros(n_rows, dtype=np.float64),
            "month_index": np.zeros(n_rows, dtype=np.float64),
            "mu": np.zeros(n_rows, dtype=np.float64),
            "t": np.full(n_rows, np.nan, dtype=np.float64),
            "year_completed": np.zeros(n_rows, dtype=np.float64),
        }

    store = {name: block(len(part)) for name, part in periods.items()}
    store["test"] = block(len(test))
    for name, part in periods.items():
        _copy_plain(store[name], part)
    _copy_plain(store["test"], test)
    return store


def _copy_plain(block, frame: pd.DataFrame) -> None:
    block["floor_area"][:] = frame["floor_area_sqm"].to_numpy(dtype=np.float64)
    block["fee"][:] = frame["fee_amount"].to_numpy(dtype=np.float64)
    block["lease"][:] = frame["lease_commence_year"].to_numpy(dtype=np.float64)
    block["age"][:] = frame["building_age_years"].to_numpy(dtype=np.float64)
    block["month_index"][:] = frame["approval_month_index"].to_numpy(dtype=np.float64)
    block["flat_type_ord"][:] = frame["flat_type_ord"].to_numpy(dtype=np.float64)
    block["flat_model_ord"][:] = frame["flat_model_ord"].to_numpy(dtype=np.float64)
    block["max_floor"][:] = frame["max_floor"].to_numpy(dtype=np.float64)
    block["year_completed"][:] = frame["year_completed"].to_numpy(dtype=np.float64)
    block["flat_age"][:] = frame["flat_age_from_lease"].to_numpy(dtype=np.float64)
    block["dist_mrt"][:] = frame["dist_nearest_open_mrt_km"].to_numpy(dtype=np.float64)
    block["dist_mall"][:] = frame["dist_nearest_mall_km"].to_numpy(dtype=np.float64)
    block["dist_primary"][:] = frame["dist_nearest_primary_km"].to_numpy(dtype=np.float64)
    block["dist_secondary"][:] = frame["dist_nearest_secondary_km"].to_numpy(dtype=np.float64)
    block["log_dist_mrt"][:] = frame["log_dist_nearest_open_mrt_km"].to_numpy(dtype=np.float64)
    block["n_mrt_1km"][:] = frame["n_open_mrt_within_1km"].to_numpy(dtype=np.float64)
    block["n_mall_1km"][:] = frame["n_mall_within_1km"].to_numpy(dtype=np.float64)
    block["n_primary_1km"][:] = frame["n_primary_within_1km"].to_numpy(dtype=np.float64)
    block["n_secondary_1km"][:] = frame["n_secondary_within_1km"].to_numpy(dtype=np.float64)
    block["n_blocks_500m"][:] = frame["n_hdb_blocks_within_500m"].to_numpy(dtype=np.float64)


def _write_coords(store, name, mask, frame: pd.DataFrame) -> None:
    store[name]["latitude"][mask] = frame["latitude"].to_numpy(dtype=np.float64)
    store[name]["longitude"][mask] = frame["longitude"].to_numpy(dtype=np.float64)
    store[name]["latitude_raw"][mask] = frame["latitude_raw"].to_numpy(dtype=np.float64)
    store[name]["longitude_raw"][mask] = frame["longitude_raw"].to_numpy(dtype=np.float64)


def _scale_coords(fit: pd.DataFrame, others: list[pd.DataFrame]) -> None:
    for frame in [fit, *others]:
        if len(frame) == 0:
            continue
        frame["latitude_raw"] = frame["latitude"].astype(float)
        frame["longitude_raw"] = frame["longitude"].astype(float)
    for column in ("latitude", "longitude"):
        values = fit[column].astype(float)
        center = float(values.mean())
        spread = float(values.std(ddof=0)) or 1.0
        fit[column] = (values - center) / spread
        for frame in others:
            if len(frame) == 0:
                continue
            frame[column] = (frame[column].astype(float) - center) / spread


def predict() -> None:
    """Write test predictions from the saved LSTM, CatBoost, mixture, and MLP."""
    frame = load_joined("train", LOAD_COLUMNS)
    split = time_split(frame, 3, None)
    _check_months(split)
    periods = {"train": split.fit, "search": split.search, "confirm": split.confirm}
    test = load_joined("test", LOAD_COLUMNS)
    test_level = predict_saved_rnn3(test)
    towns = _towns(periods)
    missing = sorted(set(test["TOWN"].astype(str)) - set(towns))
    if missing:
        raise SystemExit(f"test towns absent from the fit months: {missing}")
    store = _empty_store(periods, test)
    store["test"]["mu"][:] = test_level
    for town in towns:
        print(f"predict {town}", flush=True)
        masks = {name: part["TOWN"].astype(str).to_numpy() == town for name, part in periods.items()}
        test_mask = test["TOWN"].astype(str).to_numpy() == town
        fit = periods["train"].loc[masks["train"]].reset_index(drop=True)
        search = periods["search"].loc[masks["search"]].reset_index(drop=True)
        confirm = periods["confirm"].loc[masks["confirm"]].reset_index(drop=True)
        test_town = test.loc[test_mask].reset_index(drop=True)
        _scale_coords(fit, [search, confirm, test_town])
        _write_coords(store, "test", test_mask, test_town)
        _apply_frozen_catboost(town, fit, search, confirm, test_town, masks, test_mask, store)
        mixture = joblib.load(HERE / "gmm" / f"{town_slug(town)}.joblib")
        if mixture.get("kind") != "conditional_location_flat_per_sqm":
            raise SystemExit(f"{town} mixture is not a per-square-metre location mixture")
        if int(test_mask.sum()):
            _write_mixture(store, "test", test_mask, test_town, mixture)
    checkpoint = torch.load(HERE / "mlp" / "model.pt", map_location="cpu", weights_only=False)
    width = int(checkpoint["state"]["net.0.weight"].shape[1])
    network = _MLP(width)
    network.load_state_dict(checkpoint["state"])
    network.eval()
    test_pred = _mlp_predict(
        network,
        store["test"],
        checkpoint["y_center"],
        checkpoint["y_spread"],
        checkpoint["x_center"],
        checkpoint["x_spread"],
    )
    output = ROOT / "outputs" / "predictions" / "mean_residual_multistage.csv"
    _write_kaggle(output, test_pred)
    print(
        f"test predictions {len(test_pred)}, "
        f"mean {float(np.mean(test_pred)):.1f}, "
        f"min {float(np.min(test_pred)):.1f}, max {float(np.max(test_pred)):.1f}"
    )
    print(f"wrote {output.relative_to(ROOT).as_posix()}")


def _towns(periods) -> list[str]:
    fit_towns = set(periods["train"]["TOWN"].astype(str).unique())
    search_towns = set(periods["search"]["TOWN"].astype(str).unique())
    confirm_towns = set(periods["confirm"]["TOWN"].astype(str).unique())
    if fit_towns != search_towns or fit_towns != confirm_towns:
        raise SystemExit("fit, search, and confirm do not cover the same towns")
    return sorted(fit_towns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-catboost",
        action="store_true",
        help="Load each town's saved CatBoost and refit only the mixture and the MLP.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Write test predictions from the saved weights. Does not refit.",
    )
    parser.add_argument(
        "--residuals-only",
        action="store_true",
        help="Fit per-town CatBoost and the mixture on the RNN per-square-metre gap. Does not fit the MLP.",
    )
    parser.add_argument(
        "--mlp-only",
        action="store_true",
        help="Train the MLP on the saved CatBoost and mixture. mu, the CatBoost gap, the mixture gap, and floor area enter as separate inputs.",
    )
    args = parser.parse_args()
    if args.predict:
        predict()
    elif args.residuals_only:
        fit_residuals()
    elif args.mlp_only:
        fit_mlp_saved()
    else:
        main(frozen_catboost=args.frozen_catboost)

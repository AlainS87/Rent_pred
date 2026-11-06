"""Internal numerical support selected by the local experiments."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from torch import nn
from src.paths import PROJECT_ROOT as ROOT, ARTIFACT_ROOT
from .mean_residual_model.reginal_mean_model.rnn.model import predict_saved_rnn3
from .mean_residual_model.reginal_mean_model.sequence_store import town_slug
from .mean_residual_model.run import SEED, _check_months
from .models.catboost.model import fit_model, load_model, predict_model, save_model
from .models.catboost.preprocess import fit_preprocessor, transform_frame
from .src.pipeline.data import load_joined, time_split
from .src.pipeline.fit import rmse
HERE = ARTIFACT_ROOT / 'local' / 'stack'
CAT_FEATURES = ['latitude', 'longitude', 'block_key', 'street_key', 'flat_type_ord', 'flat_model_ord', 'flat_model_key', 'nearest_open_mrt_code', 'dist_nearest_open_mrt_km', 'log_dist_nearest_open_mrt_km', 'nearest_mall_name', 'dist_nearest_mall_km', 'nearest_primary_name', 'dist_nearest_primary_km', 'nearest_secondary_name', 'dist_nearest_secondary_km']
LOAD_COLUMNS = ['RENT_APPROVAL_DATE', 'TOWN', 'building_age_years', 'max_floor', 'year_completed', 'flat_age_from_lease', 'approval_month_index', 'floor_area_sqm', 'fee_amount', 'lease_commence_year', *CAT_FEATURES]
PREPROCESS = {'categorical': 'native', 'numeric_impute': 'median', 'rare_min_count': 30, 'scale': 'none'}
CAT_ARCH = {'early_stopping_rounds': 50, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 1000}
START_COMPONENTS = 20
MIN_COMPONENT_ROWS = 30

def _catboost(fit, y_fit, search, y_search):
    preparer = fit_preprocessor(PREPROCESS, fit, y_fit, CAT_FEATURES)
    return (fit_model(transform_frame(preparer, fit), y_fit, transform_frame(preparer, search), y_search, CAT_ARCH, SEED), preparer)

def _fit_mixture(fit: pd.DataFrame, residual: np.ndarray) -> dict:
    """Fit a mixture on scaled location, flat ranks, and the fit-month residual."""
    ordinal = _ordinal_stats(fit)
    scaled = np.column_stack([fit['latitude'].to_numpy(dtype=np.float64), fit['longitude'].to_numpy(dtype=np.float64), *_scaled_ordinals(fit, ordinal), np.asarray(residual, dtype=np.float64)])
    count = START_COMPONENTS
    mixture = None
    hard_sizes = None
    while count >= 1:
        if len(fit) < count:
            count -= 1
            continue
        candidate = GaussianMixture(n_components=count, covariance_type='full', random_state=SEED, reg_covar=1e-06)
        candidate.fit(scaled)
        labels = candidate.predict(scaled)
        sizes = np.bincount(labels, minlength=count)
        if count == 1 or np.all(sizes >= MIN_COMPONENT_ROWS):
            mixture = candidate
            hard_sizes = [int(size) for size in sizes]
            break
        count -= 1
    if mixture is None or hard_sizes is None:
        raise RuntimeError('location mixture did not fit')
    return {'gmm': mixture, 'hard_sizes': hard_sizes, 'kind': 'conditional_location_flat_per_sqm', 'n_components': int(mixture.n_components), 'ordinal': ordinal}

def _ordinal_stats(fit: pd.DataFrame) -> dict:
    stats = {}
    for column in ('flat_type_ord', 'flat_model_ord'):
        values = fit[column].astype(float)
        center = float(values.mean())
        spread = float(values.std(ddof=0)) or 1.0
        stats[column] = {'center': center, 'spread': spread}
    return stats

def _scaled_ordinals(frame: pd.DataFrame, stats: dict) -> list[np.ndarray]:
    scaled = []
    for column in ('flat_type_ord', 'flat_model_ord'):
        values = frame[column].astype(float).to_numpy(dtype=np.float64)
        center = stats[column]['center']
        spread = stats[column]['spread']
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
        location_cov = cov[:width, :width] + 1e-08 * np.eye(width)
        solved = np.linalg.solve(location_cov, (coords - mean[:width]).T).T
        heights.append(mean[width] + solved @ cov[width, :width])
        sign, logdet = np.linalg.slogdet(location_cov)
        quad = np.sum((coords - mean[:width]) * solved, axis=1)
        log_density = -0.5 * (width * np.log(2.0 * np.pi) + logdet + quad)
        if sign <= 0:
            log_density = np.full(len(coords), -1e+30)
        log_terms.append(np.log(float(gmm.weights_[index]) + 1e-300) + log_density)
    stacked = np.column_stack(log_terms)
    mix = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    mix_sum = mix.sum(axis=1)
    responsibility = mix / mix_sum[:, None]
    return np.sum(responsibility * np.column_stack(heights), axis=1)

def _write_mixture(store, name: str, mask, frame: pd.DataFrame, mixture: dict) -> None:
    coords = np.column_stack([frame['latitude'].to_numpy(dtype=np.float64), frame['longitude'].to_numpy(dtype=np.float64), *_scaled_ordinals(frame, mixture['ordinal'])])
    store[name]['gmm_gap'][mask] = _conditional(mixture['gmm'], coords)

def _fit_mlp(store, rents):
    train_x, center, spread = _matrix(store['train'], fit=True)
    torch.manual_seed(SEED)
    y = rents['train']
    y_center = float(np.mean(y))
    y_spread = float(np.std(y)) or 1.0
    network = _MLP(train_x.shape[1])
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    features = torch.tensor(train_x, dtype=torch.float32)
    target = torch.tensor((y - y_center) / y_spread, dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    best_state = None
    best_rmse = float('inf')
    best_epoch = 0
    stall = 0
    for epoch in range(1, 101):
        network.train()
        order = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], 1024):
            batch = order[start:start + 1024]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(network(features[batch]), target[batch])
            loss.backward()
            optimizer.step()
        prediction = _mlp_predict(network, store['search'], y_center, y_spread, center, spread)
        score = rmse(rents['search'], prediction)
        if score < best_rmse - 1e-06:
            best_rmse = score
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            best_epoch = epoch
            stall = 0
        else:
            stall += 1
            if stall >= 10:
                break
    if best_state is None:
        raise RuntimeError('MLP produced no weights')
    network.load_state_dict(best_state)
    network.eval()
    return (network, best_epoch, (center, y_center), (spread, y_spread))

class _MLP(nn.Module):

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

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
    area = block['floor_area']
    values = np.column_stack([block['mu'] * area, block['t'] * area, block['gmm_gap'] * area, block['floor_area'], block['fee'], block['lease'], block['age'], block['month_index'], block['latitude'], block['longitude'], block['flat_type_ord'], block['flat_model_ord'], block['max_floor'], block['year_completed'], block['flat_age'], block['dist_mrt'], block['dist_mall'], block['dist_primary'], block['dist_secondary']]).astype(np.float64)
    if fit:
        center = np.nanmedian(values, axis=0)
        filled = np.where(np.isfinite(values), values, center)
        spread = np.std(filled, axis=0)
        spread[spread == 0] = 1.0
        return ((filled - center) / spread, center, spread)
    filled = np.where(np.isfinite(values), values, center)
    return (filled - center) / spread

def _save_mlp(network, epoch, center, spread, store) -> None:
    folder = HERE / 'mlp'
    folder.mkdir(parents=True, exist_ok=True)
    x_center, y_center = center
    x_spread, y_spread = spread
    torch.save({'epoch': epoch, 'state': {key: value.detach().cpu() for key, value in network.state_dict().items()}, 'x_center': x_center, 'x_spread': x_spread, 'y_center': y_center, 'y_spread': y_spread}, folder / 'model.pt')
    del store

def _empty_store(periods, test) -> dict:

    def block(n_rows: int) -> dict:
        return {'age': np.zeros(n_rows, dtype=np.float64), 'dist_mall': np.zeros(n_rows, dtype=np.float64), 'dist_mrt': np.zeros(n_rows, dtype=np.float64), 'dist_primary': np.zeros(n_rows, dtype=np.float64), 'dist_secondary': np.zeros(n_rows, dtype=np.float64), 'fee': np.zeros(n_rows, dtype=np.float64), 'flat_age': np.zeros(n_rows, dtype=np.float64), 'flat_model_ord': np.zeros(n_rows, dtype=np.float64), 'flat_type_ord': np.zeros(n_rows, dtype=np.float64), 'floor_area': np.zeros(n_rows, dtype=np.float64), 'gmm_gap': np.zeros(n_rows, dtype=np.float64), 'latitude': np.zeros(n_rows, dtype=np.float64), 'lease': np.zeros(n_rows, dtype=np.float64), 'longitude': np.zeros(n_rows, dtype=np.float64), 'max_floor': np.zeros(n_rows, dtype=np.float64), 'month_index': np.zeros(n_rows, dtype=np.float64), 'mu': np.zeros(n_rows, dtype=np.float64), 't': np.full(n_rows, np.nan, dtype=np.float64), 'year_completed': np.zeros(n_rows, dtype=np.float64)}
    store = {name: block(len(part)) for name, part in periods.items()}
    store['test'] = block(len(test))
    for name, part in periods.items():
        _copy_plain(store[name], part)
    _copy_plain(store['test'], test)
    return store

def _copy_plain(block, frame: pd.DataFrame) -> None:
    block['floor_area'][:] = frame['floor_area_sqm'].to_numpy(dtype=np.float64)
    block['fee'][:] = frame['fee_amount'].to_numpy(dtype=np.float64)
    block['lease'][:] = frame['lease_commence_year'].to_numpy(dtype=np.float64)
    block['age'][:] = frame['building_age_years'].to_numpy(dtype=np.float64)
    block['month_index'][:] = frame['approval_month_index'].to_numpy(dtype=np.float64)
    block['flat_type_ord'][:] = frame['flat_type_ord'].to_numpy(dtype=np.float64)
    block['flat_model_ord'][:] = frame['flat_model_ord'].to_numpy(dtype=np.float64)
    block['max_floor'][:] = frame['max_floor'].to_numpy(dtype=np.float64)
    block['year_completed'][:] = frame['year_completed'].to_numpy(dtype=np.float64)
    block['flat_age'][:] = frame['flat_age_from_lease'].to_numpy(dtype=np.float64)
    block['dist_mrt'][:] = frame['dist_nearest_open_mrt_km'].to_numpy(dtype=np.float64)
    block['dist_mall'][:] = frame['dist_nearest_mall_km'].to_numpy(dtype=np.float64)
    block['dist_primary'][:] = frame['dist_nearest_primary_km'].to_numpy(dtype=np.float64)
    block['dist_secondary'][:] = frame['dist_nearest_secondary_km'].to_numpy(dtype=np.float64)

def _write_coords(store, name, mask, frame: pd.DataFrame) -> None:
    store[name]['latitude'][mask] = frame['latitude'].to_numpy(dtype=np.float64)
    store[name]['longitude'][mask] = frame['longitude'].to_numpy(dtype=np.float64)

def _scale_coords(fit: pd.DataFrame, others: list[pd.DataFrame]) -> None:
    for column in ('latitude', 'longitude'):
        values = fit[column].astype(float)
        center = float(values.mean())
        spread = float(values.std(ddof=0)) or 1.0
        fit[column] = (values - center) / spread
        for frame in others:
            if len(frame) == 0:
                continue
            frame[column] = (frame[column].astype(float) - center) / spread

def _towns(periods) -> list[str]:
    fit_towns = set(periods['train']['TOWN'].astype(str).unique())
    search_towns = set(periods['search']['TOWN'].astype(str).unique())
    confirm_towns = set(periods['confirm']['TOWN'].astype(str).unique())
    if fit_towns != search_towns or fit_towns != confirm_towns:
        raise SystemExit('fit, search, and confirm do not cover the same towns')
    return sorted(fit_towns)

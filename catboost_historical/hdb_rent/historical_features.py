"""Strictly prior-month rental features. Only the supplied reference labels are indexed.

For a sample from month M the index returns statistics over transactions from
months strictly earlier than M. This rule applies to training, validation,
and test rows. Same-month transactions are never used as history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LEVELS = {
    'block': ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE'],
    'street': ['TOWN', 'STREET', 'FLAT_TYPE'],
    'town': ['TOWN', 'FLAT_TYPE'],
}
F1_COLS = ['hr_block_n', 'hr_block_median', 'hr_block_mean', 'hr_block_std', 'hr_block_age_months']
F2_COLS = ['hr_street_n', 'hr_street_median', 'hr_street_age_months',
           'hr_town_n', 'hr_town_median']
F3_COLS = ([f'hr_{lvl}_{m}m_n' for lvl in LEVELS for m in (3, 6, 12)] +
           [f'hr_{lvl}_{m}m_median' for lvl in LEVELS for m in (3, 6, 12)] +
           ['hr_block_minus_town', 'hr_street_minus_town',
            'hr_block_12m_minus_town_12m', 'hr_town_3m_minus_12m'])
F4_COLS = ['hr_comps_n', 'hr_comps_median', 'hr_comps_weighted_mean',
           'hr_comps_nearest_km', 'hr_comps_12m_n', 'hr_comps_12m_median']
ALL_COLS = F1_COLS + F2_COLS + F3_COLS + F4_COLS


def ordinals(df: pd.DataFrame) -> np.ndarray:
    if not df['RENT_APPROVAL_DATE'].astype(str).str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('Expected YYYY-MM month values')
    return pd.PeriodIndex(df['RENT_APPROVAL_DATE'].astype(str), freq='M').asi8


@dataclass
class GroupHistory:
    months: np.ndarray
    rents: np.ndarray
    prefix: np.ndarray

    @classmethod
    def create(cls, months, rents):
        ix = np.argsort(months, kind='stable')
        dates = np.asarray(months, dtype=np.int64)[ix]
        ys = np.asarray(rents, dtype=float)[ix]
        return cls(dates, ys, np.r_[0.0, np.cumsum(ys)])

    def before(self, query_month: int, window: Optional[int] = None):
        right = int(np.searchsorted(self.months, query_month, side='left'))
        left = (int(np.searchsorted(self.months, query_month - window, side='left'))
                if window is not None else 0)
        return self.rents[left:right], self.months[left:right]


def _stat(vals, months, query_month, mode):
    n = len(vals)
    if mode == 'basic':
        if not n:
            return [0., np.nan, np.nan, np.nan, np.nan]
        return [float(n), float(np.median(vals)), float(np.mean(vals)),
                float(np.std(vals)), float(query_month - months[-1])]
    if mode == 'street':
        return ([float(n), float(np.median(vals)), float(query_month - months[-1])]
                if n else [0., np.nan, np.nan])
    if mode == 'town':
        return [float(n), float(np.median(vals))] if n else [0., np.nan]
    raise ValueError(mode)


class HistoricRents:
    """Indexed references and sparse, cached point-in-time queries.

    `reference` must contain only labels permitted at the fold's training cutoff.
    Neither validation nor test targets are accepted here.
    """
    def __init__(self, reference: pd.DataFrame, targets: np.ndarray,
                 max_neighbors: int = 24, max_radius_km: float = 1.2):
        if len(reference) != len(targets):
            raise ValueError('History reference/target length mismatch')
        self.reference = reference.reset_index(drop=True)
        self.targets = np.asarray(targets, dtype=float)
        if not np.isfinite(self.targets).all() or (self.targets <= 0).any():
            raise ValueError('Bad reference rents')
        self.months = ordinals(self.reference)
        self.series = {}
        for level, cols in LEVELS.items():
            hist = {}
            for key, ix in self.reference.groupby(cols, sort=False, dropna=False).indices.items():
                key = key if isinstance(key, tuple) else (key,)
                hist[key] = GroupHistory.create(self.months[ix], self.targets[ix])
            self.series[level] = hist
        self.max_neighbors = max_neighbors
        self.max_radius_km = max_radius_km
        self._build_addresses()

    def _build_addresses(self):
        df = self.reference
        self.address_data = {}
        for row in df[['TOWN', 'BLOCK', 'STREET', 'LATITUDE', 'LONGITUDE']].itertuples(index=False, name=None):
            town, block, street, lat, lon = row
            key = (town, block, street)
            if key in self.address_data:
                continue
            if pd.notna(lat) and pd.notna(lon) and np.isfinite(float(lat)) and np.isfinite(float(lon)):
                self.address_data[key] = (float(lat), float(lon))
        self.addr_keys = list(self.address_data)
        if self.addr_keys:
            coords = np.array([self.address_data[k] for k in self.addr_keys]) * [111.19, 107.5]
            self.tree = cKDTree(coords)
        else:
            self.tree = None
        self._nearby_cache = {}

    def _neighbors(self, address: tuple, lat: float, lon: float):
        if address in self._nearby_cache:
            return self._nearby_cache[address]
        if self.tree is None or not np.isfinite([lat, lon]).all():
            return []
        k = min(len(self.addr_keys), self.max_neighbors + 1)
        dist, ix = self.tree.query(np.array([lat * 111.19, lon * 107.5]), k=k)
        dist = np.atleast_1d(dist)
        ix = np.atleast_1d(ix)
        pairs = [(self.addr_keys[int(i)], float(d)) for d, i in zip(dist, ix)
                 if np.isfinite(d) and d <= self.max_radius_km and self.addr_keys[int(i)] != address]
        self._nearby_cache[address] = pairs[:self.max_neighbors]
        return self._nearby_cache[address]

    def materialize(self, queries: pd.DataFrame, tier: int = 4) -> pd.DataFrame:
        """Return numerical columns, aligned with queries' original row order."""
        if tier not in (0, 1, 2, 3, 4):
            raise ValueError('tier must be 0..4')
        q = queries.reset_index(drop=True)
        months = ordinals(q)
        out = pd.DataFrame(index=np.arange(len(q)))
        if tier == 0:
            return out
        active = ['block'] if tier == 1 else list(LEVELS)
        for level in active:
            self._level_features(q, months, out, level, tier)
        if tier >= 3:
            out['hr_block_minus_town'] = out['hr_block_median'] - out['hr_town_median']
            out['hr_street_minus_town'] = out['hr_street_median'] - out['hr_town_median']
            out['hr_block_12m_minus_town_12m'] = (out['hr_block_12m_median'] -
                                                 out['hr_town_12m_median'])
            out['hr_town_3m_minus_12m'] = out['hr_town_3m_median'] - out['hr_town_12m_median']
        if tier >= 4:
            self._comps_features(q, months, out)
        expected = feature_columns(tier)
        if set(out.columns) != set(expected):
            raise RuntimeError(f'Feature columns mismatch: {set(out.columns) ^ set(expected)}')
        return out[expected].copy()

    def _level_features(self, q, months, out, level, tier):
        columns = LEVELS[level]
        sizes = 5 if level == 'block' else 3 if level == 'street' else 2
        basic = np.full((len(q), sizes), np.nan)
        for key, ids in q.groupby(columns, sort=False, dropna=False).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            src = self.series[level].get(key)
            ix = np.asarray(ids)
            for mon in np.unique(months[ix]):
                sub = ix[months[ix] == mon]
                if src is not None:
                    ys, ms = src.before(int(mon))
                else:
                    ys = ms = np.empty(0, dtype=float)
                mode = 'basic' if level == 'block' else level
                basic[sub, :] = _stat(ys, ms, int(mon), mode)
        names = (F1_COLS if level == 'block' else
                 F2_COLS[:3] if level == 'street' else F2_COLS[3:])
        for j, col in enumerate(names):
            out[col] = basic[:, j]
        if tier >= 3:
            # Columns are emitted in level order only below; order fixed at end.
            for window in (3, 6, 12):
                count = np.zeros(len(q), dtype=float)
                median = np.full(len(q), np.nan)
                for key, ids in q.groupby(columns, sort=False, dropna=False).indices.items():
                    key = key if isinstance(key, tuple) else (key,)
                    src = self.series[level].get(key)
                    if src is None:
                        continue
                    ix = np.asarray(ids)
                    for mon in np.unique(months[ix]):
                        vals, _ = src.before(int(mon), window)
                        sub = ix[months[ix] == mon]
                        count[sub] = len(vals)
                        if len(vals):
                            median[sub] = float(np.median(vals))
                out[f'hr_{level}_{window}m_n'] = count
                out[f'hr_{level}_{window}m_median'] = median

    def _comps_features(self, q, months, out):
        vals = np.full((len(q), len(F4_COLS)), np.nan)
        vals[:, 0] = 0
        vals[:, 4] = 0
        columns = ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE']
        for key, ids in q.groupby(columns, sort=False, dropna=False).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            ix = np.asarray(ids)
            address, flat = key[:3], key[3]
            lat, lon = q.loc[ix[0], ['LATITUDE', 'LONGITUDE']]
            candidates = self._neighbors(address, float(lat), float(lon)) if pd.notna(lat) and pd.notna(lon) else []
            if not candidates:
                continue
            for mon in np.unique(months[ix]):
                history, dists, recent = [], [], []
                for neighbor, dist in candidates:
                    hs = self.series['block'].get((*neighbor, flat))
                    if hs is None:
                        continue
                    rents, rent_months = hs.before(int(mon))
                    if not len(rents):
                        continue
                    # Recent samples per nearby building avoid one block dominating.
                    select = rents[-5:]
                    history.extend(select.tolist())
                    dists.extend([max(dist, 0.05)] * len(select))
                    recent.extend(rents[rent_months >= int(mon) - 12].tolist())
                dst = ix[months[ix] == mon]
                if history:
                    a = np.array(history)
                    weights = 1.0 / np.array(dists)
                    vals[dst, 0] = len(a)
                    vals[dst, 1] = float(np.median(a))
                    vals[dst, 2] = float(np.average(a, weights=weights))
                    vals[dst, 3] = float(min(dists))
                if recent:
                    vals[dst, 4] = len(recent)
                    vals[dst, 5] = float(np.median(recent))
        for j, col in enumerate(F4_COLS):
            out[col] = vals[:, j]


def feature_columns(tier: int) -> list[str]:
    return ALL_COLS[:len(F1_COLS) + (len(F2_COLS) if tier >= 2 else 0) +
                    (len(F3_COLS) if tier >= 3 else 0) +
                    (len(F4_COLS) if tier >= 4 else 0)] if tier else []

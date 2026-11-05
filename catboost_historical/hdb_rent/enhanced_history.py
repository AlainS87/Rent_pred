"""Leakage-controlled, cutoff-anchored HDB rent aggregates for G1--G4.

In a fold, the reference contains TRAINING LABELS ONLY. A training row
from month M uses observations strictly earlier than M. Every validation
row uses a fixed anchor equal to the first validation month, so rolling
windows never move into the unavailable validation period.

Index normalization uses only month/flat-type median rents for MONTHS < anchor.
This is a historical feature, not a forecasted future index.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from .historical_features import HistoricRents, ordinals

G1 = ['g_block_normalized_median', 'g_block_normalized_mean', 'g_block_adj_minus_raw']
G2 = [x for win in (3, 6, 12) for x in (f'g_block_{win}m_n', f'g_block_{win}m_normalized_median')]
G3 = [
    'g_street_n', 'g_street_normalized_median',
    'g_town_n', 'g_town_normalized_median',
    'g_street_smoothed_median', 'g_block_smoothed_median',
    'g_street_12m_n', 'g_street_12m_normalized_median',
    'g_town_12m_n', 'g_town_12m_normalized_median',
]
G4 = ['g_nearby_n', 'g_nearby_normalized_median', 'g_nearby_normalized_weighted_mean',
      'g_nearby_min_km', 'g_nearby_12m_n', 'g_nearby_12m_normalized_median']
ADDITIONS = G1 + G2 + G3 + G4


def variant_columns(variant: str) -> list[str]:
    if variant == 'F1':
        return []
    tier = int(variant[1:])
    if variant[0] != 'G' or tier not in (1, 2, 3, 4):
        raise ValueError('Unknown variant: ' + variant)
    return ADDITIONS[:len(G1) + (len(G2) if tier >= 2 else 0)
                     + (len(G3) if tier >= 3 else 0)
                     + (len(G4) if tier >= 4 else 0)]


def _median(a):
    return float(np.median(a)) if len(a) else np.nan


class AnchoredRents:
    """Caches reference arrays and per-(address, cutoff) aggregates."""
    def __init__(self, reference: pd.DataFrame, y: np.ndarray, max_neighbors=24, radius_km=1.2):
        self.h = HistoricRents(reference, y, max_neighbors=max_neighbors,
                              max_radius_km=radius_km)
        df = self.h.reference
        d = df[['FLAT_TYPE', 'RENT_APPROVAL_DATE']].copy()
        d['rent'] = y
        d['month_ord'] = ordinals(df)
        tmp = d.groupby(['FLAT_TYPE', 'month_ord'], sort=False)['rent'].median()
        self.month_median = {(str(k[0]), int(k[1])): float(v) for k, v in tmp.items()}
        by_type = defaultdict(list)
        for (flat, mon), value in self.month_median.items():
            by_type[flat].append((mon, value))
        self.type_months = {}
        for flat, rows in by_type.items():
            rows.sort()
            self.type_months[flat] = (np.array([x[0] for x in rows], dtype=int),
                                      np.array([x[1] for x in rows], dtype=float))
        self.anchor_cache = {}
        self.group_cache = {}
        self.comp_cache = {}

    def _anchor_index(self, flat, anchor):
        cache_key = (flat, anchor)
        if cache_key in self.anchor_cache:
            return self.anchor_cache[cache_key]
        dates, vals = self.type_months.get(flat, (np.empty(0, int), np.empty(0)))
        subset = vals[(dates >= anchor - 3) & (dates < anchor)]
        if not len(subset):
            subset = vals[dates < anchor][-3:]
        value = float(np.median(subset)) if len(subset) else np.nan
        self.anchor_cache[cache_key] = value
        return value

    def normalize(self, rents, rent_months, flat, anchor):
        if len(rents) == 0:
            return np.array([], dtype=float)
        base = self._anchor_index(flat, anchor)
        if not np.isfinite(base):
            return np.asarray(rents, dtype=float)
        historical = np.array([self.month_median.get((flat, int(m)), np.nan)
                               for m in rent_months], dtype=float)
        ratios = np.divide(base, historical, out=np.ones(len(rents), dtype=float),
                           where=np.isfinite(historical) & (historical > 0))
        # Avoid a small historical-month sample causing a runaway feature.
        ratios = np.clip(ratios, 0.55, 1.8)
        return np.asarray(rents, float) * ratios

    def group_values(self, level, key, flat, anchor, window=None):
        ckey = (level, key, flat, anchor, window)
        if ckey in self.group_cache:
            return self.group_cache[ckey]
        hist = self.h.series[level].get(key)
        if hist is None:
            result = (0., np.nan, np.nan)
        else:
            rents, mons = hist.before(anchor, window)
            adj = self.normalize(rents, mons, flat, anchor)
            result = (float(len(rents)), _median(adj),
                      float(np.mean(adj)) if len(adj) else np.nan)
        self.group_cache[ckey] = result
        return result

    def comps(self, town, block, street, flat, anchor, lat, lon):
        ckey = (town, block, street, flat, anchor)
        if ckey in self.comp_cache:
            return self.comp_cache[ckey]
        if not np.isfinite(lat) or not np.isfinite(lon):
            result = (0., np.nan, np.nan, np.nan, 0., np.nan)
            self.comp_cache[ckey] = result
            return result
        neighbors = self.h._neighbors((town, block, street), float(lat), float(lon))
        rents, weights, distances, recent = [], [], [], []
        for addr, km in neighbors:
            hs = self.h.series['block'].get((*addr, flat))
            if hs is None:
                continue
            ys, ms = hs.before(anchor)
            if not len(ys):
                continue
            prices = self.normalize(ys, ms, flat, anchor)
            sel = prices[-5:]    # up to five latest comparable rents per building
            rents.extend(sel.tolist())
            weights.extend([1 / max(km, 0.05)] * len(sel))
            distances.append(float(km))
            recent.extend(prices[ms >= anchor - 12].tolist())
        if not len(rents):
            result = (0., np.nan, np.nan, np.nan, 0., np.nan)
        else:
            result = (float(len(rents)), _median(rents),
                      float(np.average(rents, weights=weights)),
                      float(min(distances)), float(len(recent)), _median(recent))
        self.comp_cache[ckey] = result
        return result

    def materialize(self, queries: pd.DataFrame, cutoff: str, tier: int) -> pd.DataFrame:
        """Return G features, all anchored at min(query_month, cutoff).

        `cutoff` is the FIRST month whose rents are unknown (e.g. 2024-07).
        Training rows before this month get month-specific causal history;
        validation/test months >= cutoff get one frozen historic reference.
        """
        if tier not in (1, 2, 3, 4):
            raise ValueError('tier must be 1..4')
        q = queries.reset_index(drop=True)
        if not len(q):
            raise ValueError('No query rows')
        cutoff_ord = int(pd.Period(cutoff, freq='M').ordinal)
        raw_months = ordinals(q)
        anchors = np.minimum(raw_months, cutoff_ord)
        if len(self.h.months) and np.any(self.h.months >= cutoff_ord):
            raise ValueError('Reference contains months at or after the fold cutoff')
        cols = variant_columns('G' + str(tier))
        data = {col: np.full(len(q), np.nan, dtype=float) for col in cols}
        for col in cols:
            if col.endswith('_n'):
                data[col][:] = 0.0
        keys = ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE']
        # Do not collapse by date in Python if a group spans multiple months.
        for key, ids in q.groupby(keys, sort=False, dropna=False).indices.items():
            town, block, street, flat = key
            ix = np.asarray(ids, dtype=int)
            for anchor in np.unique(anchors[ix]):
                positions = ix[anchors[ix] == anchor]
                mon = int(anchor)
                bkey = (town, block, street, flat)
                bn, bm, bmean = self.group_values('block', bkey, flat, mon)
                if 'g_block_normalized_median' in data:
                    data['g_block_normalized_median'][positions] = bm
                    data['g_block_normalized_mean'][positions] = bmean
                    # Reference raw median is computed without any month adjustment.
                    raw_hist = self.h.series['block'].get(bkey)
                    if raw_hist is not None:
                        raw_vals, _ = raw_hist.before(mon)
                        if len(raw_vals):
                            data['g_block_adj_minus_raw'][positions] = bm - float(np.median(raw_vals))
                if tier >= 2:
                    for window in (3, 6, 12):
                        n, med, _ = self.group_values('block', bkey, flat, mon, window)
                        data[f'g_block_{window}m_n'][positions] = n
                        data[f'g_block_{window}m_normalized_median'][positions] = med
                if tier >= 3:
                    skey, tkey = (town, street, flat), (town, flat)
                    sn, sm, _ = self.group_values('street', skey, flat, mon)
                    tn, tm, _ = self.group_values('town', tkey, flat, mon)
                    data['g_street_n'][positions] = sn
                    data['g_street_normalized_median'][positions] = sm
                    data['g_town_n'][positions] = tn
                    data['g_town_normalized_median'][positions] = tm
                    # Shrink sparse street rents toward the town median, then
                    # sparse block rents toward the street/town estimate.
                    street_sm = (sn * sm + 30 * tm) / (sn + 30) if np.isfinite(tm) and np.isfinite(sm) else (sm if np.isfinite(sm) else tm)
                    block_sm = (bn * bm + 8 * street_sm) / (bn + 8) if np.isfinite(bm) and np.isfinite(street_sm) else (bm if np.isfinite(bm) else street_sm)
                    data['g_street_smoothed_median'][positions] = street_sm
                    data['g_block_smoothed_median'][positions] = block_sm
                    for level, k in [('street', skey), ('town', tkey)]:
                        n, med, _ = self.group_values(level, k, flat, mon, 12)
                        data[f'g_{level}_12m_n'][positions] = n
                        data[f'g_{level}_12m_normalized_median'][positions] = med
                if tier >= 4:
                    lat, lon = q.loc[positions[0], ['LATITUDE', 'LONGITUDE']]
                    arr = self.comps(town, block, street, flat, mon,
                                     float(lat) if pd.notna(lat) else np.nan,
                                     float(lon) if pd.notna(lon) else np.nan)
                    for col, value in zip(G4, arr):
                        data[col][positions] = value
        out = pd.DataFrame(data, index=np.arange(len(q)))
        if np.isinf(out.to_numpy(dtype=float)).any():
            raise RuntimeError('Invalid infinite enhanced history feature')
        return out

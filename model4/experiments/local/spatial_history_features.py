"""Past-only rental descriptors; no fitted prediction model and no label changes."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
FEATURES = ['past_monthly_rent_mean', 'past_record_count', 'past_monthly_rent_std']
EARTH_RADIUS_KM = 6371.0088
RADIUS_KM = 0.5
QUERY_COLUMNS = ['latitude', 'longitude', 'flat_type_ord']

def neighbors(history, query):
    """Same flat type, <=500m; count every original row, sample std (ddof=1)."""
    assert 'MONTHLY_RENT' not in query.columns, 'Query labels must be removed'
    out = np.full((len(query), 3), np.nan, dtype=float)
    out[:, 1] = 0
    hcoord = history[['latitude', 'longitude']].to_numpy(float)
    qcoord = query[['latitude', 'longitude']].to_numpy(float)
    htype = history.flat_type_ord.to_numpy(float)
    qtype = query.flat_type_ord.to_numpy(float)
    rent = history.MONTHLY_RENT.to_numpy(float)
    assert np.isfinite(rent).all()
    hgood = np.isfinite(hcoord).all(axis=1) & np.isfinite(htype)
    qgood = np.isfinite(qcoord).all(axis=1) & np.isfinite(qtype)
    for kind in np.unique(qtype[qgood]):
        hmask, qmask = (hgood & (htype == kind), qgood & (qtype == kind))
        if not hmask.any():
            continue
        unique, inverse = np.unique(qcoord[qmask], axis=0, return_inverse=True)
        tree = BallTree(np.radians(hcoord[hmask]), metric='haversine')
        matches = tree.query_radius(np.radians(unique), r=RADIUS_KM / EARTH_RADIUS_KM)
        rents = rent[hmask]
        values = np.full((len(unique), 3), np.nan)
        values[:, 1] = 0
        for index, selected in enumerate(matches):
            values[index, 1] = len(selected)
            if len(selected):
                values[index, 0] = np.mean(rents[selected])
            if len(selected) > 1:
                values[index, 2] = np.std(rents[selected], ddof=1)
        out[qmask] = values[inverse]
    assert not np.isinf(out).any()
    assert np.isfinite(out[:, 1]).all() and (out[:, 1] >= 0).all()
    assert np.isnan(out[out[:, 1] == 0, 0]).all()
    assert np.isnan(out[out[:, 1] < 2, 2]).all()
    return pd.DataFrame(out, columns=FEATURES, index=query.index)

def training_history(fit, checkpoint):
    months = pd.PeriodIndex(fit.RENT_APPROVAL_DATE.astype(str).str[:7], freq='M')
    result = pd.DataFrame(np.nan, index=fit.index, columns=FEATURES)
    audits = []
    for month in sorted(months.unique()):
        checkpoint()
        hmask = (months >= month - 6) & (months < month)
        qmask = months == month
        history = fit.loc[hmask]
        query = fit.loc[qmask, QUERY_COLUMNS]
        values = neighbors(history, query)
        result.loc[qmask, FEATURES] = values.to_numpy()
        audit = {'query_month': str(month), 'query_rows': int(qmask.sum()), 'history_rows': int(hmask.sum()), 'window_start': str(month - 6), 'window_end': str(month - 1), 'history_excludes_query_month': True, 'history_first_month': str(months[hmask].min()) if hmask.any() else None, 'history_last_month': str(months[hmask].max()) if hmask.any() else None, 'zero_history_queries': int((values[FEATURES[1]] == 0).sum())}
        if hmask.any():
            assert months[hmask].max() < month and months[hmask].min() >= month - 6
        audits.append(audit)
    return (result, audits)

def frozen_history(fit, query):
    assert 'MONTHLY_RENT' not in query.columns
    months = fit.RENT_APPROVAL_DATE.astype(str).str[:7]
    history = fit.loc[(months >= '2024-04') & (months <= '2024-09')]
    assert len(history) == 17960 and months.max() == '2024-09'
    return neighbors(history, query[QUERY_COLUMNS])

def self_check():
    """Small temporal/geometry fixture; no production data or fitting."""
    frame = pd.DataFrame({'RENT_APPROVAL_DATE': ['2023-07', '2024-01', '2024-02', '2024-02', '2024-02', '2024-03'], 'latitude': [1.3, 1.3, 1.3, 1.3, 1.4, 1.3], 'longitude': [103.8] * 6, 'flat_type_ord': [3, 3, 3, 4, 3, 3], 'MONTHLY_RENT': [90000, 1000, 3000, 80000, 70000, 60000]})
    a, audits = training_history(frame, lambda: None)
    np.testing.assert_allclose(a.iloc[-1].to_numpy(), [2000, 2, np.sqrt(2000000)])
    changed = frame.copy()
    changed.loc[changed.RENT_APPROVAL_DATE == '2024-03', 'MONTHLY_RENT'] = -999
    b, _ = training_history(changed, lambda: None)
    np.testing.assert_array_equal(a.iloc[-1].to_numpy(), b.iloc[-1].to_numpy())
    assert a.iloc[0, 1] == 0 and np.isnan(a.iloc[0, 0])
    assert a.iloc[2, 1] == 1 and np.isnan(a.iloc[2, 2])
    dup = pd.concat([frame.iloc[[1]], frame.iloc[[1]]], ignore_index=True)
    got = neighbors(dup, frame.iloc[[-1]][QUERY_COLUMNS])
    np.testing.assert_allclose(got.iloc[0].to_numpy(), [1000, 2, 0])
    return {'past_only': True, 'same_type_only': True, 'radius_filter': True, 'duplicates_retained': True, 'no_history': True, 'std_ddof': 1, 'production_data_read': False, 'training_performed': False}
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    import json
    print(json.dumps(self_check()))

"""History-only property fixed effects and month effects for a rental price index."""
from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
KEYS = ['TOWN', 'STREET', 'BLOCK', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM']
SLOPE_MONTHS = 6
DAMPING = 0.9

def prepare_rows(raw):
    from experiments.local._support.src.features.keys import norm_text
    from experiments.local._support.src.features.source_fields import flat_type_key
    rows = raw.copy()
    rows['month'] = rows.RENT_APPROVAL_DATE.astype(str).str[:7]
    rows['FLAT_TYPE'] = flat_type_key(rows.FLAT_TYPE)
    for name in ['TOWN', 'STREET', 'BLOCK', 'FLAT_MODEL']:
        rows[name] = norm_text(rows[name])
    assert np.isfinite(rows.MONTHLY_RENT).all() and (rows.MONTHLY_RENT > 0).all()
    rows['log_rent'] = np.log(rows.MONTHLY_RENT.to_numpy(float))
    return rows

def fit_index(history):
    """OLS log rent = property effect + month effect, using within-property differences.

    Only properties with observations in at least two history months identify the
    index. The first history month is the zero reference. No future data is used.
    """
    distinct = history.groupby(KEYS, observed=True).month.nunique()
    eligible = distinct.index[distinct >= 2]
    row_keys = pd.MultiIndex.from_frame(history[KEYS])
    work = history.loc[row_keys.isin(eligible)].copy()
    keys = pd.MultiIndex.from_frame(work[KEYS])
    group_codes, properties = pd.factorize(keys, sort=True)
    months = sorted(work.month.unique())
    assert months == sorted(history.month.unique())
    month_codes = pd.Categorical(work.month, categories=months).codes
    count = coo_matrix((np.ones(len(work)), (group_codes, month_codes)), shape=(len(properties), len(months))).tocsr()
    group_n = np.asarray(count.sum(axis=1)).ravel()
    month_n = np.asarray(count.sum(axis=0)).ravel()
    log_rent = work.log_rent.to_numpy(float)
    group_sum = np.bincount(group_codes, weights=log_rent, minlength=len(properties))
    month_sum = np.bincount(month_codes, weights=log_rent, minlength=len(months))
    gram = np.diag(month_n) - (count.T @ count.multiply((1 / group_n)[:, None])).toarray()
    rhs = month_sum - np.asarray(count.T @ (group_sum / group_n)).ravel()
    rank = int(np.linalg.matrix_rank(gram[1:, 1:]))
    assert rank == len(months) - 1, 'Disconnected property/month index'
    month_effect = np.r_[0.0, np.linalg.solve(gram[1:, 1:], rhs[1:])]
    property_effect = (group_sum - np.asarray(count @ month_effect).ravel()) / group_n
    residual = log_rent - property_effect[group_codes] - month_effect[month_codes]
    max_month_sum = float(np.max(np.abs(np.bincount(month_codes, weights=residual))))
    max_group_sum = float(np.max(np.abs(np.bincount(group_codes, weights=residual))))
    assert max_month_sum < 1e-06 and max_group_sum < 1e-06
    assert np.isfinite(month_effect).all() and np.isfinite(property_effect).all()
    slope_x = np.arange(SLOPE_MONTHS, dtype=float)
    slope_x -= slope_x.mean()
    slope = float(np.dot(slope_x, month_effect[-SLOPE_MONTHS:]) / np.dot(slope_x, slope_x))
    return {'months': months, 'month_effect': month_effect, 'property_effect': pd.Series(property_effect, index=properties), 'slope': slope, 'damping': DAMPING, 'slope_months': SLOPE_MONTHS, 'fit_rows': len(history), 'eligible_fit_rows': len(work), 'property_groups': len(properties), 'fit_end': months[-1], 'normal_equation_max_month_residual_sum': max_month_sum, 'normal_equation_max_property_residual_sum': max_group_sum, 'time_gram_condition_number': float(np.linalg.cond(gram[1:, 1:]))}

def forecast(index, horizons, mode):
    horizons = np.asarray(horizons, dtype=float)
    assert (horizons > 0).all()
    level = float(index['month_effect'][-1])
    if mode == 'flat':
        return np.full(horizons.shape, level)
    assert mode == 'damped'
    phi = index['damping']
    return level + index['slope'] * phi * (1 - phi ** horizons) / (1 - phi)

def scales(index, dates):
    """Reference-period monthly-rent scaling; no outcome values are inputs."""
    months = pd.PeriodIndex(pd.Series(dates).astype(str).str[:7], freq='M')
    origin = pd.Period(index['fit_end'], freq='M')
    horizon = months.asi8 - origin.ordinal
    lookup = dict(zip(index['months'], index['month_effect']))
    effects = np.empty(len(months))
    for pos, (month, h) in enumerate(zip(months.astype(str), horizon)):
        effects[pos] = lookup[month] if h <= 0 else forecast(index, [h], 'damped')[0]
    result = np.exp(effects - index['month_effect'][-1])
    assert np.isfinite(result).all() and (result > 0).all()
    return result

def backtest(history, future, origin, max_horizon):
    assert history.month.max() == origin
    assert future.month.min() > origin
    index = fit_index(history)
    keys = pd.MultiIndex.from_frame(future[KEYS])
    property_effect = index['property_effect'].reindex(keys).to_numpy(float)
    work = future[['source_row_id', 'month', 'log_rent']].copy()
    work['property_effect'] = property_effect
    work['eligible'] = np.isfinite(property_effect)
    work['observed_time_component'] = work.log_rent - property_effect
    monthly = work.groupby('month').agg(total_rows=('eligible', 'size'), eligible_rows=('eligible', 'sum'), observed_time_component=('observed_time_component', 'mean')).reset_index()
    horizons = pd.PeriodIndex(monthly.month, freq='M').asi8 - pd.Period(origin, freq='M').ordinal
    assert np.array_equal(horizons, np.arange(1, max_horizon + 1))
    monthly['horizon'] = horizons
    monthly['origin'] = origin
    monthly['coverage'] = monthly.eligible_rows / monthly.total_rows
    reference = float(index['month_effect'][-1])
    monthly['observed_index'] = 100 * np.exp(monthly.observed_time_component - reference)
    for mode in ['flat', 'damped']:
        monthly[mode + '_index'] = 100 * np.exp(forecast(index, horizons, mode) - reference)
    assert np.isfinite(monthly.select_dtypes(include=[np.number]).to_numpy()).all()
    return (index, monthly, work)

def summarize(table):
    result = {}
    masks = {'h1_6': table.horizon <= 6, 'h7_16': table.horizon.between(7, 16), 'h17_22': table.horizon.between(17, 22), 'h7_plus': table.horizon >= 7, 'all': np.ones(len(table), dtype=bool)}
    for name, mask in masks.items():
        part = table.loc[mask]
        if not len(part):
            continue
        values = {mode: float(np.sqrt(np.mean((part[mode + '_index'] - part.observed_index) ** 2))) for mode in ['flat', 'damped']}
        result[name] = {'months': len(part), 'flat_rmse_index_points': values['flat'], 'damped_rmse_index_points': values['damped'], 'damped_minus_flat': values['damped'] - values['flat']}
    return result

def gate(table):
    by_origin = {origin: summarize(part) for origin, part in table.groupby('origin')}
    pooled = summarize(table)
    conditions = {'each_month_coverage_at_least_70pct': bool((table.coverage >= 0.7).all()), 'each_origin_long_horizon_not_worse': all((x['h7_plus']['damped_rmse_index_points'] <= x['h7_plus']['flat_rmse_index_points'] for x in by_origin.values())), 'pooled_long_horizon_at_least_5pct_better': pooled['h7_plus']['damped_rmse_index_points'] <= 0.95 * pooled['h7_plus']['flat_rmse_index_points'], 'each_origin_short_horizon_no_more_than_5pct_worse': all((x['h1_6']['damped_rmse_index_points'] <= 1.05 * x['h1_6']['flat_rmse_index_points'] for x in by_origin.values()))}
    return {'passed': all(conditions.values()), 'conditions': conditions, 'by_origin': by_origin, 'pooled': pooled, 'unit': 'Index points, origin index=100; NOT monthly-rent RMSE', 'interpretation': 'Retrospective supplemental diagnostic; overlapping horizons and previously analyzed history are not independent untouched tests'}

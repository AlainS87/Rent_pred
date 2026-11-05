"""Within-property monthly rent index, fitted exclusively on permitted history."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

KEYS = ['TOWN', 'STREET', 'BLOCK', 'FLAT_TYPE', 'FLAT_MODEL', 'FLOOR_AREA_SQM']

SLOPE_MONTHS = 6

DAMPING = 0.9

def prepare_rows(raw):
    from experiments.rolling_local._core.keys import norm_text
    from experiments.rolling_local._core.source_fields import flat_type_key
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

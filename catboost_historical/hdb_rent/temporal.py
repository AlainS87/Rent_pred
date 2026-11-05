"""Original seasonality and forward market correction; arithmetic unchanged."""
import numpy as np
import pandas as pd

def cyclic(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    ang = 2 * np.pi * (out['month'].to_numpy() - 1) / 12
    out['month_sin'] = np.sin(ang)
    out['month_cos'] = np.cos(ang)
    return out

def adjusted_market_index(train: pd.DataFrame, y: np.ndarray, rounds: int = 20) -> pd.DataFrame:
    """Monthly log-rent effects adjusted for town-by-flat-type composition.

    Fit additive two-way group/month fixed effects by alternating group means;
    relies EXCLUSIVELY on the supplied (past) training sample. The index is not
    used to look up individual rents, and is not fit on validation/test labels.
    """
    if len(train) != len(y):
        raise ValueError('Group data and targets must align')
    if len(train) == 0 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError('Training rents must be finite, positive and nonempty')
    months = train['RENT_APPROVAL_DATE'].astype(str).to_numpy()
    groups = (train['TOWN'].fillna('').astype(str).str.strip().str.lower() + '|' +
              train['FLAT_TYPE'].fillna('').astype(str).str.strip().str.lower()
              .str.replace(' ', '-', regex=False)).to_numpy()
    month_names, mi = np.unique(months, return_inverse=True)
    _, gi = np.unique(groups, return_inverse=True)
    log_y = np.log(np.asarray(y, dtype=float))
    nmonths = len(month_names)
    ngroups = gi.max() + 1
    month_counts = np.bincount(mi, minlength=nmonths)
    group_counts = np.bincount(gi, minlength=ngroups)
    month_effect = np.zeros(nmonths)
    for _ in range(rounds):
        group_effect = np.bincount(gi, weights=log_y - month_effect[mi], minlength=ngroups) / group_counts
        new_month = np.bincount(mi, weights=log_y - group_effect[gi], minlength=nmonths) / month_counts
        # Keep month effect centered. Its absolute offset is immaterial.
        new_month -= np.average(new_month, weights=month_counts)
        month_effect = new_month
    return pd.DataFrame({'month': month_names, 'effect': month_effect, 'count': month_counts})

def forecast_index(index: pd.DataFrame, future_months, lookback: int = 12,
                   decay_months: float = 6.0, cap_monthly: float = 0.012) -> np.ndarray:
    """Damped local-trend extrapolation (log units), strictly using known months.

    Output is a relative log-rent change since the average of the last three
    observed months. 3-month smoothing reduces noise at the cutoff.
    """
    z = index.sort_values('month').reset_index(drop=True)
    if len(z) < 8:
        raise ValueError('At least 8 historic months are necessary for trend fitting')
    m = pd.PeriodIndex(z['month'], freq='M')
    if np.any(np.diff(m.asi8) != 1):
        raise ValueError('Historic index months must be consecutive')
    effect = z['effect'].to_numpy(float)
    smoothed = pd.Series(effect).rolling(5, center=True, min_periods=1).mean().to_numpy()
    last_level = float(np.mean(effect[-3:]))
    # Estimate a local slope on smoothed history, not the 2022-23 boom average.
    recent = smoothed[-min(lookback, len(smoothed)):]
    t = np.arange(len(recent), dtype=float)
    slope = float(np.polyfit(t, recent, 1)[0])
    slope = float(np.clip(slope, -cap_monthly, cap_monthly))
    last = int(m.asi8[-1])
    future = pd.PeriodIndex(pd.Series(future_months).astype(str), freq='M')
    offset = future.asi8 - last
    if np.any(offset <= 0):
        raise ValueError('Forecast period must be strictly after last fitted month')
    # Smooth bounded projection. Any short-term slope gradually decays.
    delta = slope * decay_months * (1.0 - np.exp(-offset / decay_months))
    return np.asarray(smoothed[-1] - last_level + delta, dtype=float)

def market_multiplier(reference_train, y, prediction_months, args):
    idx = adjusted_market_index(reference_train, y)
    last_month = str(idx['month'].max())
    months = np.asarray(prediction_months, dtype=str)
    scale = np.ones(len(months), dtype=float)
    future = months > last_month
    if future.any() and args.trend_strength:
        delta = forecast_index(idx, months[future],
                               lookback=args.lookback,
                               decay_months=args.decay_months,
                               cap_monthly=args.cap_monthly)
        scale[future] = np.exp(args.trend_strength * delta)
    if not np.isfinite(scale).all():
        raise ValueError('Invalid market trend multipliers')
    return scale

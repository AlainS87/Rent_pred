"""Construct macro candidates using only complete months before each rent month.

No target is read and no future value is backfilled.  This module supplies lagged
features for the provided-auxiliary scenario; a fixed-cutoff caller must freeze
the row for cutoff + 1 month for every later prediction month.  Full-source
coverage statistics are descriptive audits, never a substitute for screening
candidate columns inside a training fold.
"""
from __future__ import annotations
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
COE_NUMERIC = ('QUOTA_PREMIUM', 'PREVAILING_QUOTA_PREMIUM', 'QUOTA', 'BIDS_RECEIVED')
STOCK_NUMERIC = ('OPEN', 'HIGH', 'LOW', 'CLOSE', 'ADJUSTED_CLOSE', 'VOLUME')
WINDOWS = (1, 3, 6)
MIN_DAYS = 5
MONTH_NAMES = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
PRESET_STOCK_GROUPS = {'banks': ('D05.SI', 'O39.SI', 'U11.SI'), 'property': ('C09.SI', 'U14.SI', '9CI.SI')}

def _numeric(series: pd.Series) -> pd.Series:
    text = series.astype('string').str.replace('[$,]', '', regex=True).str.strip()
    return pd.to_numeric(text, errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)

def _slug(value: str) -> str:
    return re.sub('[^a-z0-9]+', '_', value.lower()).strip('_') or 'unknown'

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def _ratio_change(series: pd.Series, horizon: int) -> pd.Series:
    """Require every intermediate month, rather than bridging data gaps."""
    prior = series.shift(horizon)
    complete = series.notna().rolling(horizon + 1, min_periods=horizon + 1).sum()
    valid = complete.eq(horizon + 1) & prior.gt(0)
    return series.div(prior).sub(1).where(valid)

def _attribute_rows(raw: pd.DataFrame, source: str, roles: dict[str, str]) -> list[dict]:
    rows = []
    for column in raw.columns:
        empty = raw[column].isna() | raw[column].astype('string').str.strip().eq('')
        rows.append({'source_file': source, 'source_attribute': column, 'raw_rows': len(raw), 'missing_raw': int(empty.fillna(True).sum()), 'unique_nonmissing': int(raw.loc[~empty.fillna(True), column].nunique()), 'role': roles.get(column, 'Unexpected column: audit before use'), 'availability': 'Derived values use only complete months before rent month'})
    return rows

def build_macro(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]], list[dict], dict]:
    """Return lagged monthly features, main families, lineage, and audit.

    The index is a YYYY-MM string named ``month``.  ``source_month_max`` is a
    provenance column, excluded from families.  Adjusted-close candidates stay
    in the returned frame and a separate ``stocks_adjusted_sensitivity`` family;
    their names are also listed in ``audit['sensitivity_columns']``.
    """
    data_dir, output_dir = (Path(data_dir), Path(output_dir))
    auxiliary = data_dir / 'auxiliary'
    coe_source, stock_source = ('sg-coe-prices.csv', 'sg-stock-prices.csv')
    coe_raw = pd.read_csv(auxiliary / coe_source, dtype='string')
    stock_raw = pd.read_csv(auxiliary / stock_source, dtype='string')
    required_coe = {'YEAR', 'CATEGORY', 'MONTH', 'ROUND', *COE_NUMERIC}
    required_stock = {'NAME', 'SYMBOL', 'DATE', *STOCK_NUMERIC}
    for source, raw, required in ((coe_source, coe_raw, required_coe), (stock_source, stock_raw, required_stock)):
        absent = sorted(required - set(raw.columns))
        if absent:
            raise ValueError(f'{source} missing columns: {absent}')
    coe = coe_raw.copy()
    year = _numeric(coe['YEAR'])
    year = year.where(year.eq(np.floor(year))).astype('Int64')
    month_number = coe['MONTH'].str.strip().str.lower().map(MONTH_NAMES).astype('Int64')
    coe['date'] = pd.to_datetime(year.astype('string') + '-' + month_number.astype('string') + '-01', errors='coerce')
    coe['category'] = coe['CATEGORY'].str.strip().str.lower()
    coe['round'] = _numeric(coe['ROUND'])
    coe['round'] = coe['round'].where(coe['round'].gt(0) & coe['round'].eq(np.floor(coe['round'])))
    coe_invalid_numeric = {}
    for column in COE_NUMERIC:
        numeric = _numeric(coe[column])
        invalid = numeric.lt(0)
        if column in ('QUOTA_PREMIUM', 'PREVAILING_QUOTA_PREMIUM'):
            invalid |= numeric.eq(0)
        coe_invalid_numeric[column] = int(invalid.fillna(False).sum())
        coe[column] = numeric.mask(invalid)
    e_mask = coe['category'].eq('e').fillna(False)
    e_pqp_original = _numeric(coe_raw['PREVAILING_QUOTA_PREMIUM'])
    e_pqp_zero = int((e_mask & e_pqp_original.eq(0)).sum())
    coe.loc[e_mask, 'PREVAILING_QUOTA_PREMIUM'] = np.nan
    valid_coe_key = coe['date'].notna() & coe['category'].notna() & coe['round'].notna()
    stock = stock_raw.copy()
    stock['date'] = pd.to_datetime(stock['DATE'], errors='coerce')
    stock['symbol'] = stock['SYMBOL'].str.strip()
    invalid_stock_numeric = {}
    for column in STOCK_NUMERIC:
        numeric = _numeric(stock[column])
        invalid = numeric.lt(0) if column == 'VOLUME' else numeric.le(0)
        invalid_stock_numeric[column] = int(invalid.fillna(False).sum())
        stock[column] = numeric.mask(invalid)
    inequality = pd.DataFrame({'high_below_low': stock['HIGH'].lt(stock['LOW']), 'high_below_open': stock['HIGH'].lt(stock['OPEN']), 'high_below_close': stock['HIGH'].lt(stock['CLOSE']), 'low_above_open': stock['LOW'].gt(stock['OPEN']), 'low_above_close': stock['LOW'].gt(stock['CLOSE'])}).fillna(False)
    stock['ohlc_inconsistent'] = inequality.any(axis=1)
    high_violation = pd.concat([stock['LOW'] - stock['HIGH'], stock['OPEN'] - stock['HIGH'], stock['CLOSE'] - stock['HIGH'], stock['LOW'] - stock['OPEN'], stock['LOW'] - stock['CLOSE']], axis=1).max(axis=1).clip(lower=0)
    severity_denominator = stock['CLOSE'].combine_first(stock['OPEN'])
    stock['ohlc_violation_fraction'] = high_violation.div(severity_denominator)
    stock['ohlc_severe_1pct'] = stock['ohlc_violation_fraction'].gt(0.01)
    stock['ohlc_severe_10pct'] = stock['ohlc_violation_fraction'].gt(0.1)
    stock.loc[stock['ohlc_inconsistent'], ['OPEN', 'HIGH', 'LOW', 'CLOSE']] = np.nan
    stock.loc[stock['ohlc_inconsistent'], 'ADJUSTED_CLOSE'] = np.nan
    valid_stock_key = stock['date'].notna() & stock['symbol'].notna() & stock['symbol'].ne('')
    periods: list[pd.Period] = []
    periods.extend(coe.loc[valid_coe_key, 'date'].dt.to_period('M').tolist())
    periods.extend(stock.loc[valid_stock_key, 'date'].dt.to_period('M').tolist())
    for filename in ('train.csv', 'query.csv'):
        path = data_dir / filename
        if path.exists():
            dates = pd.read_csv(path, usecols=['RENT_APPROVAL_DATE'], dtype='string')
            parsed = pd.to_datetime(dates['RENT_APPROVAL_DATE'], errors='coerce')
            periods.extend(parsed.dropna().dt.to_period('M').tolist())
    if not periods:
        raise ValueError('No valid auxiliary or rent months')
    months = pd.period_range(min(periods), max(periods), freq='M')
    features: dict[str, pd.Series] = {}
    families: dict[str, list[str]] = {'coe': [], 'stocks': [], 'stocks_adjusted_sensitivity': []}
    lineage: list[dict] = []
    sensitivity_columns: list[str] = []
    stock_symbol_columns: dict[str, list[str]] = {}
    stock_aggregate_columns: list[str] = []

    def add(name: str, values: pd.Series, family: str, source: str, attributes: list[str], construction: str, sensitivity: bool=False, symbol: str | None=None) -> None:
        if name in features:
            raise ValueError(f'Duplicate macro feature: {name}')
        if source == coe_source:
            attributes = [*attributes, 'ROUND']
        if source == stock_source and set(attributes) & {'OPEN', 'HIGH', 'LOW', 'CLOSE', 'ADJUSTED_CLOSE'}:
            attributes = [*attributes, 'OPEN', 'HIGH', 'LOW', 'CLOSE']
            construction += '; OHLC-inconsistent source rows treated as missing'
        features[name] = values.reindex(months).astype(float).replace([np.inf, -np.inf], np.nan)
        if sensitivity:
            sensitivity_columns.append(name)
            families['stocks_adjusted_sensitivity'].append(name)
        else:
            families[family].append(name)
        if family == 'stocks':
            if symbol is None and (not sensitivity):
                stock_aggregate_columns.append(name)
            elif symbol is not None and (not sensitivity):
                stock_symbol_columns.setdefault(symbol, []).append(name)
        lineage.append({'feature': name, 'family': 'stocks_adjusted_sensitivity' if sensitivity else family, 'source_file': source, 'source_attributes': '|'.join(dict.fromkeys(attributes)), 'construction': construction, 'availability': 'Complete source month <= rent month - 1; no backward fill', 'sensitivity_only': sensitivity})

    def trailing(prefix: str, series: pd.Series, family: str, source: str, attributes: list[str], description: str, symbol: str | None=None) -> None:
        for window in WINDOWS:
            values = series if window == 1 else series.rolling(window, min_periods=window).mean()
            add(f'{prefix}_trailing_{window}m', values, family, source, attributes, f'{description}; trailing {window}-month mean, requiring all {window} months; lagged one month', symbol=symbol)
    coe_work = coe.loc[valid_coe_key].copy().drop_duplicates(subset=['date', 'category', 'round', *COE_NUMERIC])
    coe_conflict = coe_work.duplicated(['date', 'category', 'round'], keep=False)
    coe_conflicting_rows = int(coe_conflict.sum())
    coe_work.loc[coe_conflict, list(COE_NUMERIC)] = np.nan
    coe_work = coe_work.drop_duplicates(['date', 'category', 'round']).sort_values(['date', 'category', 'round'])
    coe_work['month'] = coe_work['date'].dt.to_period('M')
    coe_monthly_audit = []
    category_monthly: dict[str, pd.DataFrame] = {}
    categories = sorted(set(coe_work['category'].dropna().tolist()) | {'a', 'b', 'c', 'e'})
    for category in categories:
        records = []
        selected = coe_work.loc[coe_work['category'].eq(category)]
        for month, group in selected.groupby('month', sort=True):
            premium = group['QUOTA_PREMIUM']
            pqp = group['PREVAILING_QUOTA_PREMIUM']
            quota = group['QUOTA'].sum(min_count=1)
            bids = group['BIDS_RECEIVED'].sum(min_count=1)
            endpoint_valid = len(group) >= 2 and premium.iloc[[0, -1]].notna().all()
            gap = premium.iloc[-1] - premium.iloc[0] if endpoint_valid else np.nan
            gap_relative = gap / premium.iloc[0] if endpoint_valid else np.nan
            last_premium = premium.iloc[-1]
            pqp_mean = pqp.mean()
            record = {'month': month, 'premium_mean': premium.mean(), 'premium_last': last_premium, 'premium_round_gap': gap, 'premium_round_relative_gap': gap_relative, 'quota_sum': quota, 'bids_sum': bids, 'bid_pressure': bids / quota if pd.notna(quota) and quota > 0 else np.nan, 'pqp_mean': pqp_mean, 'premium_to_pqp': last_premium / pqp_mean if pd.notna(pqp_mean) and pqp_mean > 0 else np.nan}
            records.append(record)
            coe_monthly_audit.append({'month': str(month), 'category': category, 'rounds_present': int(group['round'].nunique()), 'valid_premium_rounds': int(premium.notna().sum()), 'valid_pqp_rounds': int(pqp.notna().sum()), 'valid_quota_rounds': int(group['QUOTA'].notna().sum()), 'valid_bids_rounds': int(group['BIDS_RECEIVED'].notna().sum()), 'e_pqp_structurally_excluded': category == 'e'})
        monthly = pd.DataFrame(records).set_index('month') if records else pd.DataFrame(index=months)
        monthly = monthly.reindex(months)
        category_monthly[category] = monthly
        field_map = {'premium_mean': ['QUOTA_PREMIUM'], 'premium_last': ['QUOTA_PREMIUM', 'ROUND'], 'premium_round_gap': ['QUOTA_PREMIUM', 'ROUND'], 'premium_round_relative_gap': ['QUOTA_PREMIUM', 'ROUND'], 'quota_sum': ['QUOTA'], 'bids_sum': ['BIDS_RECEIVED'], 'bid_pressure': ['QUOTA', 'BIDS_RECEIVED'], 'pqp_mean': ['PREVAILING_QUOTA_PREMIUM'], 'premium_to_pqp': ['QUOTA_PREMIUM', 'PREVAILING_QUOTA_PREMIUM']}
        for field, columns in field_map.items():
            if category == 'e' and field in ('pqp_mean', 'premium_to_pqp'):
                continue
            series = monthly[field] if field in monthly else pd.Series(np.nan, index=months)
            attributes = ['YEAR', 'MONTH', 'CATEGORY', *columns]
            trailing(f'coe_{_slug(category)}_{field}', series, 'coe', coe_source, attributes, f"Category {category.upper()} monthly {field.replace('_', ' ')}")
        for field, raw_columns in (('premium_last', ['QUOTA_PREMIUM', 'ROUND']), ('quota_sum', ['QUOTA']), ('bids_sum', ['BIDS_RECEIVED']), ('bid_pressure', ['QUOTA', 'BIDS_RECEIVED'])):
            series = monthly[field] if field in monthly else pd.Series(np.nan, index=months)
            for horizon in WINDOWS:
                add(f'coe_{_slug(category)}_{field}_change_{horizon}m', _ratio_change(series, horizon), 'coe', coe_source, ['YEAR', 'MONTH', 'CATEGORY', *raw_columns], f'Category {category.upper()} relative {horizon}-month change in {field}; complete intervening months; lagged one month')
        premium_series = monthly.get('premium_last', pd.Series(np.nan, index=months))
        monthly_return = np.log1p(_ratio_change(premium_series, 1))
        for window in (3, 6):
            add(f'coe_{_slug(category)}_premium_logreturn_vol_{window}m', monthly_return.rolling(window, min_periods=window).std(ddof=1), 'coe', coe_source, ['YEAR', 'MONTH', 'CATEGORY', 'ROUND', 'QUOTA_PREMIUM'], f'Std of monthly last-round premium log changes over {window} complete months; lagged one month')
    for upper, lower in (('b', 'a'), ('e', 'a')):
        upper_p = category_monthly[upper].get('premium_last', pd.Series(np.nan, index=months))
        lower_p = category_monthly[lower].get('premium_last', pd.Series(np.nan, index=months))
        trailing(f'coe_{upper}_over_{lower}_premium_ratio', upper_p.div(lower_p.where(lower_p.gt(0))), 'coe', coe_source, ['YEAR', 'MONTH', 'CATEGORY', 'ROUND', 'QUOTA_PREMIUM'], f'Ratio of monthly last-round premium, category {upper.upper()} / {lower.upper()}')
    stock_work = stock.loc[valid_stock_key].copy().drop_duplicates(subset=['symbol', 'date', *STOCK_NUMERIC])
    stock_conflict = stock_work.duplicated(['symbol', 'date'], keep=False)
    stock_conflicting_rows = int(stock_conflict.sum())
    stock_work.loc[stock_conflict, list(STOCK_NUMERIC)] = np.nan
    stock_work = stock_work.drop_duplicates(['symbol', 'date']).sort_values(['symbol', 'date'])
    stock_work['month'] = stock_work['date'].dt.to_period('M')
    monthly_by_symbol: dict[str, pd.DataFrame] = {}
    daily_return_by_symbol: dict[str, pd.Series] = {}
    symbol_audit = []
    symbols = sorted(stock_work['symbol'].dropna().unique().tolist())
    used_slugs: set[str] = set()
    symbol_slugs: dict[str, str] = {}
    for symbol in symbols:
        slug = _slug(symbol)
        if slug in used_slugs:
            slug += '_' + hashlib.sha1(symbol.encode('utf-8')).hexdigest()[:8]
        used_slugs.add(slug)
        symbol_slugs[symbol] = slug
        group = stock_work.loc[stock_work['symbol'].eq(symbol)].copy()
        close = group['CLOSE']
        date_gap = group['date'].diff().dt.days
        log_return = np.log(close.div(close.shift(1)))
        log_return = log_return.where(date_gap.between(1, 10))
        simple_return = np.expm1(log_return)
        group['daily_logreturn'] = log_return
        group['hl_range'] = (group['HIGH'] - group['LOW']).div(close)
        group['oc_return'] = close.div(group['OPEN']).sub(1)
        daily_return_by_symbol[symbol] = pd.Series(simple_return.to_numpy(), index=group['date'])
        month_records = []
        for month, quotes in group.groupby('month', sort=True):
            valid_close = quotes['CLOSE'].dropna()
            valid_adj = quotes['ADJUSTED_CLOSE'].dropna()
            valid_volume = quotes['VOLUME'].dropna()
            valid_daily = quotes['daily_logreturn'].dropna()
            hl = quotes['hl_range'].dropna()
            oc = quotes['oc_return'].dropna()
            month_records.append({'month': month, 'close': valid_close.iloc[-1] if len(valid_close) >= MIN_DAYS else np.nan, 'adjusted_close': valid_adj.iloc[-1] if len(valid_adj) >= MIN_DAYS else np.nan, 'daily_vol': valid_daily.std(ddof=1) * np.sqrt(252) if len(valid_daily) >= MIN_DAYS else np.nan, 'hl_range': hl.mean() if len(hl) >= MIN_DAYS else np.nan, 'oc_return': oc.mean() if len(oc) >= MIN_DAYS else np.nan, 'volume_mean': valid_volume.mean() if len(valid_volume) >= MIN_DAYS else np.nan, 'valid_close_days': len(valid_close)})
        monthly = pd.DataFrame(month_records).set_index('month').reindex(months)
        monthly_by_symbol[symbol] = monthly
        attributes = ['NAME', 'SYMBOL', 'DATE']
        prefix = f'stock_{slug}'
        for horizon in WINDOWS:
            add(f'{prefix}_return_{horizon}m', _ratio_change(monthly['close'], horizon), 'stocks', stock_source, [*attributes, 'CLOSE'], f'{symbol}: relative change between month-end valid CLOSE prices {horizon} months apart; at least {MIN_DAYS} valid closes per month and no intervening missing month; lagged one month', symbol=symbol)
            add(f'{prefix}_volume_change_{horizon}m', _ratio_change(monthly['volume_mean'], horizon), 'stocks', stock_source, [*attributes, 'VOLUME'], f'{symbol}: {horizon}-month relative change in own mean daily VOLUME; at least {MIN_DAYS} observations per month, no intervening missing month; lagged one month', symbol=symbol)
            add(f'{prefix}_adjusted_return_{horizon}m', _ratio_change(monthly['adjusted_close'], horizon), 'stocks', stock_source, [*attributes, 'ADJUSTED_CLOSE'], f'{symbol}: adjusted month-end price return over {horizon} months; sensitivity only because retrospective corporate-action adjustments are not timestamped; lagged one month', sensitivity=True, symbol=symbol)
        for field, fields, description in (('daily_vol', ['CLOSE'], 'Annualized within-month std of consecutive observed daily CLOSE log returns; skip gaps exceeding 10 calendar days'), ('hl_range', ['HIGH', 'LOW', 'CLOSE'], 'Monthly mean daily (HIGH - LOW) / CLOSE'), ('oc_return', ['OPEN', 'CLOSE'], 'Monthly mean daily CLOSE / OPEN - 1')):
            trailing(f'{prefix}_{field}', monthly[field], 'stocks', stock_source, [*attributes, *fields], f'{symbol}: {description}; minimum {MIN_DAYS} valid days', symbol=symbol)
        raw_symbol = stock.loc[stock['symbol'].eq(symbol)]
        missing_by_column = {f'missing_or_invalid_{col.lower()}': int(raw_symbol[col].isna().sum()) for col in STOCK_NUMERIC}
        symbol_audit.append({'symbol': symbol, 'feature_slug': slug, 'names': '|'.join(sorted(raw_symbol['NAME'].dropna().unique().tolist())), 'source_rows': len(raw_symbol), 'valid_unique_date_rows': len(group), 'first_date': group['date'].min().strftime('%Y-%m-%d'), 'last_date': group['date'].max().strftime('%Y-%m-%d'), 'months_with_min_close_days': int(monthly['close'].notna().sum()), 'ohlc_inconsistent_rows': int(raw_symbol['ohlc_inconsistent'].sum()), 'ohlc_violation_over_1pct_rows': int(raw_symbol['ohlc_severe_1pct'].sum()), 'ohlc_violation_over_10pct_rows': int(raw_symbol['ohlc_severe_10pct'].sum()), 'max_ohlc_violation_fraction': raw_symbol['ohlc_violation_fraction'].max(), 'raw_daily_return_over_threefold_rows': int(log_return.abs().gt(np.log(3)).sum()), 'main_candidate_count': len(stock_symbol_columns.get(symbol, [])), **missing_by_column})
    group_members = {'all': tuple(symbols), **PRESET_STOCK_GROUPS}
    group_coverage_rows = []
    for group_name, requested in group_members.items():
        present = [symbol for symbol in requested if symbol in monthly_by_symbol]
        if not present:
            continue
        min_members = 3 if group_name == 'all' else min(2, len(present))
        return_matrix = pd.concat({symbol: _ratio_change(monthly_by_symbol[symbol]['close'], 1) for symbol in present}, axis=1)
        observed = return_matrix.notna().sum(axis=1)
        portfolio_return = return_matrix.mean(axis=1).where(observed.ge(min_members))
        daily_matrix = pd.concat({symbol: daily_return_by_symbol[symbol] for symbol in present}, axis=1).sort_index()
        daily_members = daily_matrix.notna().sum(axis=1)
        daily_portfolio = daily_matrix.mean(axis=1).where(daily_members.ge(min_members))
        daily_portfolio_log = np.log1p(daily_portfolio.where(daily_portfolio.gt(-1)))
        daily_group = daily_portfolio_log.groupby(daily_portfolio_log.index.to_period('M'))
        portfolio_vol = (daily_group.std(ddof=1) * np.sqrt(252)).where(daily_group.count().ge(MIN_DAYS)).reindex(months)
        prefix = f"stock_group_{('all_stock' if group_name == 'all' else group_name)}"
        attrs = ['NAME', 'SYMBOL', 'DATE', 'CLOSE']
        for horizon in WINDOWS:
            compound = np.expm1(np.log1p(portfolio_return).rolling(horizon, min_periods=horizon).sum())
            add(f'{prefix}_return_{horizon}m', compound, 'stocks', stock_source, attrs, f'{group_name}: compound {horizon} monthly equal-weight constituent CLOSE returns; observed membership only, minimum {min_members}; no averaging nominal share prices; lagged one month')
        trailing(f'{prefix}_portfolio_daily_vol', portfolio_vol, 'stocks', stock_source, attrs, f'{group_name}: annualized within-month std of equal-weight observed daily portfolio log returns; minimum {min_members} constituents per day and {MIN_DAYS} valid days; weights may change with availability')
        for field, cols in (('hl_range', ['HIGH', 'LOW', 'CLOSE']), ('oc_return', ['OPEN', 'CLOSE'])):
            matrix = pd.concat({symbol: monthly_by_symbol[symbol][field] for symbol in present}, axis=1)
            mean = matrix.mean(axis=1).where(matrix.notna().sum(axis=1).ge(min_members))
            trailing(f'{prefix}_{field}', mean, 'stocks', stock_source, ['NAME', 'SYMBOL', 'DATE', *cols], f'{group_name}: equal-weight mean of own-stock normalized monthly {field}; minimum {min_members} available constituents')
        volume_matrix = pd.concat({symbol: _ratio_change(monthly_by_symbol[symbol]['volume_mean'], 1) for symbol in present}, axis=1)
        volume_change = volume_matrix.mean(axis=1).where(volume_matrix.notna().sum(axis=1).ge(min_members))
        trailing(f'{prefix}_mean_volume_change', volume_change, 'stocks', stock_source, ['NAME', 'SYMBOL', 'DATE', 'VOLUME'], f'{group_name}: equal-weight mean of constituent-relative monthly mean-volume changes; minimum {min_members}; nominal volumes never averaged across shares')
        for month in months:
            group_coverage_rows.append({'source_month': str(month), 'group': group_name, 'members_with_monthly_return': int(observed.loc[month]), 'minimum_required_members': min_members, 'requested_members': '|'.join(requested), 'available_in_file': '|'.join(present)})
    unlagged = pd.DataFrame(features, index=months)
    prediction_months = pd.period_range(months.min(), months.max() + 1, freq='M')
    monthly_features = unlagged.reindex(prediction_months).shift(1)
    monthly_features.index = prediction_months.astype(str)
    monthly_features.index.name = 'month'
    monthly_features['source_month_max'] = (prediction_months - 1).astype(str)
    attribute_audit = _attribute_rows(coe_raw, coe_source, {'YEAR': 'Year component of monthly chronology and cutoff availability', 'MONTH': 'Month component of chronology; parsed named months', 'CATEGORY': 'Separate complete A/B/C/E and any unexpected-category panels', 'ROUND': 'Order monthly auctions; last-round and between-round changes', 'QUOTA_PREMIUM': 'Monthly price level, round spread, changes and volatility', 'PREVAILING_QUOTA_PREMIUM': 'Monthly prevailing premium and price ratio; E excluded as structurally zero', 'QUOTA': 'Monthly quota, relative changes, bid pressure denominator', 'BIDS_RECEIVED': 'Monthly bids, relative changes, bid pressure numerator'}) + _attribute_rows(stock_raw, stock_source, {'NAME': 'Company identity registry and audit; preset-group annotation', 'SYMBOL': 'Every-symbol panel and normalized portfolio membership', 'DATE': 'Daily order, chronology, gap checks, monthly cutoff availability', 'OPEN': 'Normalized open-to-close movement; consistency checks', 'HIGH': 'Normalized intraday range; consistency checks', 'LOW': 'Normalized intraday range; consistency checks', 'CLOSE': 'Every-symbol returns/momentum, daily volatility, portfolio features; no cross-stock nominal price mean', 'ADJUSTED_CLOSE': 'Explicit alternative-return sensitivity candidates, excluded from main stock family', 'VOLUME': 'Every-symbol relative volume changes and normalized group changes; no cross-stock nominal volume mean'})
    feature_screening = []
    symbol_for_column = {column: symbol for symbol, columns in stock_symbol_columns.items() for column in columns}
    for column in monthly_features.columns:
        if column == 'source_month_max':
            continue
        values = monthly_features[column]
        feature_screening.append({'feature': column, 'symbol': symbol_for_column.get(column, ''), 'valid_months_all_source': int(values.notna().sum()), 'unique_values_all_source': int(values.nunique(dropna=True)), 'constant_or_all_missing_all_source': bool(values.nunique(dropna=True) <= 1), 'sensitivity_only': column in sensitivity_columns, 'screening_scope': 'Descriptive only; redo coverage/variability selection using training months only'})
    audit = {'source_files': [coe_source, stock_source], 'source_rows': {coe_source: len(coe_raw), stock_source: len(stock_raw)}, 'coe_categories': categories, 'coe_invalid_dates_or_keys': int((~valid_coe_key).sum()), 'coe_negative_or_zero_premium_by_attribute': coe_invalid_numeric, 'coe_conflicting_duplicate_key_rows': coe_conflicting_rows, 'coe_e_pqp_zero_rows': e_pqp_zero, 'coe_e_pqp_policy': 'Entire E-category PQP excluded; structurally zero values are not low market prices', 'stock_symbols': symbols, 'stock_symbol_count': len(symbols), 'stock_invalid_dates_or_keys': int((~valid_stock_key).sum()), 'stock_nonpositive_prices_or_negative_volume': invalid_stock_numeric, 'stock_ohlc_inequality_counts': {col: int(inequality[col].sum()) for col in inequality}, 'stock_ohlc_inconsistent_rows': int(stock['ohlc_inconsistent'].sum()), 'stock_ohlc_violation_over_1pct_rows': int(stock['ohlc_severe_1pct'].sum()), 'stock_ohlc_violation_over_10pct_rows': int(stock['ohlc_severe_10pct'].sum()), 'stock_ohlc_policy': 'Inconsistent OHLC and associated adjusted quotes marked missing in derived copy only; source unchanged', 'stock_conflicting_duplicate_key_rows': stock_conflicting_rows, 'symbol_slugs': symbol_slugs, 'stock_symbol_columns': stock_symbol_columns, 'stock_aggregate_columns': stock_aggregate_columns, 'sensitivity_columns': sensitivity_columns, 'preset_groups': PRESET_STOCK_GROUPS, 'attribute_audit': attribute_audit, 'monthly_minimum_valid_stock_days': MIN_DAYS, 'first_prediction_month': str(prediction_months.min()), 'last_prediction_month': str(prediction_months.max()), 'feature_counts': {key: len(value) for key, value in families.items()}, 'availability_policy': 'All predictors lagged to at most rent month - 1; no forward or backward filling', 'fixed_cutoff_policy': 'Caller must freeze cutoff+1 row for all later target months; later actual auxiliary is unavailable in that scenario', 'selection_policy': 'No outcome selection here. Main pipeline must rescreen individual-stock coverage/variance on training months only, fit standardized low-dimensional PCA on one row per training month, and retain predefined summaries. Full-source screening CSV is diagnostic only.', 'effective_macro_sample_policy': 'Monthly macro values repeat across rent rows; independent temporal support is number of observed training months, not transaction count', 'adjusted_close_policy': 'Separate sensitivity only: retrospective corporate-action adjustment timing is absent', 'daily_large_return_policy': 'Threefold observed jumps flagged, retained because corporate actions cannot be verified from this source alone'}
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_features.to_csv(output_dir / 'macro_monthly_features.csv')
    pd.DataFrame(lineage).to_csv(output_dir / 'macro_lineage.csv', index=False)
    pd.DataFrame(attribute_audit).to_csv(output_dir / 'macro_attribute_audit.csv', index=False)
    pd.DataFrame(coe_monthly_audit).to_csv(output_dir / 'coe_monthly_audit.csv', index=False)
    pd.DataFrame(symbol_audit).to_csv(output_dir / 'stock_symbol_audit.csv', index=False)
    pd.DataFrame(group_coverage_rows).to_csv(output_dir / 'stock_group_monthly_coverage.csv', index=False)
    screening_frame = pd.DataFrame(feature_screening)
    screening_frame.to_csv(output_dir / 'macro_feature_screening.csv', index=False)
    screening_frame.loc[screening_frame['feature'].str.startswith('stock_')].to_csv(output_dir / 'stock_feature_screening.csv', index=False)
    (output_dir / 'macro_audit.json').write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return (monthly_features, families, lineage, _jsonable(audit))

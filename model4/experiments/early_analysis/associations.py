"""Exploratory conditional associations on development data only.

These screens never select forecasting inputs. Geographic inference clusters
by building; macro inference uses independent calendar months and a small HAC
correction. Neither set of associations identifies causal effects.
"""
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import runtime
from runtime import ANALYSIS
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
CUTOFF = '2023-10'
RIDGE_ALPHA = 0.001
MIN_MACRO_MONTHS = 18
SPATIAL_FAMILIES = ['block_structure', 'block_location', 'mrt', 'schools', 'malls']
BIN_FEATURES = ['mrt_status_open_nearest_km', 'school_level_primary_nearest_km', 'school_level_secondary_s1_s4_nearest_km', 'school_level_secondary_s1_s5_nearest_km', 'mall_nearest_km', 'cbd_raffles_place_distance_km', 'max_floor', 'building_age', 'mrt_status_open_top3_mean_km', 'mall_count_1p0km']

def _bh(values):
    """Benjamini-Hochberg q values, excluding untested hypotheses."""
    p = np.asarray(values, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid):
        order = valid[np.argsort(p[valid])]
        corrected = p[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1.0, np.minimum.accumulate(corrected[::-1])[::-1])
    return result

def _control_design(frame, include_month):
    numeric = frame[['floor_area_sqm', 'log_floor_area', 'lease_age']].astype(float)
    numeric = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(numeric))
    columns = ['flat_type', 'flat_model', 'town'] + (['month'] if include_month else [])
    categorical = frame[columns].fillna('__MISSING__').astype(str)
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True, dtype=float)
    return sparse.hstack([np.ones((len(frame), 1)), numeric, encoder.fit_transform(categorical)], format='csr')

def _residualizer(design):
    """A single Gram factorization is equivalent to a tiny-penalty Ridge fit.

    The intercept is unpenalized. Reuse the control solve across batched numeric
    exposures instead of fitting hundreds of transaction-scale models.
    """
    gram = (design.T @ design).toarray()
    gram += np.diag(np.r_[0.0, np.full(gram.shape[0] - 1, RIDGE_ALPHA)])
    from scipy.linalg import cho_factor, cho_solve
    factor = cho_factor(gram, lower=True, check_finite=False)

    def residual(values):
        values = np.asarray(values, dtype=float)
        coefficient = cho_solve(factor, np.asarray(design.T @ values), check_finite=False)
        return values - np.asarray(design @ coefficient)
    return residual

def _cluster_effect(x, y, addresses, control_count):
    denominator = float(x @ x)
    if denominator < 1e-10:
        return {'status': 'no_variation_after_controls'}
    beta = float(x @ y / denominator)
    errors = y - beta * x
    groups, labels = pd.factorize(addresses, sort=True)
    cluster_scores = np.bincount(groups, weights=x * errors, minlength=len(labels))
    n, g = (len(x), len(labels))
    if g <= 1 or n <= control_count + 1:
        return {'status': 'insufficient_clusters'}
    correction = g / (g - 1) * ((n - 1) / (n - control_count - 1))
    variance = correction * float(cluster_scores @ cluster_scores) / denominator ** 2
    standard_error = float(np.sqrt(max(variance, 0)))
    critical = float(student_t.ppf(0.975, g - 1))
    p = float(2 * student_t.sf(abs(beta / standard_error), g - 1)) if standard_error > 0 else np.nan
    return {'status': 'tested', 'rows': n, 'address_clusters': g, 'coefficient_sgd_per_unit': beta, 'cluster_standard_error': standard_error, 'ci_low_95': beta - critical * standard_error, 'ci_high_95': beta + critical * standard_error, 'residual_exposure_sd': float(np.std(x, ddof=1)), 'sgd_per_residual_sd': beta * float(np.std(x, ddof=1)), 'p_value': p, 'partial_correlation': float(np.corrcoef(x, y)[0, 1])}

def _bins(feature, raw_x, residual_x, residual_y, addresses):
    bins = pd.qcut(pd.Series(raw_x), q=8, duplicates='drop')
    rows = []
    for interval in bins.cat.categories:
        mask = bins.eq(interval).to_numpy()
        values = residual_y[mask]
        n = len(values)
        if not n:
            continue
        mean = float(values.mean())
        codes, labels = pd.factorize(addresses[mask], sort=True)
        scores = np.bincount(codes, weights=values - mean, minlength=len(labels))
        g = len(labels)
        se = np.sqrt(g / (g - 1) * float(scores @ scores) / n ** 2) if g > 1 else np.nan
        delta = student_t.ppf(0.975, max(g - 1, 1)) * se
        rows.append({'feature': feature, 'bin_left': float(interval.left), 'bin_right': float(interval.right), 'raw_exposure_mean': float(np.mean(raw_x[mask])), 'residual_exposure_mean': float(np.mean(residual_x[mask])), 'adjusted_rent_residual_mean': mean, 'ci_low_95': mean - delta, 'ci_high_95': mean + delta, 'rows': n, 'address_clusters': g, 'interpretation': 'zero-centered rent residual after dwelling, town and month controls; descriptive bin'})
    return rows

def _partial_macro(x, y, controls, calendar_positions):
    rx = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
    ry = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return {'status': 'no_variation_after_time_controls'}
    r = float(np.corrcoef(rx, ry)[0, 1])
    denominator = float(rx @ rx)
    beta = float(rx @ ry / denominator)
    scores = rx * (ry - beta * rx)
    full = np.zeros(int(calendar_positions.max()) + 1)
    full[calendar_positions] = scores
    meat = float(scores @ scores)
    for lag in (1, 2, 3):
        meat += 2 * (1 - lag / 4) * float(full[lag:] @ full[:-lag])
    df = len(x) - controls.shape[1] - 1
    variance = max(meat, 0) * len(x) / df / denominator ** 2
    se = float(np.sqrt(variance))
    p_hac = float(2 * student_t.sf(abs(beta / se), df)) if se > 0 else np.nan
    statistic = r * np.sqrt(df / max(1 - r * r, 1e-15))
    return {'status': 'tested', 'partial_correlation': r, 'coefficient_sgd_per_unit': beta, 'hac3_standard_error': se, 'p_value_hac3': p_hac, 'p_value_iid_month': float(2 * student_t.sf(abs(statistic), df)), 'degrees_of_freedom': df}

def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    out = ANALYSIS / 'results'
    out.mkdir(exist_ok=True)
    full = pd.read_pickle(ANALYSIS / 'features' / 'train_features.pkl')
    dev = full.loc[full['month'] <= CUTOFF].copy()
    del full
    assert dev['month'].max() == CUTOFF and dev['month'].nunique() == 34
    families = json.loads((ANALYSIS / 'features' / 'feature_families.json').read_text())
    lineage = pd.read_csv(ANALYSIS / 'audit' / 'feature_lineage.csv').set_index('feature')
    target = dev['target'].to_numpy(dtype=float)
    design = _control_design(dev, include_month=True)
    candidate_cols = list(dict.fromkeys((c for family in SPATIAL_FAMILIES for c in families.get(family, []) if c in dev and pd.api.types.is_numeric_dtype(dev[c]))))
    records, bin_rows, masks = ([], [], {})
    family_of = {c: family for family in SPATIAL_FAMILIES for c in families.get(family, [])}
    for feature in candidate_cols:
        values = dev[feature].to_numpy(dtype=float)
        valid = np.isfinite(values)
        coverage = float(valid.mean())
        base = {'feature': feature, 'family': family_of[feature], 'training_coverage': coverage}
        if coverage < 0.9:
            records.append(dict(base, status='coverage_below_90pct'))
        elif np.unique(values[valid]).size < 2:
            records.append(dict(base, status='constant_in_development'))
        else:
            key = valid.tobytes()
            masks.setdefault(key, {'mask': valid, 'columns': []})['columns'].append(feature)
    for group in masks.values():
        mask, columns = (group['mask'], group['columns'])
        residualize = _residualizer(design[mask])
        y = residualize(target[mask])
        addresses = dev.loc[mask, 'address_key'].to_numpy()
        for start in range(0, len(columns), 20):
            batch = columns[start:start + 20]
            raw_values = dev.loc[mask, batch].to_numpy(dtype=float)
            residual_values = residualize(raw_values)
            for i, feature in enumerate(batch):
                result = _cluster_effect(residual_values[:, i], y, addresses, design.shape[1])
                result.update(feature=feature, family=family_of[feature], training_coverage=float(mask.mean()))
                if feature in lineage.index:
                    result['source_file'] = lineage.at[feature, 'source_file']
                    result['source_attributes'] = lineage.at[feature, 'source_attributes']
                records.append(result)
                if feature in BIN_FEATURES and result['status'] == 'tested':
                    bin_rows.extend(_bins(feature, raw_values[:, i], residual_values[:, i], y, addresses))
    spatial = pd.DataFrame(records)
    spatial['bh_q_value'] = _bh(spatial.get('p_value', pd.Series(np.nan, index=spatial.index)))
    spatial['bh_exploratory_05'] = spatial['bh_q_value'].le(0.05)
    spatial.to_csv(out / 'controlled_spatial_associations.csv', index=False)
    pd.DataFrame(bin_rows).to_csv(out / 'adjusted_distance_bins.csv', index=False)
    composition_residual = _residualizer(_control_design(dev, include_month=False))(target)
    composition = pd.DataFrame({'month': dev['month'].to_numpy(), 'rent': target, 'composition_residual': composition_residual}).groupby('month').mean()
    monthly = pd.read_pickle(ANALYSIS / 'features' / 'macro_monthly.pkl').reindex(composition.index)
    months = pd.PeriodIndex(composition.index, freq='M')
    positions = np.array([period.ordinal - months[0].ordinal for period in months])
    time_controls = np.column_stack([np.ones(len(months)), positions, np.sin(2 * np.pi * months.month / 12), np.cos(2 * np.pi * months.month / 12)])
    symbol_meta = json.loads((ANALYSIS / 'audit' / 'macro_audit.json').read_text())
    slug_to_symbol = {slug: symbol for symbol, slug in symbol_meta['symbol_slugs'].items()}
    macro_records = []
    for family in ['coe', 'stocks', 'stocks_adjusted_sensitivity']:
        for feature in families.get(family, []):
            x = monthly[feature].to_numpy(dtype=float)
            valid = np.isfinite(x)
            n = int(valid.sum())
            symbol = next((symbol for slug, symbol in slug_to_symbol.items() if feature.startswith('stock_' + slug + '_')), '')
            result = {'feature': feature, 'family': family, 'symbol': symbol, 'observed_months': n, 'missing_months': len(months) - n, 'development_months': len(months), 'sensitivity_only': family.endswith('sensitivity')}
            if n < MIN_MACRO_MONTHS:
                result['status'] = 'insufficient_months_minimum_18'
            elif np.unique(x[valid]).size < 2:
                result['status'] = 'constant_in_development'
            else:
                residual_y = composition['composition_residual'].to_numpy()[valid]
                result.update(_partial_macro(x[valid], residual_y, time_controls[valid], positions[valid]))
                result['raw_monthly_rent_correlation'] = float(np.corrcoef(x[valid], composition['rent'].to_numpy()[valid])[0, 1])
            macro_records.append(result)
    macro = pd.DataFrame(macro_records)
    macro['bh_q_value_hac3'] = _bh(macro.get('p_value_hac3', pd.Series(np.nan, index=macro.index)))
    macro['bh_exploratory_05'] = macro['bh_q_value_hac3'].le(0.05)
    macro.to_csv(out / 'monthly_macro_screen.csv', index=False)
    notes = {'development_cutoff': CUTOFF, 'development_rows': len(dev), 'development_months': len(months), 'confirmation_labels_used': False, 'model_column_selection_uses_this_screen': False, 'spatial_controls': 'area, log area, lease age, flat type, flat model, town, approval-month fixed effects', 'composition_controls': 'same dwelling/town controls without month fixed effects, preserving temporal variation', 'macro_time_controls': 'intercept, linear calendar trend, annual sine and cosine', 'macro_inference': 'one observation per month; Newey-West HAC with 3 calendar-month lags; t reference distribution is approximate with only 34 months', 'spatial_inference': 'FWL residual slope, tiny alpha=0.001 Ridge control fit, building-cluster sandwich SE with finite-sample correction', 'multiple_comparisons': 'BH separately across tested spatial candidates and tested macro candidates; exploratory association, no causal claim or model selection', 'bins': 'eight quantile bins when distinct values allow; zero-centered conditional rent residual; building-cluster intervals; control-fit uncertainty not propagated', 'static_spatial_limit': 'given snapshot, incomplete opening histories; cannot interpret snapshot open as historical operation', 'limitations': 'linear/additive controls and omitted unit floor/renovation/tenant attributes can leave confounding; serial and spatial dependence may remain; highly correlated exposures tested separately', 'spatial_status_counts': spatial['status'].value_counts().to_dict(), 'macro_status_counts': macro['status'].value_counts().to_dict(), 'all_symbols_in_screen': sorted(set(macro['symbol']) - set([''])), 'bin_features': sorted({row['feature'] for row in bin_rows}), 'elapsed_seconds': round(time.perf_counter() - started, 3)}
    (out / 'association_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    print(json.dumps({key: notes[key] for key in ['development_rows', 'development_months', 'spatial_status_counts', 'macro_status_counts', 'elapsed_seconds']}), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()

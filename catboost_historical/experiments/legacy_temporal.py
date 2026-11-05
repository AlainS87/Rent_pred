#!/usr/bin/env python3
"""Leakage-safe, chronological ablation of CatBoost seasonality and market trend.

A: original CatBoost (180 iterations by default)
B: A + cyclic month features
C: A * forward-only composition-adjusted market trend
D: B * same trend
E: longer-trained B * same trend (capacity ablation)

No official rental data or test target data is used for features or trends.
"""
from __future__ import annotations

import argparse
from hdb_rent.data import DEFAULT_DATA_DIR, OUTPUT_DIR
from hdb_rent.temporal import cyclic, adjusted_market_index, forecast_index
from hdb_rent.evaluation import score
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from hdb_rent.base import CAT, features, model, make_submission

DEFAULT_FOLDS = [
    ('long18', '2023-10', '2025-03'),
    ('medium12', '2024-04', '2025-03'),
    ('short3', '2025-01', '2025-03'),
]
MODELS = ('A', 'B', 'C', 'D', 'E')










def fit_predict(variant: str, X: pd.DataFrame, y: np.ndarray, training: np.ndarray,
                heldout: np.ndarray, a) -> np.ndarray:
    iters = a.long_iterations if variant == 'E' else a.iterations
    m = model(iters, a.seed, a.threads, a.depth, a.learning_rate)
    m.fit(X.loc[training], y[training], cat_features=CAT,
          verbose=False)
    p = np.asarray(m.predict(X.loc[heldout]), dtype=float)
    return p


def get_folds(arg: str):
    if not arg:
        return DEFAULT_FOLDS
    result = []
    for i, item in enumerate(arg.split(',')):
        start, end = item.split(':')
        if start > end:
            raise ValueError('Fold start after fold end')
        result.append((f'fold{i+1}', start, end))
    return result


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def load_data(a, test_needed):
    train = pd.read_csv(a.data_dir / a.train_file)
    test = pd.read_csv(a.data_dir / a.test_file) if test_needed else None
    if 'MONTHLY_RENT' not in train.columns:
        raise ValueError('train.csv must contain MONTHLY_RENT')
    if test is not None and 'MONTHLY_RENT' in test.columns:
        raise ValueError('test.csv unexpectedly contains MONTHLY_RENT')
    if not pd.Series(train['RENT_APPROVAL_DATE']).astype(str).str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('Bad training month format')
    y = train['MONTHLY_RENT'].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite training targets')
    if test is not None:
        joint = pd.concat([train.drop(columns=['MONTHLY_RENT']), test], ignore_index=True)
    else:
        joint = train.drop(columns=['MONTHLY_RENT'])
    X, raw = features(joint, a.data_dir)
    X = X.reset_index(drop=True)
    return train, test, X, y, raw


def ablate(a, train, X, y):
    folds = get_folds(a.folds)
    dates = train['RENT_APPROVAL_DATE'].astype(str).to_numpy()
    cyclic_X = cyclic(X)
    metrics = []
    sample_predictions = []
    for name, start, end in folds:
        fit = dates < start
        val = (dates >= start) & (dates <= end)
        if not fit.any() or not val.any():
            raise ValueError(f'Empty training/validation for {name}: {start}:{end}')
        if dates[fit].max() >= dates[val].min():
            raise RuntimeError('Temporal leakage!')
        print(f'\n[{name}] train {fit.sum():,} through {dates[fit].max()}, '
              f'validation {val.sum():,} ({start} to {end})', flush=True)
        idx = adjusted_market_index(train.loc[fit], y[fit])
        trend = forecast_index(idx, dates[val], lookback=a.lookback,
                               decay_months=a.decay_months, cap_monthly=a.cap_monthly)
        scale = np.exp(trend * a.trend_strength)
        print(f'  extrapolated market ratio: min={scale.min():.4f}, max={scale.max():.4f}', flush=True)
        idx.to_csv(a.output_dir / f'market_index_{name}.csv', index=False)
        pred_a = fit_predict('A', X, y, fit, val, a)
        pred_b = fit_predict('B', cyclic_X, y, fit, val, a)
        pred_e = fit_predict('E', cyclic_X, y, fit, val, a)
        predictions = {'A': pred_a, 'B': pred_b, 'C': pred_a * scale,
                       'D': pred_b * scale, 'E': pred_e * scale}
        for method, pred in predictions.items():
            sc = score(y[val], pred)
            metrics.append({'fold': name, 'val_from': start, 'val_to': end,
                            'train_end': dates[fit].max(), 'model': method, **sc})
            print(f'  {method}  RMSE {sc["rmse"]:.3f}  MAE {sc["mae"]:.3f} '
                  f'bias {sc["bias_pred_minus_actual"]:+.2f}', flush=True)
        # One row per validation sample, A-E columns, for error diagnosis.
        fold_preds = pd.DataFrame({'fold': name, 'train_end': dates[fit].max(),
                                  'date': dates[val], 'actual': y[val],
                                  'market_scale': scale})
        for method, pred in predictions.items():
            fold_preds[method] = pred
        sample_predictions.append(fold_preds)
    by_fold = pd.DataFrame(metrics)
    by_fold.to_csv(a.output_dir / 'ablation_by_fold.csv', index=False)
    df = pd.concat(sample_predictions, ignore_index=True)
    df.to_csv(a.output_dir / 'oof_predictions.csv', index=False)
    summary = (by_fold.groupby('model', as_index=False)
               .agg(mean_fold_rmse=('rmse', 'mean'), worst_fold_rmse=('rmse', 'max'),
                    mean_fold_mae=('mae', 'mean'), mean_fold_bias=('bias_pred_minus_actual', 'mean'))
               .sort_values('mean_fold_rmse'))
    summary.to_csv(a.output_dir / 'ablation_summary.csv', index=False)
    by_month = []
    for (fold, month), g in df.groupby(['fold', 'date'], sort=True):
        for method in MODELS:
            by_month.append({'fold': fold, 'month': month, 'model': method,
                             **score(g.actual.to_numpy(), g[method].to_numpy())})
    pd.DataFrame(by_month).to_csv(a.output_dir / 'ablation_by_month.csv', index=False)
    # Duration since the training cutoff, not calendar-year groupings.
    elapsed = (pd.PeriodIndex(df['date'], freq='M').asi8 -
               pd.PeriodIndex(df['train_end'], freq='M').asi8)
    df['horizon_months'] = elapsed
    df['horizon_bucket'] = pd.cut(elapsed, bins=[0, 3, 6, 12, 24, 999],
                                  labels=['01-03', '04-06', '07-12', '13-24', '25+'])
    rows_horizon = []
    for (fold, bucket), g in df.groupby(['fold', 'horizon_bucket'], observed=True):
        for method in MODELS:
            rows_horizon.append({'fold': fold, 'horizon': str(bucket), 'model': method,
                                 **score(g['actual'].to_numpy(), g[method].to_numpy())})
    pd.DataFrame(rows_horizon).to_csv(a.output_dir / 'ablation_by_horizon.csv', index=False)
    result = {'folds': [{'name': n, 'start': s, 'end': e} for n, s, e in folds],
              'baseline_iterations': a.iterations, 'long_iterations': a.long_iterations,
              'depth': a.depth, 'learning_rate': a.learning_rate,
              'trend_lookback_months': a.lookback, 'trend_decay_months': a.decay_months,
              'trend_max_monthly_log_slope': a.cap_monthly,
              'trend_strength': a.trend_strength,
              'ranking': summary.to_dict('records'),
              'selection_metric': 'equal-weight mean across chronological folds',
              'note': 'Fold windows overlap; do not compute an unqualified pooled OOF score. '
                      'No validation or test rental targets were used to calculate trends.'}
    save_json(a.output_dir / 'experiment_config_and_results.json', result)
    print('\n== Fold-average RMSE (lower is better) ==\n' + summary.to_string(index=False), flush=True)
    return str(summary.iloc[0]['model'])


def submit(a, train, test, X, y, chosen):
    if test is None:
        raise ValueError('Must provide test data to submit')
    print(f'\nBuilding submission for method {chosen}', flush=True)
    tr_n = len(train)
    train_flag = np.ones(tr_n, dtype=bool)
    holdout_flag = np.ones(len(test), dtype=bool)
    use_cyclic = chosen in ('B', 'D', 'E')
    all_x = cyclic(X) if use_cyclic else X
    iters = a.long_iterations if chosen == 'E' else a.iterations
    m = model(iters, a.seed, a.threads, a.depth, a.learning_rate)
    m.fit(all_x.iloc[:tr_n], y, cat_features=CAT, verbose=False)
    pred = np.asarray(m.predict(all_x.iloc[tr_n:]), dtype=float)
    if chosen in ('C', 'D', 'E'):
        index = adjusted_market_index(train, y)
        index.to_csv(a.output_dir / 'market_index_full_train.csv', index=False)
        last_month = train.RENT_APPROVAL_DATE.astype(str).max()
        # Test includes March 2025 and later; 2025-03 is a boundary month.
        # Test dates at or before the final training month receive no extrapolation.
        test_months = test.RENT_APPROVAL_DATE.astype(str).to_numpy()
        future = test_months > last_month
        scale = np.ones(len(test), dtype=float)
        if future.any():
            scale[future] = np.exp(a.trend_strength * forecast_index(index, test_months[future],
                                 lookback=a.lookback, decay_months=a.decay_months,
                                 cap_monthly=a.cap_monthly))
        pred *= scale
    if not np.isfinite(pred).all():
        raise RuntimeError('Prediction contains nonfinite values')
    out = a.output_dir / f'submission_{chosen}.csv'
    make_submission(pred, len(test)).to_csv(out, index=False)
    m.save_model(str(a.output_dir / f'submission_model_{chosen}.cbm'))
    print(f'Saved {out} with {len(test):,} rows and Id,Predicted columns', flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--train-file', default='train.csv')
    p.add_argument('--test-file', default='test.csv')
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'legacy_temporal')
    p.add_argument('--mode', choices=['ablate', 'submit', 'both'], default='ablate')
    p.add_argument('--submit-model', choices=(*MODELS, 'best'), default='best')
    p.add_argument('--folds', default='', help='Comma-separated start:end e.g. 2024-04:2025-03,2025-01:2025-03')
    p.add_argument('--iterations', type=int, default=180)
    p.add_argument('--long-iterations', type=int, default=700)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=0.055)
    p.add_argument('--threads', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lookback', type=int, default=12)
    p.add_argument('--decay-months', type=float, default=6.0)
    p.add_argument('--cap-monthly', type=float, default=0.012)
    p.add_argument('--trend-strength', type=float, default=1.0,
                   help='Multiplier for predicted log market change. 1=full; 0.5=half correction.')
    a = p.parse_args()
    if a.iterations <= 0 or a.long_iterations <= 0 or a.decay_months <= 0 or not (0 <= a.trend_strength <= 2):
        p.error('Iterations and decay-months must be positive; trend-strength must be in [0,2]')
    if a.mode == 'submit' and a.submit_model == 'best' and not (a.output_dir / 'ablation_summary.csv').exists():
        p.error('Run --mode ablate first to use --submit-model best, or specify A/B/C/D/E')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, X, y, raw = load_data(a, a.mode != 'ablate')
    print(f'Rows train={len(train):,}, test={len(test) if test is not None else 0:,}, '
          f'features baseline={X.shape[1]}, HDB matched={raw.LATITUDE.iloc[:len(train)].notna().mean():.2%}', flush=True)
    best = None
    if a.mode in ('ablate', 'both'):
        best = ablate(a, train, X.iloc[:len(train)], y)
    if a.mode in ('submit', 'both'):
        selection = a.submit_model
        if selection == 'best':
            if best is None:
                selection = str(pd.read_csv(a.output_dir / 'ablation_summary.csv').iloc[0].model)
            else:
                selection = best
        submit(a, train, test, X, y, selection)


if __name__ == '__main__':
    main()

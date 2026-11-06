#!/usr/bin/env python3
"""Controlled CatBoost HDB rental experiments: isolate seasonality and market trend.

A: original features, unadjusted CatBoost
B: original features + (month_sin, month_cos)
C: identical model to A, then apply causal forecast trend (multiplicative)
D: identical model to B, then apply causal forecast trend (multiplicative)

All four models share EXACTLY the same iteration count / seed / hyperparameters.
Across a grid of iterations, train only the longest model for each feature set
and use its tree prefixes for each smaller iteration count. No early stopping.
"""
from __future__ import annotations

import argparse
from hdb_rent.data import DEFAULT_DATA_DIR, OUTPUT_DIR
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hdb_rent.base import CAT, features, make_submission, model
from hdb_rent.temporal import adjusted_market_index, cyclic, forecast_index
from hdb_rent.evaluation import score

# Nonoverlapping *development* folds, plus a reserved last-period diagnostic.
# Its labels do not participate in this script's parameter selection, but
# prior experiments already inspected this period: NOT a truly unseen holdout.
STRICT_FOLDS = (
    ('dev_2023_h2', '2023-07', '2023-12', 'development'),
    ('dev_2024_h1', '2024-01', '2024-06', 'development'),
    ('dev_2024_h2', '2024-07', '2024-12', 'development'),
    ('holdout_2025_q1', '2025-01', '2025-03', 'holdout'),
)
# Optional directly-comparable historical folds; overlap by design.
LEGACY_FOLDS = (
    ('long18', '2023-10', '2025-03', 'development'),
    ('medium12', '2024-04', '2025-03', 'development'),
    ('short3', '2025-01', '2025-03', 'development'),
)
VARIANTS = ('A', 'B', 'C', 'D')


def parse_iterations(s: str) -> list[int]:
    try:
        v = [int(x.strip()) for x in s.split(',')]
    except ValueError as e:
        raise ValueError('--iterations must be comma-separated integers') from e
    if not v or min(v) <= 0 or len(set(v)) != len(v):
        raise ValueError('--iterations must list unique positive numbers')
    return sorted(v)


def get_folds(preset: str):
    return STRICT_FOLDS if preset == 'strict' else LEGACY_FOLDS


def check_folds(folds, dates):
    observed = pd.Series(dates, dtype='string')
    for name, start, end, role in folds:
        if start > end:
            raise ValueError(f'{name}: validation start after end')
        train_mask = (observed < start).to_numpy(dtype=bool)
        val_mask = ((observed >= start) & (observed <= end)).to_numpy(dtype=bool)
        if not train_mask.any() or not val_mask.any():
            raise ValueError(f'{name}: missing train or validation samples')
        if observed[train_mask].max() >= observed[val_mask].min():
            raise RuntimeError(f'{name}: future-label leakage')
    return True


def load_inputs(args, with_test=False):
    tr_path = args.data_dir / args.train_file
    train = pd.read_csv(tr_path)
    test = pd.read_csv(args.data_dir / args.test_file) if with_test else None
    required = {'RENT_APPROVAL_DATE', 'MONTHLY_RENT', 'TOWN', 'FLAT_TYPE'}
    if not required.issubset(train.columns):
        raise ValueError(f'Missing training columns: {sorted(required - set(train.columns))}')
    if test is not None and 'MONTHLY_RENT' in test:
        raise ValueError('test.csv must not contain MONTHLY_RENT')
    if not train.RENT_APPROVAL_DATE.astype(str).str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('Bad training RENT_APPROVAL_DATE format')
    y = train.MONTHLY_RENT.to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Training rents must be positive and finite')
    joint = (pd.concat([train.drop(columns=['MONTHLY_RENT']), test], ignore_index=True)
             if with_test else train.drop(columns=['MONTHLY_RENT']))
    X, raw = features(joint, args.data_dir)
    X = X.reset_index(drop=True)
    if len(X) != len(joint):
        raise RuntimeError('Feature merge unexpectedly altered row count')
    if test is not None and len(X) != len(train) + len(test):
        raise RuntimeError('Test row alignment failed')
    return train, test, X, y, raw


def market_multipliers(train_subset, y_subset, val_months, args):
    """Fit on training labels only; don't use validation/test labels."""
    historic_index = adjusted_market_index(train_subset, y_subset)
    final_month = str(historic_index.month.max())
    months = np.asarray(val_months, dtype=str)
    mult = np.ones(len(months), dtype=float)
    after = months > final_month
    if after.any() and args.trend_strength != 0:
        projected = forecast_index(historic_index, months[after], lookback=args.lookback,
                                   decay_months=args.decay_months,
                                   cap_monthly=args.cap_monthly)
        mult[after] = np.exp(args.trend_strength * projected)
    if not np.isfinite(mult).all():
        raise RuntimeError('Invalid market multipliers')
    return mult, historic_index


def prediction_prefixes(X_train, y_train, X_val, args, counts):
    """One expensive fit, one prediction per tree prefix (no early stopping)."""
    max_iter = max(counts)
    predictor = model(max_iter, args.seed, args.threads, args.depth, args.learning_rate)
    predictor.fit(X_train, y_train, cat_features=CAT, verbose=False)
    # Prefix predictions agree with separately trained CatBoost models as long
    # as hyperparameters, data ordering, seed and feature columns are unchanged.
    pred = {}
    for n in counts:
        pred[n] = np.asarray(predictor.predict(X_val, ntree_end=n), dtype=float)
    return pred


def validation(args, train, X, y):
    counts = parse_iterations(args.iterations)
    dates = train.RENT_APPROVAL_DATE.astype(str).to_numpy()
    folds = get_folds(args.fold_preset)
    check_folds(folds, dates)
    X_cyclic = cyclic(X)
    metrics = []
    detail = []
    by_month_rows = []
    by_horizon_rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Running {args.fold_preset} folds, iters={counts}, trend-strength={args.trend_strength}', flush=True)
    for name, start, end, role in folds:
        tr = dates < start
        va = (dates >= start) & (dates <= end)
        print(f'\n[{name} | {role}] fit={tr.sum():,} through {dates[tr].max()}, '
              f'validate={va.sum():,} {start}..{end}', flush=True)
        scale, historic = market_multipliers(train.loc[tr], y[tr], dates[va], args)
        historic.to_csv(args.output_dir / f'market_index_{name}.csv', index=False)
        print(f'  predicted rent multiplier {scale.min():.5f}..{scale.max():.5f}', flush=True)
        base = prediction_prefixes(X.loc[tr], y[tr], X.loc[va], args, counts)
        seas = prediction_prefixes(X_cyclic.loc[tr], y[tr], X_cyclic.loc[va], args, counts)
        horizons = (pd.PeriodIndex(dates[va], freq='M').asi8 -
                    pd.Period(dates[tr].max(), freq='M').ordinal)
        buckets = pd.cut(horizons, bins=[0, 3, 6, 12, 18, 24, 999],
                         labels=['01-03', '04-06', '07-12', '13-18', '19-24', '25+'])
        truth = y[va]
        for n in counts:
            predictions = {'A': base[n], 'B': seas[n],
                           'C': base[n] * scale, 'D': seas[n] * scale}
            for variant, pred in predictions.items():
                met = score(truth, pred)
                metrics.append({'fold': name, 'role': role, 'train_end': dates[tr].max(),
                                'val_start': start, 'val_end': end, 'iterations': n,
                                'model': variant, **met})
                if n == counts[0] or n == 1200 or n == counts[-1]:
                    print(f'  {variant}{n}: RMSE {met["rmse"]:.3f}, '
                          f'MAE {met["mae"]:.3f}, bias {met["bias_pred_minus_actual"]:+.1f}', flush=True)
                # Save *sample predictions* only for 1200 if selected, else max,
                # avoids writing huge wide-format matrix for every iteration.
                if n == (1200 if 1200 in counts else max(counts)):
                    detail.append(pd.DataFrame({'fold': name, 'role': role,
                                                'date': dates[va], 'actual': truth,
                                                'iterations': n, 'model': variant,
                                                'prediction': pred, 'market_multiplier': scale}))
                val_frame = pd.DataFrame({'date': dates[va], 'y': truth, 'p': pred,
                                          'horizon': buckets})
                for month, grp in val_frame.groupby('date', sort=True):
                    by_month_rows.append({'fold': name, 'role': role, 'iterations': n,
                                          'model': variant, 'month': month,
                                          **score(grp.y, grp.p)})
                for bucket, grp in val_frame.groupby('horizon', observed=True):
                    by_horizon_rows.append({'fold': name, 'role': role, 'iterations': n,
                                            'model': variant, 'horizon': str(bucket),
                                            **score(grp.y, grp.p)})
    all_metrics = pd.DataFrame(metrics)
    all_metrics.to_csv(args.output_dir / 'scores_by_fold.csv', index=False)
    pd.DataFrame(by_month_rows).to_csv(args.output_dir / 'scores_by_month.csv', index=False)
    pd.DataFrame(by_horizon_rows).to_csv(args.output_dir / 'scores_by_horizon.csv', index=False)
    if detail:
        pd.concat(detail, ignore_index=True).to_csv(args.output_dir / 'sample_predictions.csv', index=False)

    dev = all_metrics.loc[all_metrics.role == 'development'].copy()
    summary = (dev.groupby(['iterations', 'model'], as_index=False)
               .agg(dev_mean_rmse=('rmse', 'mean'), dev_worst_rmse=('rmse', 'max'),
                    dev_mean_mae=('mae', 'mean'), dev_mean_bias=('bias_pred_minus_actual', 'mean'),
                    n_dev_folds=('rmse', 'size'))
               .sort_values(['dev_mean_rmse', 'iterations', 'model']).reset_index(drop=True))
    baseline = summary.loc[summary.model == 'A', ['iterations', 'dev_mean_rmse']].rename(
        columns={'dev_mean_rmse': 'baseline_A_same_iterations_rmse'})
    summary = summary.merge(baseline, on='iterations', how='left', validate='m:1')
    summary['delta_vs_A_same_iterations'] = (summary.dev_mean_rmse - summary.baseline_A_same_iterations_rmse)
    holdout = all_metrics.loc[all_metrics.role == 'holdout', ['iterations', 'model', 'rmse', 'mae', 'bias_pred_minus_actual']]
    holdout = holdout.rename(columns={'rmse': 'holdout_rmse', 'mae': 'holdout_mae',
                                      'bias_pred_minus_actual': 'holdout_bias'})
    if len(holdout):
        summary = summary.merge(holdout, on=['iterations', 'model'], how='left', validate='1:1')
    summary.to_csv(args.output_dir / 'scores_summary.csv', index=False)

    # 1200-only view of the effects with iteration count strictly controlled.
    focus = 1200 if 1200 in counts else counts[0]
    summary.loc[summary.iterations == focus].sort_values('model').to_csv(
        args.output_dir / 'ablation_fixed_iterations.csv', index=False)
    sweep = summary.sort_values(['model', 'iterations'])
    sweep.to_csv(args.output_dir / 'iteration_sweep.csv', index=False)

    best = summary.iloc[0]
    selected = {'model': str(best['model']), 'iterations': int(best['iterations']),
                'trend_strength': args.trend_strength, 'dev_mean_rmse': float(best['dev_mean_rmse']),
                'selection': 'development folds only; 2025-Q1 diagnostic excluded, but previously inspected in earlier experiments'}
    (args.output_dir / 'selected_config.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    manifest = {'fold_preset': args.fold_preset, 'folds': [dict(name=n,start=s,end=e,role=r) for n,s,e,r in folds],
                'iterations': counts, 'trend_strength': args.trend_strength,
                'depth': args.depth, 'learning_rate': args.learning_rate, 'seed': args.seed,
                'market_trend': {'lookback_months': args.lookback,
                                 'decay_months': args.decay_months,
                                 'cap_monthly_log_slope': args.cap_monthly},
                'train_rows': len(train), 'train_file': args.train_file,
                'feature_count_base': X.shape[1],
                'feature_count_cyclic': X_cyclic.shape[1],
                'warning': 'Legacy folds overlap. Strict preset has disjoint validation months '
                           'and a final 2025-Q1 diagnostic (previously inspected, not truly unseen). Even strict historical '
                           'validation may not predict 2026 leaderboard performance.'}
    (args.output_dir / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\n== Development-fold ranking (holdout NOT included) ==', flush=True)
    print(summary[['model', 'iterations', 'dev_mean_rmse', 'delta_vs_A_same_iterations']]
          .to_string(index=False), flush=True)
    print(f'\nSelected on development folds: {selected}', flush=True)
    print(f'Results in: {args.output_dir.resolve()}', flush=True)
    return selected


def submission(args, train, test, X, y):
    if test is None:
        raise ValueError('Test input is required')
    if args.submit_model == 'best':
        config_path = args.output_dir / 'selected_config.json'
        if not config_path.exists():
            raise ValueError('For --submit-model best, first run --mode validate with the SAME output directory')
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
        name, n = cfg['model'], int(cfg['iterations'])
        # Restore the selected validation configuration, rather than silently
        # using default hyperparameters if the user ran validation with overrides.
        manifest_path = args.output_dir / 'experiment_manifest.json'
        if not manifest_path.exists():
            raise ValueError('Missing experiment_manifest.json for the chosen model')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        args.trend_strength = float(cfg['trend_strength'])
        for attr, source in [('depth', 'depth'), ('learning_rate', 'learning_rate'),
                             ('seed', 'seed'), ('lookback', 'lookback_months'),
                             ('decay_months', 'decay_months'), ('cap_monthly', 'cap_monthly_log_slope')]:
            value = (manifest['market_trend'][source] if source in manifest['market_trend']
                     else manifest[source])
            setattr(args, attr, value)
        if manifest.get('train_rows') != len(train):
            raise ValueError('Training row count differs from selected experiment')
    else:
        name, n = args.submit_model, args.submit_iterations
        if n is None:
            raise ValueError('Specify --submit-iterations when --submit-model is A/B/C/D')
    if name not in VARIANTS or int(n) <= 0:
        raise ValueError('Invalid submission model or iterations')
    print(f'Fitting {name} at {n} iterations on {len(train):,} training samples', flush=True)
    X_used = cyclic(X) if name in ('B','D') else X
    ntrain = len(train)
    cb = model(n, args.seed, args.threads, args.depth, args.learning_rate)
    cb.fit(X_used.iloc[:ntrain], y, cat_features=CAT, verbose=False)
    pred = np.asarray(cb.predict(X_used.iloc[ntrain:]), dtype=float)
    if name in ('C','D'):
        scale, index = market_multipliers(train, y, test.RENT_APPROVAL_DATE.astype(str).to_numpy(), args)
        index.to_csv(args.output_dir / f'submission_market_index_{name}{n}.csv', index=False)
        pred = pred * scale
    out = args.output_dir / f'submission_{name}{n}.csv'
    make_submission(pred, len(test)).to_csv(out, index=False)
    cb.save_model(str(args.output_dir / f'model_{name}{n}.cbm'))
    report = {'model': name, 'iterations': int(n), 'trend_strength': args.trend_strength,
              'test_rows': len(test), 'columns': ['Id', 'Predicted'],
              'pred_min': float(pred.min()), 'pred_mean': float(pred.mean()),
              'pred_max': float(pred.max()), 'file': str(out)}
    (args.output_dir / f'submission_{name}{n}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {out} with {len(test):,} rows and columns Id,Predicted', flush=True)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--train-file', default='train.csv')
    p.add_argument('--test-file', default='test.csv')
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'controlled_sweep')
    p.add_argument('--mode', choices=['validate', 'submit', 'both'], default='validate')
    p.add_argument('--fold-preset', choices=['strict', 'legacy'], default='strict')
    p.add_argument('--iterations', default='800,1200,1600,2000',
                   help='Comma-separated grid for validation (one training run per feature set)')
    p.add_argument('--trend-strength', type=float, default=0.5)
    p.add_argument('--submit-model', choices=[*VARIANTS, 'best'], default='best')
    p.add_argument('--submit-iterations', type=int, default=None)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=0.055)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--lookback', type=int, default=12)
    p.add_argument('--decay-months', type=float, default=6.0)
    p.add_argument('--cap-monthly', type=float, default=0.012)
    a = p.parse_args(argv)
    try:
        parse_iterations(a.iterations)
    except ValueError as e:
        p.error(str(e))
    if a.trend_strength < 0 or a.trend_strength > 2:
        p.error('--trend-strength must be in [0,2]')
    if a.decay_months <= 0 or a.lookback < 2 or a.cap_monthly <= 0 or a.depth <= 0:
        p.error('Invalid trend or depth parameter')
    return a


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, X, y, raw = load_inputs(args, with_test=args.mode in ('submit','both'))
    print(f'Read {len(train):,} train rows and {0 if test is None else len(test):,} test rows; '
          f'base features={X.shape[1]}, geo-match={raw.LATITUDE.iloc[:len(train)].notna().mean():.2%}', flush=True)
    if args.mode in ('validate', 'both'):
        validation(args, train, X.iloc[:len(train)], y)
    if args.mode in ('submit','both'):
        submission(args, train, test, X, y)


if __name__ == '__main__':
    main()

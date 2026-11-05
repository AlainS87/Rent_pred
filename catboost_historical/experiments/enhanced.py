#!/usr/bin/env python3
"""F1 vs G1-G4: causal fixed-cutoff HDB rental-history ablation.

All variants reuse the exact F1 CatBoost/market-trend pipeline; G variants
append only rental-history columns. Never reads public HDB rent labels.
"""
from __future__ import annotations

import argparse
from hdb_rent.data import DEFAULT_DATA_DIR, OUTPUT_DIR
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from hdb_rent.evaluation import score
from experiments.historical import load, get_folds, market_multiplier
from hdb_rent.historical_features import HistoricRents, feature_columns
from hdb_rent.enhanced_history import AnchoredRents, variant_columns
from hdb_rent.base import CAT, model, make_submission

VALID = ('F1', 'G1', 'G2', 'G3', 'G4')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--train-file', default='train.csv')
    p.add_argument('--test-file', default='test.csv')
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'g_history')
    p.add_argument('--mode', choices=['validate', 'submit', 'both'], default='validate')
    p.add_argument('--fold-preset', choices=['strict', 'legacy'], default='strict')
    p.add_argument('--folds', default='', help='YYYY-MM:YYYY-MM,YYYY-MM:YYYY-MM')
    p.add_argument('--variants', default='F1,G1,G2,G3,G4')
    p.add_argument('--submit-variant', choices=list(VALID), default='G1')
    p.add_argument('--iterations', type=int, default=1600)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=0.055)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--trend-strength', type=float, default=0.5)
    p.add_argument('--lookback', type=int, default=12)
    p.add_argument('--decay-months', type=float, default=6.)
    p.add_argument('--cap-monthly', type=float, default=0.012)
    p.add_argument('--max-neighbors', type=int, default=24)
    p.add_argument('--radius-km', type=float, default=1.2)
    a = p.parse_args(argv)
    a.variants = [x.strip().upper() for x in a.variants.split(',')]
    if not a.variants or len(set(a.variants)) != len(a.variants) or any(v not in VALID for v in a.variants):
        p.error('--variants must contain distinct entries chosen from F1,G1,G2,G3,G4')
    if a.iterations < 1 or a.depth < 1 or a.threads == 0 or a.radius_km <= 0 or a.max_neighbors < 1:
        p.error('Invalid model or geography argument')
    if not 0 <= a.trend_strength <= 2 or a.lookback < 2 or a.decay_months <= 0:
        p.error('Invalid trend argument')
    return a


def hist_features(reference, targets, queries, cutoff, variants, args, compare_f1=False):
    """Return [F1 canonical columns][G1..G4 incremental columns]."""
    t0 = time.monotonic()
    ref = reference.reset_index(drop=True)
    query = queries.reset_index(drop=True)
    tier = max([int(v[1:]) for v in variants if v.startswith('G')], default=0)
    if tier:
        idx = AnchoredRents(ref, targets, args.max_neighbors, args.radius_km)
        old = idx.h.materialize(query, 1)
        new = idx.materialize(query, cutoff, tier)
        result = pd.concat([old, new], axis=1)
    else:
        result = HistoricRents(ref, targets, args.max_neighbors, args.radius_km).materialize(query, 1)
    if compare_f1:
        other = HistoricRents(ref, targets, args.max_neighbors, args.radius_km).materialize(query, 1)
        pd.testing.assert_frame_equal(result[feature_columns(1)], other, check_exact=True)
    if set(feature_columns(1)) - set(result.columns):
        raise RuntimeError('F1 reference features missing')
    print(f'  Generated {len(result):,} rows, {len(result.columns)} history columns '
          f'in {time.monotonic()-t0:.1f}s', flush=True)
    return result


def select_columns(base, extra, variant):
    cols = feature_columns(1) + variant_columns(variant)
    if not set(cols).issubset(extra.columns):
        raise RuntimeError('A requested history column does not exist')
    result = pd.concat([base.reset_index(drop=True), extra[cols].reset_index(drop=True)], axis=1)
    if result.columns.duplicated().any():
        raise RuntimeError('Duplicate features')
    return result


def write_scores(rows, month_rows, horizon_rows, out, variants):
    all_scores = pd.DataFrame(rows)
    if all_scores.empty:
        raise RuntimeError('Empty validation scores')
    all_scores.to_csv(out/'scores_by_fold.csv', index=False)
    pd.DataFrame(horizon_rows).to_csv(out/'scores_by_horizon.csv', index=False)
    pd.DataFrame(month_rows).to_csv(out/'scores_by_month.csv', index=False)
    dev = all_scores[all_scores.role == 'development']
    summary = (dev.groupby('model', as_index=False)
               .agg(mean_rmse=('rmse','mean'),worst_rmse=('rmse','max'),
                    mean_mae=('mae','mean'),mean_bias=('bias_pred_minus_actual','mean'),
                    n_folds=('rmse','count')))
    if 'F1' in variants:
        f1 = float(summary.loc[summary.model == 'F1', 'mean_rmse'].iloc[0])
        summary['delta_vs_F1'] = summary.mean_rmse - f1
        diag = all_scores[all_scores.role == 'diagnostic']
        if len(diag):
            diag = diag[['model','rmse','mae','bias_pred_minus_actual']].rename(columns={
                'rmse':'diagnostic_rmse','mae':'diagnostic_mae',
                'bias_pred_minus_actual':'diagnostic_bias'})
            summary = summary.merge(diag, on='model', how='left',validate='one_to_one')
    summary = summary.sort_values('mean_rmse')
    summary.to_csv(out/'scores_summary.csv', index=False)
    print('\n=== Development validation ranking ===\n'+summary.to_string(index=False), flush=True)
    return summary


def run_validate(args, train, base, raw, y):
    months = train['RENT_APPROVAL_DATE'].astype(str).to_numpy()
    folds = get_folds(args)
    by_fold, by_horizon, by_month = [], [], []
    for name, start, end, role in folds:
        tr = np.flatnonzero(months < start)
        va = np.flatnonzero((months >= start) & (months <= end))
        if not len(tr) or not len(va):
            raise ValueError(f'Empty training/validation in {name}: {start}..{end}')
        print(f'\n[{name}] fit={len(tr):,} until {months[tr].max()}, '
              f'validate={len(va):,} {start}..{end}', flush=True)
        selected = np.r_[tr, va]
        extra = hist_features(raw.iloc[tr], y[tr], raw.iloc[selected],
                              start, args.variants, args)
        scale = market_multiplier(train.iloc[tr], y[tr], months[va], args)
        offset = (pd.PeriodIndex(months[va], freq='M').asi8 -
                  pd.Period(months[tr].max(), freq='M').ordinal)
        buckets = pd.cut(offset, bins=[0,3,6,12,18,24,999],
                         labels=['01-03','04-06','07-12','13-18','19-24','25+'])
        for variant in args.variants:
            X = select_columns(base.iloc[selected], extra, variant)
            cb = model(args.iterations, args.seed, args.threads, args.depth, args.learning_rate)
            cb.fit(X.iloc[:len(tr)], y[tr], cat_features=CAT, verbose=False)
            p = np.asarray(cb.predict(X.iloc[len(tr):]), float) * scale
            met = score(y[va], p)
            by_fold.append({'fold':name, 'role':role, 'train_end':months[tr].max(),
                            'val_start':start, 'val_end':end,'model':variant,
                            'iterations':args.iterations,'feature_count':X.shape[1],**met})
            print(f'  {variant} | RMSE {met["rmse"]:.3f} | MAE {met["mae"]:.3f} | '
                  f'Bias {met["bias_pred_minus_actual"]:+.3f}', flush=True)
            for bucket in pd.unique(buckets.dropna()):
                mask = np.asarray(buckets == bucket)
                by_horizon.append({'fold':name, 'role':role, 'model':variant,
                                   'horizon':str(bucket), 'iterations':args.iterations,
                                   **score(y[va][mask], p[mask])})
            for month in np.unique(months[va]):
                mask = months[va] == month
                by_month.append({'fold':name, 'role':role,'model':variant,
                                 'month':month, 'iterations':args.iterations,
                                 **score(y[va][mask], p[mask])})
        # Partial progress is saved fold-by-fold, so an interruption does not
        # discard completed experiments.
        write_scores(by_fold, by_month, by_horizon, args.output_dir, args.variants)
    return write_scores(by_fold, by_month, by_horizon, args.output_dir, args.variants)


def run_submit(args, train, test, base, raw, y):
    if test is None:
        raise ValueError('test.csv required for submission')
    n = len(train)
    cutoff = str(pd.Period(train.RENT_APPROVAL_DATE.max(), freq='M') + 1)
    print(f'\n[Submit] {args.submit_variant}, all {n:,} labels, cutoff={cutoff}', flush=True)
    extra = hist_features(raw.iloc[:n], y, raw, cutoff, [args.submit_variant], args)
    X = select_columns(base, extra, args.submit_variant)
    cb = model(args.iterations, args.seed, args.threads, args.depth, args.learning_rate)
    cb.fit(X.iloc[:n], y, cat_features=CAT, verbose=False)
    p = np.asarray(cb.predict(X.iloc[n:]), float)
    p *= market_multiplier(train, y, test.RENT_APPROVAL_DATE.astype(str).to_numpy(), args)
    if len(p) != len(test) or not np.isfinite(p).all():
        raise RuntimeError('Wrong test prediction length or invalid values')
    name = f'submission_{args.submit_variant}_{args.iterations}.csv'
    path = args.output_dir/name
    make_submission(p, len(test)).to_csv(path, index=False)
    check = pd.read_csv(path)
    if check.columns.tolist() != ['Id','Predicted'] or len(check) != len(test) or \
            not np.array_equal(check['Id'].to_numpy(), np.arange(len(test))) or \
            not np.isfinite(check['Predicted'].to_numpy()).all():
        raise RuntimeError('Kaggle CSV verification failed')
    cb.save_model(str(args.output_dir/f'model_{args.submit_variant}_{args.iterations}.cbm'))
    (args.output_dir/'submission_info.json').write_text(json.dumps({
        'model':args.submit_variant, 'iterations':args.iterations,
        'train_rows':n, 'test_rows':len(test), 'history_cutoff_exclusive':cutoff,
        'trend_strength':args.trend_strength, 'pred_mean':float(p.mean()),
        'pred_min':float(p.min()), 'pred_max':float(p.max()),
        'output':str(path), 'columns':['Id','Predicted']
    },indent=2), encoding='utf-8')
    print(f'Verified submission: {path} ({len(test):,} rows)',flush=True)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tr, te, base, raw, y = load(args)
    manifest = {'configuration': 'D1600 + original F1, with incremental G features',
                'training':{'iterations':args.iterations,'depth':args.depth,
                            'learning_rate':args.learning_rate,'seed':args.seed,
                            'threads':args.threads,'trend_strength':args.trend_strength},
                'variants':args.variants, 'folds':get_folds(args),
                'history':'only training fold labels earlier than each training sample; '
                          'all validation history frozen before first validation month',
                'normalization':'flat-type monthly medians within historical training labels only',
                'train_rows':len(tr), 'test_rows':len(te) if te is not None else 0,
                'caveat':'2025-Q1 has been inspected repeatedly; leaderboard tuning risks overfitting.'}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if args.mode in ('validate','both'):
        run_validate(args, tr, base.iloc[:len(tr)], raw.iloc[:len(tr)], y)
    if args.mode in ('submit','both'):
        run_submit(args, tr, te, base, raw, y)


if __name__ == '__main__':
    main()

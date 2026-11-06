#!/usr/bin/env python3
"""F0-F4 rental-history ablation for the CS5228 HDB rental dataset.

F0: exact D1600 configuration (CatBoost + cyclical time + 50% market shift)
F1: + same block, same flat type prior-month rental history
F2: + street- and town-level prior-month history
F3: + recent 3/6/12-month rental comparables and price-spread statistics
F4: + nearby *other blocks* (<=1.2km) with the same flat type

All features are as-of: a row dated M sees transactions dated <M only.
For validation, the reference pool contains TRAIN-FOLD labels only.
No HDB government rent records or test-set rents are read.
"""
from __future__ import annotations

import argparse
from hdb_rent.data import DEFAULT_DATA_DIR, OUTPUT_DIR
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from hdb_rent.base import CAT, features, make_submission, model
from hdb_rent.temporal import adjusted_market_index, cyclic, forecast_index
from hdb_rent.evaluation import score
from hdb_rent.temporal import market_multiplier
from hdb_rent.historical_features import HistoricRents, feature_columns

FOLDS = {
    'strict': [('dev_2023_h2', '2023-07', '2023-12', 'development'),
               ('dev_2024_h1', '2024-01', '2024-06', 'development'),
               ('dev_2024_h2', '2024-07', '2024-12', 'development'),
               ('diagnostic_2025_q1', '2025-01', '2025-03', 'diagnostic')],
    'legacy': [('long18', '2023-10', '2025-03', 'development'),
               ('medium12', '2024-04', '2025-03', 'development'),
               ('short3', '2025-01', '2025-03', 'development')],
}


def args_parse(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--train-file', default='train.csv')
    p.add_argument('--test-file', default='test.csv')
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'history')
    p.add_argument('--mode', choices=['validate', 'submit', 'both'], default='validate')
    p.add_argument('--fold-preset', choices=list(FOLDS), default='strict')
    p.add_argument('--folds', default='', help='Custom validation periods e.g. 2024-07:2024-12,2025-01:2025-03')
    p.add_argument('--variants', default='F0,F1,F2,F3,F4',
                   help='Comma-separated variants to evaluate, e.g. F0,F1,F2,F3')
    p.add_argument('--submit-variant', choices=[f'F{i}' for i in range(5)], default='F0')
    p.add_argument('--iterations', type=int, default=1600)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=0.055)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--trend-strength', type=float, default=0.5)
    p.add_argument('--lookback', type=int, default=12)
    p.add_argument('--decay-months', type=float, default=6.0)
    p.add_argument('--cap-monthly', type=float, default=0.012)
    p.add_argument('--max-neighbors', type=int, default=24)
    p.add_argument('--radius-km', type=float, default=1.2)
    args = p.parse_args(argv)
    args.variants = [x.strip().upper() for x in args.variants.split(',')]
    if not args.variants or len(set(args.variants)) != len(args.variants) or any(x not in [f'F{i}' for i in range(5)] for x in args.variants):
        p.error('--variants must be distinct entries from F0,F1,F2,F3,F4')
    if args.iterations <= 0 or args.depth < 1 or args.threads == 0 or args.max_neighbors <= 0 or args.radius_km <= 0:
        p.error('Invalid model or geography parameters')
    if not (0 <= args.trend_strength <= 2) or args.lookback < 2 or args.decay_months <= 0:
        p.error('Invalid trend parameters')
    return args


def resolve_csv(folder, filename):
    name = folder / filename
    if name.is_file():
        return name
    stems = list(folder.glob(Path(filename).stem + '*csv'))
    if len(stems) == 1:
        print(f'[Input] {name} not found; using {stems[0]}', flush=True)
        return stems[0]
    if len(stems) > 1:
        raise FileNotFoundError(f'{name} is missing; multiple potential matches: {stems}')
    raise FileNotFoundError(f'{name} is missing; set --data-dir to the folder containing CSV files, '
                            f'or use --train-file / --test-file')


def load(args):
    train = pd.read_csv(resolve_csv(args.data_dir, args.train_file))
    required = {'RENT_APPROVAL_DATE', 'MONTHLY_RENT', 'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE'}
    if not required.issubset(train.columns):
        raise ValueError(f'Missing columns in train: {required - set(train.columns)}')
    test = pd.read_csv(resolve_csv(args.data_dir, args.test_file)) if args.mode != 'validate' else None
    if test is not None and 'MONTHLY_RENT' in test.columns:
        raise ValueError('test.csv cannot contain MONTHLY_RENT')
    if test is not None and not set(train.columns).difference(['MONTHLY_RENT']).issubset(test.columns):
        raise ValueError('Train and test feature columns differ')
    y = pd.to_numeric(train['MONTHLY_RENT'], errors='coerce').to_numpy(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Non-positive or missing monthly rent')
    joint = (pd.concat([train.drop(columns=['MONTHLY_RENT']), test], ignore_index=True)
             if test is not None else train.drop(columns=['MONTHLY_RENT']))
    original, raw = features(joint, args.data_dir)
    if len(raw) != len(joint):
        raise RuntimeError('Geo merge broke original row alignment')
    base = cyclic(original.reset_index(drop=True))
    raw = raw.reset_index(drop=True)
    months = pd.Series(raw['RENT_APPROVAL_DATE'].astype(str))
    if not months.str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('Invalid month strings')
    print(f'Input: train={len(train):,}, test={len(test) if test is not None else 0:,}; '
          f'base features={len(base.columns)}, matched coordinates='
          f'{raw.LATITUDE.iloc[:len(train)].notna().mean():.1%}', flush=True)
    return train, test, base, raw, y


def get_folds(args):
    if not args.folds:
        return FOLDS[args.fold_preset]
    result = []
    for i, item in enumerate(args.folds.split(','), start=1):
        try:
            start, end = [x.strip() for x in item.split(':')]
            if start > end or not pd.Series([start, end]).str.fullmatch(r'\d{4}-\d{2}').all():
                raise ValueError()
        except ValueError:
            raise ValueError(f'Bad --folds entry: {item}; expected YYYY-MM:YYYY-MM')
        result.append((f'custom_{i}', start, end, 'development'))
    return result




def assemble_features(base, extra, variant):
    tier = int(variant[1:])
    if tier == 0:
        return base
    needed = feature_columns(tier)
    if any(n not in extra for n in needed):
        raise RuntimeError('History columns missing')
    return pd.concat([base, extra[needed]], axis=1)


def build_history_features(args, reference, reference_y, queries, tier):
    if tier <= 0:
        return pd.DataFrame(index=np.arange(len(queries)))
    begin = time.monotonic()
    index = HistoricRents(reference, reference_y, args.max_neighbors, args.radius_km)
    print(f'  Building F{tier} historical features from {len(reference):,} eligible labelled rows...', flush=True)
    out = index.materialize(queries, tier)
    print(f'  Finished {len(queries):,} feature rows, {len(out.columns)} new columns '
          f'in {time.monotonic()-begin:.1f}s', flush=True)
    return out


def run_validation(args, train, base, raw, y):
    months = train['RENT_APPROVAL_DATE'].astype(str).to_numpy()
    max_tier = max(int(x[1:]) for x in args.variants)
    by_fold, by_horizon, by_month = [], [], []
    folds = get_folds(args)
    for name, start, end, role in folds:
        in_train = months < start
        in_val = (months >= start) & (months <= end)
        if not in_train.any() or not in_val.any():
            raise ValueError(f'{name}: empty historical train or validation: {start}:{end}')
        train_ids, val_ids = np.flatnonzero(in_train), np.flatnonzero(in_val)
        if months[train_ids].max() >= months[val_ids].min():
            raise RuntimeError('Temporal fold overlap')
        print(f'\n[{name}] fit={len(train_ids):,} to {months[train_ids].max()}, '
              f'val={len(val_ids):,} {start}..{end}', flush=True)
        # Only past fold labels populate the index. Querying prior months excludes
        # an observation's own label AND all transactions from the same month.
        selected_rows = np.r_[train_ids, val_ids]
        extra = build_history_features(args, raw.iloc[train_ids], y[train_ids],
                                       raw.iloc[selected_rows], max_tier)
        scale = market_multiplier(train.iloc[train_ids], y[train_ids], months[val_ids], args)
        length_train = len(train_ids)
        horizon = (pd.PeriodIndex(months[val_ids], freq='M').asi8 -
                   pd.Period(months[train_ids].max(), freq='M').ordinal)
        buckets = pd.cut(horizon, bins=[0, 3, 6, 12, 18, 24, 999],
                         labels=['01-03', '04-06', '07-12', '13-18', '19-24', '25+'])
        for variant in args.variants:
            combined = assemble_features(base.iloc[selected_rows].reset_index(drop=True), extra, variant)
            cb = model(args.iterations, args.seed, args.threads, args.depth, args.learning_rate)
            cb.fit(combined.iloc[:length_train], y[train_ids], cat_features=CAT, verbose=False)
            pred = np.asarray(cb.predict(combined.iloc[length_train:]), dtype=float) * scale
            metrics = score(y[val_ids], pred)
            by_fold.append({'fold': name, 'role': role, 'train_end': months[train_ids].max(),
                            'val_start': start, 'val_end': end, 'model': variant,
                            'iterations': args.iterations, 'feature_count': combined.shape[1], **metrics})
            print(f'  {variant}: RMSE {metrics["rmse"]:.3f}, MAE {metrics["mae"]:.3f}, '
                  f'Bias {metrics["bias_pred_minus_actual"]:+.3f}', flush=True)
            for bucket in pd.unique(buckets.dropna()):
                m = np.asarray(buckets == bucket)
                by_horizon.append({'fold': name, 'role': role, 'model': variant,
                                   'horizon': str(bucket), 'iterations': args.iterations,
                                   **score(y[val_ids][m], pred[m])})
            for month in np.unique(months[val_ids]):
                m = months[val_ids] == month
                by_month.append({'fold': name, 'role': role, 'model': variant,
                                 'month': month, **score(y[val_ids][m], pred[m])})
    folds_df = pd.DataFrame(by_fold)
    folds_df.to_csv(args.output_dir / 'scores_by_fold.csv', index=False)
    pd.DataFrame(by_horizon).to_csv(args.output_dir / 'scores_by_horizon.csv', index=False)
    pd.DataFrame(by_month).to_csv(args.output_dir / 'scores_by_month.csv', index=False)
    dev = folds_df.loc[folds_df.role == 'development']
    summary = (dev.groupby('model', as_index=False)
               .agg(mean_rmse=('rmse', 'mean'), worst_rmse=('rmse', 'max'),
                    mean_mae=('mae', 'mean'), mean_bias=('bias_pred_minus_actual', 'mean'),
                    folds=('rmse', 'size')))
    baseline = summary.loc[summary.model == 'F0', 'mean_rmse']
    if len(baseline):
        summary['delta_vs_F0'] = summary['mean_rmse'] - baseline.iloc[0]
    summary = summary.sort_values('mean_rmse')
    summary.to_csv(args.output_dir / 'scores_summary.csv', index=False)
    print('\n== Development fold ranking ==\n' + summary.to_string(index=False), flush=True)
    print('NOTE: The 2025-Q1 diagnostic was previously inspected and is not independent.', flush=True)
    return summary


def run_submission(args, train, test, base, raw, y):
    if test is None:
        raise ValueError('Submission requires test.csv')
    variant = args.submit_variant
    print(f'\n[Submit] {variant}, {args.iterations} iterations, all {len(train):,} labelled rows', flush=True)
    extras = build_history_features(args, raw.iloc[:len(train)], y, raw, int(variant[1:]))
    X = assemble_features(base, extras, variant)
    cb = model(args.iterations, args.seed, args.threads, args.depth, args.learning_rate)
    cb.fit(X.iloc[:len(train)], y, cat_features=CAT, verbose=False)
    pred = np.asarray(cb.predict(X.iloc[len(train):]), dtype=float)
    test_months = test['RENT_APPROVAL_DATE'].astype(str).to_numpy()
    pred *= market_multiplier(train, y, test_months, args)
    submission = make_submission(pred, len(test))
    name = f'submission_{variant}_{args.iterations}.csv'
    output = args.output_dir / name
    submission.to_csv(output, index=False)
    check = pd.read_csv(output)
    if check.columns.tolist() != ['Id', 'Predicted'] or len(check) != len(test) or check.Predicted.isna().any():
        raise RuntimeError('Kaggle submission verification failed')
    cb.save_model(str(args.output_dir / f'catboost_{variant}_{args.iterations}.cbm'))
    (args.output_dir / 'submission_info.json').write_text(json.dumps({
        'variant': variant, 'iterations': args.iterations, 'trend_strength': args.trend_strength,
        'rows': len(test), 'pred_min': float(pred.min()), 'pred_mean': float(pred.mean()),
        'pred_max': float(pred.max()), 'output': str(output),
        'info': 'Only Kaggle training labels were used for historical features; test labels are never read.'
    }, indent=2), encoding='utf-8')
    print(f'Saved and verified: {output} ({len(test):,} Id,Predicted rows)', flush=True)


def main(argv=None):
    args = args_parse(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, base, raw, y = load(args)
    manifest = {'variants': args.variants, 'iterations': args.iterations,
                'depth': args.depth, 'learning_rate': args.learning_rate,
                'seed': args.seed, 'threads': args.threads,
                'trend_strength': args.trend_strength,
                'folds': get_folds(args), 'train_rows': len(train),
                'test_rows': len(test) if test is not None else 0,
                'asof': 'only RENT_APPROVAL_DATE < query month',
                'caveats': ['No target label from current/future month may enter any feature',
                            '2025-Q1 previously seen; not independent',
                            'Official HDB rent records not used',
                            'Leaderboard tuning may overfit public leaderboard']}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.mode in ('validate', 'both'):
        run_validation(args, train, base.iloc[:len(train)], raw.iloc[:len(train)], y)
    if args.mode in ('submit', 'both'):
        run_submission(args, train, test, base, raw, y)


if __name__ == '__main__':
    main()

"""Original CatBoost baseline validation and submission workflow."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from hdb_rent.base import CAT, features, model, make_submission
from hdb_rent.data import DEFAULT_DATA_DIR, OUTPUT_DIR

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--train-file', default='train.csv')
    p.add_argument('--test-file', default='test.csv')
    p.add_argument('--mode', choices=['validate', 'submit', 'both'], default='both')
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'baseline')
    p.add_argument('--iterations', type=int, default=180)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=0.055)
    p.add_argument('--threads', type=int, default=-1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--valid-from', default='2025-01')
    p.add_argument('--valid-to', default='2025-03')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(a.data_dir / a.train_file)
    test = pd.read_csv(a.data_dir / a.test_file) if a.mode != 'validate' else None
    if train['MONTHLY_RENT'].isna().any():
        raise ValueError('Missing training targets')
    combined = pd.concat([train.drop(columns=['MONTHLY_RENT']), test], ignore_index=True) if test is not None else train.drop(columns=['MONTHLY_RENT'])
    X, raw = features(combined, a.data_dir)
    X_train = X.iloc[:len(train)].copy()
    y = train['MONTHLY_RENT'].to_numpy(float)
    metrics = {'train_rows': len(train), 'test_rows': len(test) if test is not None else 0,
               'hdb_geocoding_match': float(raw['LATITUDE'].iloc[:len(train)].notna().mean()),
               'feature_count': len(X.columns), 'features': X.columns.tolist()}
    print(f'Train={len(train):,}, Test={0 if test is None else len(test):,}, features={len(X.columns)}', flush=True)
    print('HDB matched: {:.2%}'.format(metrics['hdb_geocoding_match']), flush=True)
    if a.mode in ('validate', 'both'):
        dates = train['RENT_APPROVAL_DATE'].astype(str)
        val = (dates >= a.valid_from) & (dates <= a.valid_to)
        tr = dates < a.valid_from
        if tr.sum() == 0 or val.sum() == 0:
            raise ValueError('Empty train/validation time split; change --valid-from and --valid-to')
        m = model(a.iterations, a.seed, a.threads, a.depth, a.learning_rate)
        m.fit(X_train.loc[tr], y[tr], cat_features=CAT,
              eval_set=(X_train.loc[val], y[val]), early_stopping_rounds=120, verbose=200)
        pred = m.predict(X_train.loc[val])
        score = float(np.sqrt(mean_squared_error(y[val], pred)))
        metrics.update(validation_rmse=score, validation_rows=int(val.sum()),
                       fit_rows=int(tr.sum()), best_iteration=int(m.best_iteration_),
                       validation_period=f'{a.valid_from} through {a.valid_to}')
        print(f'Future holdout RMSE: {score:.5f}, best iteration: {m.best_iteration_}', flush=True)
        pd.DataFrame({'actual': y[val], 'predicted': pred, 'date': dates[val].to_numpy()}).to_csv(a.output_dir/'validation_predictions.csv', index=False)
        m.save_model(str(a.output_dir/'validation_model.cbm'))
    if a.mode in ('submit', 'both'):
        m = model(a.iterations, a.seed, a.threads, a.depth, a.learning_rate)
        m.fit(X_train, y, cat_features=CAT, verbose=200)
        pred = m.predict(X.iloc[len(train):])
        if len(pred) != len(test) or not np.isfinite(pred).all():
            raise RuntimeError('Invalid submission predictions')
        out = a.output_dir/'submission.csv'
        submission = make_submission(pred, len(test))
        submission.to_csv(out, index=False)
        m.save_model(str(a.output_dir/'submission_model.cbm'))
        metrics['submission_file'] = str(out)
        print(f'Saved {out} ({len(pred):,} predictions)', flush=True)
    with (a.output_dir/'metrics.json').open('w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics:', a.output_dir/'metrics.json', flush=True)

if __name__ == '__main__':
    main()

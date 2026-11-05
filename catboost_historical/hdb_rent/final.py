"""Original F1 training and complete, verified saved-model inference.

Run as ``python -m hdb_rent.final {validate,train,predict,submit} --help``.
The history reference remains the parent training CSV; it is never duplicated.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import warnings

from catboost import CatBoostRegressor
import numpy as np

from .base import CAT, model
from .data import (DEFAULT_DATA_DIR, OUTPUT_DIR, environment, file_hash,
                   implementation_hashes, input_hashes)
from .submission import write_submission
from .temporal import market_multiplier
from experiments.historical import (assemble_features, build_history_features,
                                    load, run_validation)

PIPELINE_VERSION = 1
DEFAULT_CONFIG = dict(iterations=1600, depth=6, learning_rate=0.055, seed=42,
                      threads=4, trend_strength=0.5, lookback=12,
                      decay_months=6.0, cap_monthly=0.012,
                      max_neighbors=24, radius_km=1.2)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def pipeline_args(data_dir, train_file, test_file, config, mode='submit'):
    return SimpleNamespace(data_dir=Path(data_dir), train_file=train_file,
                           test_file=test_file, mode=mode, **config)


def feature_matrix(args):
    """Same joint preprocessing, reference selection and order as original F1."""
    train, test, base, raw, targets = load(args)
    extra = build_history_features(args, raw.iloc[:len(train)], targets, raw, 1)
    return train, test, assemble_features(base, extra, 'F1'), targets


def train_bundle(args):
    start = time.monotonic()
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    inputs = input_hashes(args.data_dir, args.train_file)
    training_test_hash = file_hash(args.data_dir / args.test_file)
    run = pipeline_args(args.data_dir, args.train_file, args.test_file, config)
    train, test, matrix, targets = feature_matrix(run)
    estimator = model(args.iterations, args.seed, args.threads, args.depth, args.learning_rate)
    estimator.fit(matrix.iloc[:len(train)], targets, cat_features=CAT, verbose=False)
    if inputs != input_hashes(args.data_dir, args.train_file):
        raise ValueError('Reference inputs changed during training')
    if training_test_hash != file_hash(args.data_dir / args.test_file):
        raise ValueError('Test input changed during training')
    args.model_dir.mkdir(parents=True, exist_ok=True)
    estimator.save_model(str(args.model_dir / 'model.cbm'))
    manifest = {
        'pipeline_version': PIPELINE_VERSION, 'variant': 'F1', 'configuration': config,
        'environment': environment(), 'implementation_sha256': implementation_hashes(),
        'input_sha256': inputs, 'training_test_sha256': training_test_hash,
        'train_file': args.train_file, 'test_file': args.test_file,
        'train_rows': len(train), 'test_rows': len(test),
        'training_month_min': str(train.RENT_APPROVAL_DATE.min()),
        'training_month_max': str(train.RENT_APPROVAL_DATE.max()),
        'feature_names': matrix.columns.tolist(),
        'feature_dtypes': [str(t) for t in matrix.dtypes], 'categorical_features': CAT,
        'estimator_parameters': estimator.get_all_params(),
        'model_sha256': file_hash(args.model_dir / 'model.cbm'),
        'training_seconds': time.monotonic() - start,
        'history': 'Rebuilt from the exact training CSV; reference month < query month.',
        'trend': 'Recomputed from the exact training labels; applied only after training month max.',
        'inference': 'Requires this package, manifest.json, model.cbm, and the verified parent inputs.',
    }
    save_json(args.model_dir / 'manifest.json', manifest)
    print(f'Saved complete F1 bundle metadata: {args.model_dir} '
          f'({manifest["training_seconds"]:.1f}s)', flush=True)
    return manifest


def predict_bundle(args):
    start = time.monotonic()
    meta = json.loads((args.model_dir / 'manifest.json').read_text(encoding='utf-8'))
    if meta['pipeline_version'] != PIPELINE_VERSION or meta['variant'] != 'F1':
        raise ValueError('Unsupported prediction pipeline')
    if implementation_hashes() != meta['implementation_sha256']:
        raise ValueError('Pipeline code differs from the saved implementation')
    if input_hashes(args.data_dir, meta['train_file']) != meta['input_sha256']:
        raise ValueError('Training history or geographic input checksum mismatch')
    if file_hash(args.model_dir / 'model.cbm') != meta['model_sha256']:
        raise ValueError('Estimator checksum mismatch')
    current = environment()
    if current['packages'] != meta['environment']['packages']:
        raise ValueError('Dependency versions differ from training; use the saved environment versions')
    if current['python'] != meta['environment']['python']:
        warnings.warn('Python version differs from training; numerical equivalence is unverified')
    test_file = args.test_file or meta['test_file']
    test_hash = file_hash(args.data_dir / test_file)
    run = pipeline_args(args.data_dir, meta['train_file'], test_file, meta['configuration'])
    train, test, matrix, targets = feature_matrix(run)
    if len(train) != meta['train_rows'] or matrix.columns.tolist() != meta['feature_names']:
        raise ValueError('Training reference or feature order differs from the saved pipeline')
    if [str(t) for t in matrix.dtypes] != meta['feature_dtypes']:
        raise ValueError('Feature dtypes differ from the saved pipeline')
    estimator = CatBoostRegressor()
    estimator.load_model(str(args.model_dir / 'model.cbm'))
    if estimator.feature_names_ != meta['feature_names']:
        raise ValueError('Estimator feature schema differs from the manifest')
    predictions = np.asarray(estimator.predict(matrix.iloc[len(train):]), dtype=float)
    predictions *= market_multiplier(train, targets,
                                     test.RENT_APPROVAL_DATE.astype(str).to_numpy(), run)
    if test_hash != file_hash(args.data_dir / test_file):
        raise ValueError('Test input changed during inference')
    report = write_submission(predictions, args.output, args.expected_rows)
    report.update(model_sha256=meta['model_sha256'], test_sha256=test_hash,
                  matches_training_test_input=test_hash == meta['training_test_sha256'],
                  row_order='Original CSV order; no test sorting or filtering.',
                  configuration=meta['configuration'], environment=current,
                  inference_seconds=time.monotonic() - start)
    save_json(args.output.with_suffix('.json'), report)
    print(f'Saved and checked {args.output}: {len(predictions):,} predictions '
          f'({report["inference_seconds"]:.1f}s)', flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('validate', 'train', 'predict', 'submit'):
        p = commands.add_parser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
        if name != 'validate':
            p.add_argument('--model-dir', type=Path, default=OUTPUT_DIR / 'f1')
        if name != 'predict':
            p.add_argument('--train-file', default='train.csv')
            p.add_argument('--test-file', default='test.csv')
            for key, value in DEFAULT_CONFIG.items():
                p.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)
        if name in ('predict', 'submit'):
            p.add_argument('--output', type=Path, default=OUTPUT_DIR / 'f1/submission_F1_1600.csv')
            p.add_argument('--expected-rows', type=int, default=50000,
                           help='Competition uses 50000; override only for a small fixture')
        if name == 'predict':
            p.add_argument('--test-file', default=None, help='Defaults to the manifest test filename')
        if name == 'validate':
            p.add_argument('--fold-preset', choices=['strict', 'legacy'], default='strict')
            p.add_argument('--folds', default='', help='Optional YYYY-MM:YYYY-MM periods')
            p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'f1_validation')
    args = parser.parse_args(argv)
    if args.command != 'predict':
        if args.iterations < 1 or args.threads == 0 or args.depth < 1 or args.learning_rate <= 0:
            parser.error('Invalid training configuration')
        if not 0 <= args.trend_strength <= 2 or args.lookback < 2 or args.decay_months <= 0:
            parser.error('Invalid trend configuration')
        if args.cap_monthly <= 0 or args.max_neighbors <= 0 or args.radius_km <= 0:
            parser.error('Invalid trend or geography configuration')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'validate':
        args.mode, args.variants = 'validate', ['F1']
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train, _, base, raw, targets = load(args)
        save_json(args.output_dir / 'run_configuration.json', {
            'configuration': {k: getattr(args, k) for k in DEFAULT_CONFIG},
            'environment': environment(), 'input_sha256': input_hashes(args.data_dir, args.train_file),
            'fold_preset': args.fold_preset, 'custom_folds': args.folds,
        })
        run_validation(args, train, base, raw, targets)
    else:
        if args.command in ('train', 'submit'):
            train_bundle(args)
        if args.command in ('predict', 'submit'):
            predict_bundle(args)


if __name__ == '__main__':
    main()

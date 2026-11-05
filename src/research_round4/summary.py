"""Read-only checkpoint references and fixed-recipe six-origin reporting.

Only the new run's reference copies and summary directory are written.  Loading
model bundles, predicting, selecting settings and training are not part of this
module.  Truth is attached only after all candidate prediction rows agree.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from src.backtest.data import sha256
from src.backtest.execution import atomic_json, read_json, replace_file
from src.backtest.protocol import ROOT, TARGET, horizon, months
from src.backtest.scoring import attach_truth, metrics
from src.research_round2 import run as round2
from src.research_round3 import run as round3
from src.research_round4.common import recipe, model_contract, restore_model

IDENTITY = ['origin', 'row_id', 'RENT_APPROVAL_DATE', 'horizon']
EXPECTED_ROWS = {'six_origins': 178508, 'reused_origins': 82239, 'new_origins': 96269}


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\', '/')


def _below(directory, relative):
    path = (Path(directory) / relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError('checkpoint path escapes its local source directory')
    return path


def _partition(config):
    origins, reused, new = config['origins'], config['reused_origins'], config['new_origins']
    if (len(set(origins)) != len(origins) or len(set(reused)) != len(reused)
            or len(set(new)) != len(new) or set(reused) & set(new)
            or set(reused) | set(new) != set(origins)):
        raise ValueError('origin reuse/new partition is inconsistent')
    if config['models'] != ['B0', 'C3']:
        raise ValueError('summary requires the fixed B0/C3 comparison')
    return origins, reused, new


def _frame(frame, query, origin, name):
    if frame is None or list(frame.columns) != [*IDENTITY, name]:
        raise ValueError('prediction checkpoint has unexpected columns')
    if TARGET in frame or len(frame) != len(query) or frame.row_id.duplicated().any():
        raise ValueError('prediction rows contain truth, duplicates or a row-count mismatch')
    if not (frame.origin.astype(str) == origin).all():
        raise ValueError('prediction origin differs from requested query')
    if not np.array_equal(frame.row_id.to_numpy(), query.row_id.to_numpy()):
        raise ValueError('prediction row identities/order differ from requested query')
    if not np.array_equal(frame.RENT_APPROVAL_DATE.astype(str).to_numpy(), query.RENT_APPROVAL_DATE.astype(str).to_numpy()):
        raise ValueError('prediction dates differ from requested query')
    if not np.array_equal(frame.horizon.to_numpy(), horizon(query, origin)):
        raise ValueError('prediction horizons differ from requested query')
    if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
        raise ValueError('prediction checkpoint has nonfinite values')
    return frame


def _checkpoint_record(source, origin, name, source_pin, manifest_file_hash, root_complete_hash):
    checkpoint = source / 'models' / origin / name / 'complete.json'
    saved = read_json(checkpoint)
    attempt = _below(checkpoint.parent, saved['attempt'])
    predictions = attempt / 'predictions.parquet'
    actual = saved['actual_contract']
    training_backend = actual.get('backend', actual.get('device'))
    if training_backend is None or actual.get('cpu_threads') != 4:
        raise ValueError('source checkpoint lacks its actual device/CPU4 contract')
    prediction = actual.get('prediction', {'prediction_device': training_backend, 'reason': None})
    return {
        'origin': origin, 'model': name,
        'source_root': _relative(source),
        'source_manifest': _relative(source / 'manifest.json'),
        # Manifest pin is canonical JSON excluding its self-hash; the file's
        # byte checksum is separate and must not be confused with that pin.
        'source_manifest_sha256': source_pin,
        'source_manifest_file_sha256': manifest_file_hash,
        'source_root_complete': _relative(source / 'complete.json'),
        'source_root_complete_sha256': root_complete_hash,
        'source_checkpoint': _relative(checkpoint),
        'source_checkpoint_sha256': sha256(checkpoint),
        'source_predictions': _relative(predictions),
        'source_predictions_sha256': sha256(predictions),
        'source_model_files': {
            _relative(attempt / file): digest for file, digest in saved['files'].items()
            if file in ('model.ubj', 'model.cbm', 'bundle.joblib')
        },
        'training_backend': training_backend,
        'prediction_device': prediction.get('prediction_device'),
        'prediction_backend_detail': prediction,
        'cpu_threads': actual['cpu_threads'],
        'selected_iterations': actual['selected_iterations'],
        'actual_contract': actual,
    }


def collect_references(base, config):
    """Verify two old origins under their own frozen recipes, without writes."""
    _, reused, _ = _partition(config)
    references = {origin: {} for origin in reused}
    for name in config['models']:
        spec = config['reuse_sources'][name]
        source = _below(ROOT, spec['root'])
        if not source.is_relative_to((ROOT / 'artifacts/backtest').resolve()):
            raise ValueError('reference source must be inside workspace backtest outputs')
        pin = spec['manifest_sha256']
        frozen = (round2 if name == 'B0' else round3).verify(source, pin)
        if frozen['config'] != recipe(config, name):
            raise ValueError('reused recipe differs from its original frozen model recipe')
        completed = read_json(source / 'complete.json')
        if (completed['manifest_sha256'] != pin or name not in completed['models']
                or not set(reused).issubset(completed['completed_origins'])):
            raise ValueError('reference source root is incomplete or belongs to a different manifest')
        manifest_file_hash, root_complete_hash = sha256(source / 'manifest.json'), sha256(source / 'complete.json')
        for origin in reused:
            history, query = round2.data_at(base, origin, recipe(config, name))
            contract = model_contract(history, query, origin, name, config, pin)
            directory = source / 'models' / origin / name
            frame = _frame(restore_model(directory, contract, name), query, origin, name)
            record = _checkpoint_record(source, origin, name, pin, manifest_file_hash, root_complete_hash)
            record.update(local_copy=f'references/{origin}_{name}.parquet', rows=len(frame), execution='reused_checkpoint', retrained=False)
            references[origin][name] = record
    return references


def _check_reference_sources(record):
    pairs = (
        ('source_manifest', 'source_manifest_file_sha256'),
        ('source_root_complete', 'source_root_complete_sha256'),
        ('source_checkpoint', 'source_checkpoint_sha256'),
        ('source_predictions', 'source_predictions_sha256'),
    )
    for path_key, hash_key in pairs:
        if sha256(_below(ROOT, record[path_key])) != record[hash_key]:
            raise ValueError('read-only reference source changed after verification')


def copy_references(root, references):
    """Copy exact verified prediction bytes into the new run; never overwrite."""
    root = Path(root).resolve()
    if not root.is_relative_to((ROOT / 'artifacts/backtest').resolve()) or not root.name.startswith('round4_'):
        raise ValueError('reference copies require a new workspace backtest directory')
    for models in references.values():
        for record in models.values():
            source = _below(ROOT, record['source_predictions'])
            destination = _below(root, record['local_copy'])
            if root == _below(ROOT, record['source_root']) or destination == source:
                raise ValueError('refusing to copy references onto the old source run')
            _check_reference_sources(record)
            if destination.exists():
                if sha256(destination) != record['source_predictions_sha256']:
                    raise ValueError('existing reference copy has changed')
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.partial')
            shutil.copyfile(source, temporary)
            if sha256(temporary) != record['source_predictions_sha256'] or sha256(source) != record['source_predictions_sha256']:
                raise ValueError('reference changed during byte copy')
            replace_file(temporary, destination)
            if sha256(destination) != record['source_predictions_sha256']:
                raise ValueError('reference destination byte checksum mismatch')


def _deltas(scores):
    reference = scores.loc[scores.model == 'B0'].set_index(['origin', 'horizon'])
    result = scores.loc[scores.model != 'B0'].copy()
    result['B0_rmse'] = [reference.loc[(o, h), 'rmse'] for o, h in zip(result.origin, result.horizon)]
    result['improvement_rmse'] = result.B0_rmse - result.rmse
    result['improvement_pct'] = 100 * result.improvement_rmse / result.B0_rmse
    result['B0_bias'] = [reference.loc[(o, h), 'bias'] for o, h in zip(result.origin, result.horizon)]
    result['bias_difference_vs_B0'] = result.bias - result.B0_bias
    return result


def _horizon_diagnostic(scores, names, weights, scope):
    wanted = weights.index[weights.index >= 1]
    model_scores, missing_sets = [], []
    for name in names:
        month = scores.loc[(scores.origin == 'pooled_instances') & scores.horizon.str.startswith('month_') & (scores.model == name)].copy()
        month['h'] = month.horizon.str.removeprefix('month_').astype(int)
        month = month.set_index('h')
        missing = sorted(set(wanted) - set(month.index))
        missing_sets.append(missing)
        value = None if missing else float(np.sqrt(np.average(month.loc[wanted, 'sse'] / month.loc[wanted, 'n'], weights=weights.loc[wanted])))
        pooled = scores.loc[(scores.origin == 'pooled_instances') & (scores.horizon == 'all') & (scores.model == name)]
        if len(pooled) != 1:
            raise ValueError('diagnostic is missing a unique pooled score')
        model_scores.append({'model': name, 'prediction_instances': int(pooled.iloc[0]['n']),
                             'query_horizon_reweighted_rmse_diagnostic': value})
    if any(missing != missing_sets[0] for missing in missing_sets):
        raise ValueError('candidate horizon support differs')
    missing = missing_sets[0]
    return {'scope': scope, 'complete_test_horizon_support': not missing, 'missing_horizons': missing,
            'excluded_h0': int(weights.get(0, 0)), 'target_positive_horizon_rows': int(weights.loc[wanted].sum()),
            'test_rows_in_missing_horizons': int(weights.reindex(missing).sum()) if missing else 0,
            'renormalized_supported_horizon_score_computed': False,
            'scores': model_scores,
            'limit': 'exact-horizon mixture diagnostic only; not an external validation score. Missing target horizons yield null, without filling or renormalizing.'}


def summarize(root, manifest):
    """Score six fixed origins and separately the four newly trained origins."""
    root = Path(root).resolve()
    config, refs = manifest['config'], manifest['references']
    origins, reused, new = _partition(config)
    if (not root.is_relative_to((ROOT / 'artifacts/backtest').resolve()) or not root.name.startswith('round4_')
            or root in {_below(ROOT, source['root']) for source in config['reuse_sources'].values()}):
        raise ValueError('summary writes must target the new run, never old reference roots')
    if (root / 'complete.json').exists():
        raise ValueError('refusing to rewrite an already completed run summary')
    base = round2.load_base()
    parts, provenance = [], {}
    for origin in origins:
        history, query = round2.data_at(base, origin, config)
        joined, provenance[origin] = None, {}
        for name in config['models']:
            if origin in reused:
                record = dict(refs[origin][name])
                spec = config['reuse_sources'][name]
                if (record['source_root'] != _relative(_below(ROOT, spec['root']))
                        or record['source_manifest_sha256'] != spec['manifest_sha256']):
                    raise ValueError('frozen reused source differs from approved source')
                _check_reference_sources(record)
                local = _below(root, record['local_copy'])
                if sha256(local) != record['source_predictions_sha256']:
                    raise ValueError('local reused prediction copy changed')
                frame = _frame(pd.read_parquet(local), query, origin, name)
                if len(frame) != record['rows']:
                    raise ValueError('reused prediction row receipt differs')
                record['resolved_local_copy'] = _relative(local)
            else:
                contract = model_contract(history, query, origin, name, config, manifest['manifest_sha256'])
                frame = _frame(restore_model(root / 'models' / origin / name, contract, name), query, origin, name)
                record = _checkpoint_record(root, origin, name, manifest['manifest_sha256'], sha256(root / 'manifest.json'), None)
                record.update(rows=len(frame), execution='new_training', retrained=True, local_copy=None,
                              source_root_complete_sha256_limit='root completion receipt is written after summary; no circular self-hash')
            provenance[origin][name] = record
            joined = frame if joined is None else joined.merge(frame, on=IDENTITY, how='inner', validate='one_to_one', sort=False)
        if len(joined) != len(query) or not np.array_equal(joined.row_id.to_numpy(), query.row_id.to_numpy()):
            raise ValueError('candidate row join differs from the common query')
        parts.append(joined)
    predictions = attach_truth(pd.concat(parts, ignore_index=True))
    fresh = predictions.loc[predictions.origin.isin(new)].copy()
    reused_predictions = predictions.loc[predictions.origin.isin(reused)].copy()
    actual_rows = {'six_origins': len(predictions), 'new_origins': len(fresh),
                   'reused_origins': int(predictions.origin.isin(reused).sum())}
    if actual_rows != EXPECTED_ROWS:
        raise ValueError(f'fixed six-window row counts differ: {actual_rows}')
    scores = metrics(predictions, config['models'], config)
    fresh_scores = metrics(fresh, config['models'], config)
    reused_scores = metrics(reused_predictions, config['models'], config)
    test = pd.read_csv(ROOT / 'data/raw/query.csv', usecols=['RENT_APPROVAL_DATE'])
    hs = months(test).asi8 - pd.Period(config['last_label_month'], freq='M').ordinal
    if ((hs < 0) | (hs > config['max_horizon'])).any():
        raise ValueError('unexpected test horizon outside same-month or h1..16')
    weights = pd.Series(hs).value_counts().sort_index()
    if int(weights.get(0, 0)) != 380:
        raise ValueError('same-month test count differs from frozen source expectation')
    diagnostic = {'exact_horizon_counts': {str(h): int(n) for h, n in weights.items()}, 'excluded_h0': 380,
                  'six_origins': _horizon_diagnostic(scores, config['models'], weights, 'six_origins'),
                  'new_origins': _horizon_diagnostic(fresh_scores, config['models'], weights, 'new_origins'),
                  'reused_origins': _horizon_diagnostic(reused_scores, config['models'], weights, 'reused_origins'),
                  'scope': config['scope']}
    if (diagnostic['six_origins']['missing_horizons'] or diagnostic['reused_origins']['missing_horizons']
            or diagnostic['new_origins']['missing_horizons'] != [16]):
        raise ValueError('unexpected horizon support for six-origin/new-four reporting')
    directory = root / 'summary'
    directory.mkdir(exist_ok=True)
    predictions.to_parquet(directory / 'predictions.parquet', index=False)
    scores.to_csv(directory / 'metrics.csv', index=False)
    fresh_scores.to_csv(directory / 'new_origins_metrics.csv', index=False)
    reused_scores.to_csv(directory / 'reused_origins_metrics.csv', index=False)
    _deltas(scores).to_csv(directory / 'deltas.csv', index=False)
    _deltas(fresh_scores).to_csv(directory / 'new_origins_deltas.csv', index=False)
    _deltas(reused_scores).to_csv(directory / 'reused_origins_deltas.csv', index=False)
    atomic_json(directory / 'origin_sources.json', provenance)
    atomic_json(directory / 'references_provenance.json', {'references': refs, 'reused_origins': reused,
                'original_sources_read_only': True, 'reused_models_retrained': False})
    atomic_json(directory / 'test_horizon_diagnostic.json', diagnostic)
    atomic_json(directory / 'summary_counts.json', {'prediction_instances': actual_rows, 'origins': origins,
                'new_origins': new, 'reused_origins': reused, 'models': config['models'],
                'scope': config['scope'], 'weighting': 'overlapping origin-row prediction instances; not independent transactions',
                'new_four_h16_missing_test_rows': diagnostic['new_origins']['test_rows_in_missing_horizons']})
    return directory

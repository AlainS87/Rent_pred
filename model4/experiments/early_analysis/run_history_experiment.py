"""Locked supplemental comparisons for strictly historical target encodings."""
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import runtime
from runtime import ANALYSIS
import json
import time
import hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from run_experiments import SPLITS, PARAMS, SCENARIOS, prepare_fold, configurations, prepare_frame, metric
from history_encoding import construct_history

def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS / 'results'
    lock = json.loads((out / 'history_experiment_lock.json').read_text())
    assert lock['smoothing'] == 20 and lock['configurations'] == ['original_address_plus_history', 'full_aux_plus_history']
    (out / 'history_lock_sha256.txt').write_text(hashlib.sha256((out / 'history_experiment_lock.json').read_bytes()).hexdigest())
    raw = pd.read_pickle(ANALYSIS / 'features/train_features.pkl')
    families = json.loads((ANALYSIS / 'features/feature_families.json').read_text())
    scores = []
    monthly = []
    coverage = []
    predictions = []
    lineage_all = []
    for split in SPLITS:
        source_train = raw.loc[raw.month <= split['cutoff']]
        source_valid = raw.loc[raw.month.between(split['start'], split['end'])]
        hist_train, hist_valid, lineage, audit = construct_history(source_train, source_valid, None, split['cutoff'], smoothing=20)
        metadata = [c for c in hist_train if 'source_month' in c]
        hist_cols = [c for c in hist_train if c not in metadata]
        coverage.extend((dict(split=split['name'], **row) for row in audit['monthly_coverage']))
        for r in lineage:
            r = dict(r)
            r['split'] = split['name']
            lineage_all.append(r)
        (out / f"history_audit_{split['name']}.json").write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
        for scenario in SCENARIOS:
            train, valid, selected, _ = prepare_fold(raw, families, split, scenario)
            for c in hist_cols:
                train[c] = hist_train[c].astype(np.float32)
                valid[c] = hist_valid[c].astype(np.float32)
            config = configurations(selected)
            for base in ['original_address', 'full_aux']:
                name = base + '_plus_history'
                cols = config[base] + hist_cols
                tr, cat = prepare_frame(train[cols])
                va, _ = prepare_frame(valid[cols])
                started = time.time()
                model = CatBoostRegressor(**dict(PARAMS, thread_count=2))
                model.fit(tr, train.target, cat_features=cat)
                pred = model.predict(va)
                scores.append(dict(split=split['name'], role=split['role'], scenario=scenario, configuration=name, model='catboost', features=len(cols), history_features=len(hist_cols), training_rows=len(train), validation_rows=len(valid), seconds=round(time.time() - started, 2), **metric(valid.target, pred)))
                print(json.dumps(scores[-1]), flush=True)
                result = pd.DataFrame({'row_id': valid.row_id.to_numpy(), 'month': valid.month.to_numpy(), 'target': valid.target.to_numpy(), 'prediction': pred, 'configuration': name, 'scenario': scenario, 'split': split['name']})
                for month, p in result.groupby('month'):
                    monthly.append({'split': split['name'], 'scenario': scenario, 'configuration': name, 'month': month, 'rows': len(p), **metric(p.target, p.prediction)})
                if split['role'] == 'confirmation':
                    predictions.append(result)
                    model.save_model(str(out / f'history_confirmation_{base}_{scenario}.cbm'))
                    if base == 'full_aux':
                        snapshot = hist_valid.copy()
                        snapshot.insert(0, 'row_id', valid.row_id)
                        snapshot.insert(1, 'month', valid.month)
                        snapshot.to_csv(ANALYSIS / 'features' / f'history_confirmation_features_{scenario}.csv.gz', index=False, compression='gzip')
                pd.DataFrame(scores).to_csv(out / 'history_scores.csv', index=False)
                pd.DataFrame(monthly).to_csv(out / 'history_monthly_scores.csv', index=False)
        pd.DataFrame(coverage).to_csv(out / 'history_coverage.csv', index=False)
        pd.DataFrame(lineage_all).to_csv(out / 'history_feature_lineage.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / 'history_confirmation_predictions.csv.gz', index=False, compression='gzip')
    print('Strictly historical target-encoding comparisons completed.', flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()

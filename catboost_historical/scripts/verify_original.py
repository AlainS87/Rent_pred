"""Optional full-data feature parity check against the preserved original project.

Requires the local ignored reference directory; ordinary training and unit tests
do not depend on it. Never writes bytecode into the reference directory.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from hdb_rent.data import DEFAULT_DATA_DIR, PROJECT_DIR, environment, file_hash
from hdb_rent.final import DEFAULT_CONFIG, feature_matrix, pipeline_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=PROJECT_DIR / 'cs-5228-2610-project 2')
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--output', type=Path, default=PROJECT_DIR / 'outputs/verification/feature_parity.json')
    args = p.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source.resolve()))
    import historical_ablation as original
    start = time.monotonic()
    settings = original.args_parse(['--data-dir', str(args.source), '--mode', 'submit',
                                    '--submit-variant', 'F1'])
    tr, te, base, raw, targets = original.load(settings)
    extras = original.build_history_features(settings, raw.iloc[:len(tr)], targets, raw, 1)
    old = original.assemble_features(base, extras, 'F1')
    new_tr, new_te, new, new_targets = feature_matrix(
        pipeline_args(args.data_dir, 'train.csv', 'test.csv', DEFAULT_CONFIG))
    pd.testing.assert_frame_equal(tr, new_tr, check_exact=True)
    pd.testing.assert_frame_equal(te, new_te, check_exact=True)
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    np.testing.assert_array_equal(targets, new_targets)
    report = {'train_rows': len(tr), 'test_rows': len(te), 'matrix_rows': len(old),
              'feature_count': old.shape[1], 'feature_names': old.columns.tolist(),
              'features_and_dtypes_exactly_equal': True, 'targets_exactly_equal': True,
              'original_train_and_test_row_order_preserved': True,
              'original_source_sha256': {name: file_hash(args.source / name) for name in
                                        ['train.py', 'ablation_legacy.py', 'historical_ablation.py', 'historical_features.py']},
              'environment': environment(), 'seconds': time.monotonic() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Exercise public CLIs, model reload, metadata integrity and all ablation variants."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd
from hdb_rent.data import PROJECT_DIR
from tests.fixtures import make_data


class SmokeTests(unittest.TestCase):
    def run_cli(self, module, *args, succeeds=True):
        process = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                                 cwd=PROJECT_DIR, text=True, capture_output=True)
        if succeeds:
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        else:
            self.assertNotEqual(process.returncode, 0)
        return process

    def test_saved_model_full_pipeline_and_input_identity(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / 'outputs') as directory:
            root = Path(directory)
            make_data(root / 'data')
            bundle, output = root / 'model', root / 'submission.csv'
            self.run_cli('hdb_rent.final', 'submit', '--data-dir', root / 'data',
                         '--iterations', 12, '--threads', 2, '--model-dir', bundle,
                         '--expected-rows', 20, '--output', output)
            first = pd.read_csv(output)
            self.run_cli('hdb_rent.final', 'predict', '--data-dir', root / 'data',
                         '--model-dir', bundle, '--expected-rows', 20, '--output', root / 'reloaded.csv')
            pd.testing.assert_frame_equal(first, pd.read_csv(root / 'reloaded.csv'), check_exact=True)
            metadata = json.loads((bundle / 'manifest.json').read_text())
            self.assertEqual(len(metadata['feature_names']), 41)
            self.assertEqual(metadata['configuration']['iterations'], 12)
            # A changed history source must never be silently accepted.
            path = root / 'data/train.csv'
            path.write_text(path.read_text() + '\n')
            failure = self.run_cli('hdb_rent.final', 'predict', '--data-dir', root / 'data',
                                   '--model-dir', bundle, '--expected-rows', 20,
                                   '--output', root / 'bad.csv', succeeds=False)
            self.assertIn('checksum mismatch', failure.stderr)
            self.assertFalse((root / 'bad.csv').exists())

    def test_experiment_clis_with_small_inputs(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_DIR / 'outputs') as directory:
            root = Path(directory)
            make_data(root / 'data')
            common = ['--data-dir', root / 'data', '--threads', 2]
            for module in ['experiments.historical', 'experiments.enhanced']:
                out = root / module
                self.run_cli(module, *common, '--iterations', 4, '--folds', '2024-01:2024-03',
                             '--output-dir', out)
                self.assertEqual(len(pd.read_csv(out / 'scores_by_fold.csv')), 5)
            self.run_cli('experiments.temporal', *common, '--iterations', '2,4',
                         '--output-dir', root / 'temporal')
            self.assertEqual(len(pd.read_csv(root / 'temporal/scores_by_fold.csv')), 32)
            self.run_cli('experiments.legacy_temporal', *common, '--iterations', 2,
                         '--long-iterations', 4, '--folds', '2024-01:2024-03',
                         '--output-dir', root / 'legacy')
            self.run_cli('experiments.baseline', *common, '--iterations', 4,
                         '--valid-from', '2024-01', '--valid-to', '2024-03',
                         '--mode', 'both', '--output-dir', root / 'baseline')

    @classmethod
    def setUpClass(cls):
        (PROJECT_DIR / 'outputs').mkdir(exist_ok=True)

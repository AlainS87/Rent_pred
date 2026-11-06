"""Synthetic transformation and metric checks; no estimator is fitted."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from experiments.rolling_local.config import read_config
from experiments.rolling_local.data import address_keys, enforce_phase_info, rank_for_phase
from experiments.rolling_local.metrics import _metrics


def temporary_run():
    # Keep file-writing fixtures in the writable checkout, including on managed
    # Windows environments whose redirected system temp has different ACLs.
    return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)


class PureContracts(unittest.TestCase):
    def test_rmse_is_original_equal_weight_unit(self):
        result = _metrics(np.array([1., 2.]), np.array([1., 4.]))
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.))
        self.assertEqual(result["bias_pred_minus_truth"], 1.)

    def test_rank_uses_only_history_and_unknown_median_rank(self):
        train = pd.DataFrame({"FLAT_MODEL": ["a", "a", "b", "c"],
                              "MONTHLY_RENT": [10., 30., 5., 40.], "month": ["2020-01"]*4})
        apply = pd.DataFrame({"FLAT_MODEL": ["new", "a"], "MONTHLY_RENT": [9999., 0.]})
        with temporary_run() as root:
            fitted, transformed = rank_for_phase(train, apply, Path(root))
        self.assertEqual(fitted.flat_model_ord.tolist(), [1., 1., 0., 2.])
        self.assertEqual(transformed.flat_model_ord.tolist(), [1., 1.])

    def test_macro_freezes_future_values_without_double_lag(self):
        columns = ["coe_c_premium_lag1", "bank_basket_adj_close_lag1", "property_basket_adj_close_lag1"]
        train = pd.DataFrame({"RENT_APPROVAL_DATE": ["2020-01", "2020-02"], **{c: [3., 5.] for c in columns}})
        apply = pd.DataFrame({"RENT_APPROVAL_DATE": ["2020-03", "2020-04"], **{c: [999., 888.] for c in columns}})
        with temporary_run() as root:
            unchanged, transformed = enforce_phase_info(train, apply, "2020-02", Path(root))
            audit = json.loads((Path(root)/"macro_freeze.json").read_text())
        pd.testing.assert_frame_equal(unchanged, train)
        for column in columns:
            self.assertEqual(transformed[column].tolist(), [5., 5.])
            self.assertEqual(audit["columns"][column]["reference_month"], "2020-01")

    def test_inconsistent_monthly_macro_is_rejected(self):
        columns = ["coe_c_premium_lag1", "bank_basket_adj_close_lag1", "property_basket_adj_close_lag1"]
        train = pd.DataFrame({"RENT_APPROVAL_DATE": ["2020-02", "2020-02"], **{c: [3., 5.] for c in columns}})
        apply = pd.DataFrame({"RENT_APPROVAL_DATE": ["2020-03"]})
        with temporary_run() as root, self.assertRaises(ValueError):
            enforce_phase_info(train, apply, "2020-02", Path(root))

    def test_address_key_requires_unambiguous_components(self):
        frame = pd.DataFrame({"TOWN": [" North "], "street_key": [" Main Road "], "block_key": ["01"]})
        self.assertEqual(address_keys(frame).iloc[0], "north|main road|01")
        frame.loc[0, "block_key"] = "a|b"
        with self.assertRaises(ValueError):
            address_keys(frame)

    def test_example_configuration_validation_is_data_independent(self):
        root = Path(__file__).resolve().parents[2]
        config = read_config(root / "experiments/rolling_local/example_config.json")
        self.assertEqual(config["origins"], ["2023-05", "2022-03", "2022-10"])
        self.assertEqual(config["forecast_months"], 16)

    def test_local_fit_calls_are_not_top_level_statements(self):
        root = Path(__file__).resolve().parents[2]
        for package in (root/"experiments/rolling_local", root/"experiments/rolling_local/_core"):
            for path in package.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If)):
                        continue
                    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                        name = ast.unparse(call.func).split(".")[-1]
                        self.assertNotIn(name, {"fit", "fit_model", "predict", "predict_model", "run_local"}, str(path))


if __name__ == "__main__":
    unittest.main()

"""Run with: python -m unittest -v test_evaluate_official.py"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics.official_matching import normal_flat_type, normal_street, standardise_table


class OfficialMatchingTests(unittest.TestCase):
    def test_normalisation(self):
        left = normal_street(pd.Series(["Ang Mo Kio Avenue 4", "Lorong 1 Toa Payoh", "Yishun Ring Road"]))
        right = normal_street(pd.Series(["ANG MO KIO AVE 4", "LOR 1 TOA PAYOH", "YISHUN RING RD"]))
        self.assertEqual(left.tolist(), right.tolist())
        self.assertEqual(normal_flat_type(pd.Series(["4 room", "4-room", "4-ROOM"])).nunique(), 1)

    def test_end_to_end_command_with_duplicate_prices(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            dates = ["2025-03", "2025-04", "2025-04", "2025-04", "2025-04", "2025-04"]
            names = ["Ang Mo Kio", "Punggol", "Jurong West", "Yishun", "Geylang", "Yishun"]
            block = ["105", "604A", "112", "101", "88", "101"]
            streets = ["Ang Mo Kio Avenue 4", "Punggol Road", "Jurong West Street 42", "Yishun Ring Road", "Geylang East Avenue 1", "Yishun Ring Road"]
            flats = ["4 room", "4-room", "3-room", "3-room", "3-room", "3-room"]
            df = pd.DataFrame({"RENT_APPROVAL_DATE": dates, "TOWN": names, "BLOCK": block,
                               "STREET": streets, "FLAT_TYPE": flats})
            preds = pd.DataFrame({"MONTHLY_RENT": [2190, 3000, 2850, 2800, 3100, 3300]})
            # Official record for March includes a known *train* transaction ($2000),
            # so the test-only residual after subtraction should be $2200.
            official = pd.DataFrame({
                "rent_approval_date": ["2025-03", "2025-03", "2025-04", "2025-04", "2025-04", "2025-04", "2025-04"],
                "town": ["ANG MO KIO", "ANG MO KIO", "PUNGGOL", "JURONG WEST", "JURONG WEST", "YISHUN", "YISHUN"],
                "block": ["105", "105", "604A", "112", "112", "101", "101"],
                "street_name": ["ANG MO KIO AVE 4", "ANG MO KIO AVE 4", "PUNGGOL RD", "JURONG WEST ST 42", "JURONG WEST ST 42", "YISHUN RING RD", "YISHUN RING RD"],
                "flat_type": ["4-ROOM", "4-ROOM", "4-ROOM", "3-ROOM", "3-ROOM", "3-ROOM", "3-ROOM"],
                "monthly_rent": [2000, 2200, 3100, 2900, 2900, 2700, 3400],
            })
            train = pd.DataFrame({"RENT_APPROVAL_DATE": ["2025-03"], "TOWN": ["ang mo kio"],
                                  "BLOCK": ["105"], "STREET": ["ang mo kio avenue 4"],
                                  "FLAT_TYPE": ["4-room"], "MONTHLY_RENT": [2000]})
            df.to_csv(p / "test.csv", index=False)
            preds.to_csv(p / "pred.csv", index=False)
            official.to_csv(p / "official.csv", index=False)
            train.to_csv(p / "train.csv", index=False)
            cmd = [sys.executable, "-m", "diagnostics.official_matching",
                   "--test-file", str(p / "test.csv"), "--pred-file", str(p / "pred.csv"),
                   "--official-csv", str(p / "official.csv"), "--train-file", str(p / "train.csv"),
                   "--output-dir", str(p / "out")]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            details = pd.read_csv(p / "out" / "official_match_details.csv")
            with open(p / "out" / "official_metrics.json") as f:
                report = json.load(f)
            self.assertEqual(len(details), 6)
            self.assertEqual(details.match_type.tolist(), [
                "single_candidate_pair", "single_candidate_pair", "one_possible_rent",
                "ambiguous_multiple_rents", "unmatched", "ambiguous_multiple_rents"])
            self.assertEqual(report["training_subtraction"]["train_rows_subtracted"], 1)
            self.assertEqual(report["single_candidate_pair_vs_official"]["rows"], 2)
            expected_rmse = np.sqrt((10 ** 2 + 100 ** 2) / 2)
            self.assertAlmostEqual(report["single_candidate_pair_vs_official"]["rmse"], expected_rmse)
            self.assertEqual(report["match_counts"]["ambiguous_multiple_rents"], 2)
            self.assertEqual(report["match_counts"]["unmatched"], 1)
            self.assertEqual(details.loc[3, "official_min"], 2700)
            self.assertEqual(details.loc[3, "official_max"], 3400)
            self.assertTrue((p / "out" / "official_by_month.csv").exists())

    def test_prediction_length_rejected(self):
        from diagnostics.official_matching import load_predictions
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            pd.DataFrame({"MONTHLY_RENT": [3000]}).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "Prediction rows"):
                load_predictions(p, 2)

    def test_unknown_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing 'date'"):
            standardise_table(pd.DataFrame({"x": [1]}))


if __name__ == "__main__":
    unittest.main()

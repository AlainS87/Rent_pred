import unittest
import numpy as np
import pandas as pd
from hdb_rent.historical_features import HistoricRents, feature_columns
from hdb_rent.enhanced_history import AnchoredRents


class HistoryTests(unittest.TestCase):
    def reference(self):
        return pd.DataFrame({'RENT_APPROVAL_DATE': ['2023-01', '2023-01', '2023-02', '2023-03'],
                             'TOWN': ['town'] * 4, 'BLOCK': ['1'] * 4, 'STREET': ['street'] * 4,
                             'FLAT_TYPE': ['4-room'] * 4, 'LATITUDE': [1.3] * 4,
                             'LONGITUDE': [103.8] * 4})

    def test_same_month_and_future_labels_excluded(self):
        ref = self.reference()
        first = HistoricRents(ref, np.array([100., 300., 900., 10000.])).materialize(ref, 1)
        second = HistoricRents(ref, np.array([100., 300., 99999., 999999.])).materialize(ref, 1)
        self.assertEqual(first.columns.tolist(), feature_columns(1))
        self.assertEqual(first.hr_block_n.tolist(), [0, 0, 2, 3])
        self.assertTrue(first.iloc[:2, 1:].isna().all().all())
        self.assertEqual(first.iloc[2].hr_block_median, 200.)
        self.assertEqual(first.iloc[2].hr_block_std, 100.)
        self.assertEqual(first.iloc[2].hr_block_age_months, 1.)
        pd.testing.assert_frame_equal(first.iloc[:3], second.iloc[:3], check_exact=True)

    def test_fold_reference_is_frozen_for_validation(self):
        ref = self.reference().iloc[:2]
        query = self.reference().iloc[2:]
        out = HistoricRents(ref, np.array([100., 300.])).materialize(query, 1)
        self.assertEqual(out.hr_block_n.tolist(), [2, 2])
        self.assertEqual(out.hr_block_median.tolist(), [200., 200.])
        self.assertEqual(out.hr_block_age_months.tolist(), [1., 2.])

    def test_enhanced_cutoff_and_unknown_block(self):
        ref = self.reference().iloc[:2]
        query = self.reference().iloc[2:].copy()
        index = AnchoredRents(ref, np.array([100., 300.]))
        out = index.materialize(query, '2023-02', 4)
        np.testing.assert_array_equal(out.iloc[0], out.iloc[1])
        query['BLOCK'] = 'new block'
        basic = index.h.materialize(query, 1)
        self.assertTrue((basic.hr_block_n == 0).all())
        self.assertTrue(basic.hr_block_median.isna().all())
        with self.assertRaisesRegex(ValueError, 'Reference contains'):
            AnchoredRents(self.reference(), np.array([100., 300., 900., 10000.])).materialize(query, '2023-02', 1)

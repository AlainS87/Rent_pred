import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from hdb_rent.temporal import adjusted_market_index, cyclic, forecast_index, market_multiplier


class TemporalTests(unittest.TestCase):
    def test_boundary_month_is_unadjusted(self):
        ref = pd.DataFrame({'RENT_APPROVAL_DATE': pd.period_range('2023-01', '2023-12', freq='M').astype(str),
                            'TOWN': ['town'] * 12, 'FLAT_TYPE': ['4 room'] * 12})
        y = np.linspace(1000., 1400., 12)
        args = SimpleNamespace(trend_strength=.5, lookback=12, decay_months=6., cap_monthly=.012)
        months = ['2023-12', '2024-01', '2024-06', '2023-11']
        values = market_multiplier(ref, y, months, args)
        self.assertEqual(values[0], 1.)
        self.assertEqual(values[-1], 1.)
        self.assertGreater(values[1], 1.)
        index = adjusted_market_index(ref, y)
        np.testing.assert_array_equal(values[1:3], np.exp(.5 * forecast_index(index, months[1:3])))
        with self.assertRaisesRegex(ValueError, 'strictly after'):
            forecast_index(index, ['2023-12'])
        args.trend_strength = 0.
        np.testing.assert_array_equal(market_multiplier(ref, y, months, args), np.ones(4))

    def test_seasonality_column_order_and_phase(self):
        frame = pd.DataFrame({'existing': [7, 8], 'month': [1, 4]})
        out = cyclic(frame)
        self.assertEqual(out.columns.tolist(), ['existing', 'month', 'month_sin', 'month_cos'])
        self.assertAlmostEqual(out.month_sin.iloc[0], 0.)
        self.assertAlmostEqual(out.month_cos.iloc[0], 1.)
        self.assertAlmostEqual(out.month_sin.iloc[1], 1.)

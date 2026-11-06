import unittest
import numpy as np
import pandas as pd
from hdb_rent.submission import make_submission, validate_submission


class SubmissionTests(unittest.TestCase):
    def test_competition_shape(self):
        result = make_submission(np.full(50000, 3000.), 50000)
        self.assertEqual(validate_submission(result)['prediction_count'], 50000)

    def test_bad_schema_order_length_or_values_rejected(self):
        valid = make_submission([3000., 3100.], 2)
        invalid = [valid.rename(columns={'Id': 'id'}), valid.iloc[::-1], valid.iloc[:1],
                   valid.assign(Predicted=[np.nan, 3000.]), valid.assign(Predicted=[np.inf, 3000.])]
        for frame in invalid:
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    validate_submission(frame, 2)

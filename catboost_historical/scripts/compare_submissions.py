"""Validate two submissions and compare predictions in original row order."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hdb_rent.submission import validate_submission
from hdb_rent.data import environment, file_hash


def compare(reference, candidate, expected_rows=50000):
    left, right = pd.read_csv(reference), pd.read_csv(candidate)
    validate_submission(left, expected_rows)
    validate_submission(right, expected_rows)
    delta = np.abs(left.Predicted.to_numpy() - right.Predicted.to_numpy())
    return {'prediction_count': len(left),
            'max_absolute_difference': float(delta.max()),
            'mean_absolute_difference': float(delta.mean()),
            'ids_identical': bool(left.Id.equals(right.Id)),
            'reference_sha256': file_hash(reference), 'candidate_sha256': file_hash(candidate),
            'environment': environment(),
            'order_note': 'IDs are validated in CSV order. Input-test identity is recorded by final inference.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--expected-rows', type=int, default=50000)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = compare(a.reference, a.candidate, a.expected_rows)
    content = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(content, encoding='utf-8')
    print(content, end='')


if __name__ == '__main__':
    main()

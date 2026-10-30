"""In-model preprocessing for ridge regression.

This layer selects the genome columns and turns nominal fields into numbers.
"""

from src.pipeline.preprocess import TabularPreprocessor, assert_model_matrix


def build_preprocessor(spec: dict) -> TabularPreprocessor:
    return TabularPreprocessor(spec)


def fit_preprocessor(spec: dict, frame, target, features: list[str]) -> TabularPreprocessor:
    preparer = build_preprocessor(spec)
    preparer.fit(frame, target, features)
    return preparer


def transform_frame(preparer: TabularPreprocessor, frame):
    matrix = preparer.transform(frame)
    return assert_model_matrix(matrix, len(frame), preparer.output_columns_)

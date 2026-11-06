"""In-model preprocessing for XGBoost.

Nominal columns stay pandas categories so a tree can partition the labels.
Numeric columns stay imputed and unscaled. The fixed date, block, and
furnished overrides still apply.
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
    if preparer.spec["categorical"] == "native":
        return matrix
    return assert_model_matrix(matrix, len(frame), preparer.output_columns_)

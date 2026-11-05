"""Impute, scale, and encode a genome's preprocess gene.

String cleanup is fixed: strip and lower case, and turn hyphens in
``FLAT_TYPE`` into spaces. It is not a gene.

Three columns also ignore the categorical gene. ``RENT_APPROVAL_DATE``
becomes months since 2021-01. ``BLOCK`` becomes the shrunk fit-month mean
rent of ``TOWN|BLOCK|STREET``. ``FURNISHED`` becomes 1 or 0, and is omitted
when the fit months have only one value.
"""

import numpy as np
import pandas as pd

from src.pipeline.rules import PREPROCESS_KNOBS

BIN_COUNT = 10
TARGET_STRENGTH = 20
DATE_ORIGIN_YEAR = 2021
DATE_ORIGIN_MONTH = 1
COLUMN_OVERRIDES = "date-month,block-building-target,furnished-binary"

CATEGORICAL_NAMES = {
    "TOWN",
    "STREET",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "postal_code",
    "subzone",
    "planning_area",
    "region",
}


class TabularPreprocessor:
    """Fit on the training months and apply the same map to later months."""

    def __init__(self, spec: dict):
        self.spec = {
            "numeric_impute": spec["numeric_impute"],
            "scale": spec["scale"],
            "categorical": spec["categorical"],
            "rare_min_count": int(spec["rare_min_count"]),
        }
        for knob, choices in PREPROCESS_KNOBS.items():
            allowed = set(choices)
            if knob == "categorical":
                allowed.add("native")
            if self.spec[knob] not in allowed:
                raise ValueError(f"preprocess {knob}={self.spec[knob]} is not a legal value")
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.numeric_fill_: dict[str, float] = {}
        self.numeric_center_: dict[str, float] = {}
        self.numeric_spread_: dict[str, float] = {}
        self.levels_: dict[str, list[str]] = {}
        self.output_columns_: list[str] = []
        self.selected_columns_: list[str] = []
        self.target_values_: dict[str, dict[str, float]] = {}
        self.target_default_: float = 0.0
        self.bin_values_: dict[str, dict[str, int]] = {}
        self.unseen_bin_: int = BIN_COUNT
        self.include_building_: bool = False
        self.building_values_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, target, features: list[str]) -> "TabularPreprocessor":
        """Select ``features`` from ``frame`` and fit maps on those columns only.

        ``target`` is the fit-month rent. Target encoding uses it. Search,
        confirm, and test rents are never passed here.
        """
        if not features:
            raise ValueError("the preprocessor needs at least one feature")
        required = list(features)
        if "BLOCK" in required:
            required.extend(name for name in ("TOWN", "STREET") if name not in required)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"preprocessor is missing columns: {missing}")
        rents = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(rents) != len(frame):
            raise ValueError("target length does not match the frame")
        self.selected_columns_ = list(features)
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        self.include_building_ = False
        self.building_values_ = {}
        for column in self.selected_columns_:
            if column == "FURNISHED":
                bits = _furnished_bit(frame[column])
                if np.unique(bits[np.isfinite(bits)]).size >= 2:
                    self.numeric_columns_.append(column)
                continue
            if column == "RENT_APPROVAL_DATE":
                self.numeric_columns_.append(column)
                continue
            if column == "BLOCK":
                self.include_building_ = True
                continue
            if _is_categorical(column, frame[column]):
                self.categorical_columns_.append(column)
            else:
                self.numeric_columns_.append(column)
        for column in self.numeric_columns_:
            values = _model_numeric(frame, column)
            finite = values[np.isfinite(values)]
            self.numeric_fill_[column] = _fill_value(finite, self.spec["numeric_impute"])
            filled = np.where(np.isfinite(values), values, self.numeric_fill_[column])
            center, spread = _scale_params(filled, self.spec["scale"])
            self.numeric_center_[column] = center
            self.numeric_spread_[column] = spread
        minimum = self.spec["rare_min_count"]
        finite_rent = rents[np.isfinite(rents)]
        self.target_default_ = float(np.mean(finite_rent)) if finite_rent.size else 0.0
        self.target_values_ = {}
        self.bin_values_ = {}
        if self.include_building_:
            building = _building_key(frame)
            self.building_values_ = _target_map(
                building, rents, _ranked_levels(building), self.target_default_
            )
        for column in self.categorical_columns_:
            text = _category_text(frame[column], column)
            ranked = _ranked_levels(text)
            if self.spec["categorical"] == "native":
                self.levels_[column] = [level for level, _count in ranked]
                continue
            kept = [level for level, count in ranked if count >= minimum]
            self.levels_[column] = kept
            if self.spec["categorical"] == "target":
                self.target_values_[column] = _target_map(text, rents, ranked, self.target_default_)
            if self.spec["categorical"] == "bin":
                self.bin_values_[column] = _frequency_bins(ranked)
        self.output_columns_ = _output_names(self)
        return self

    def transform(self, frame: pd.DataFrame):
        required = self._required_columns()
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"transform is missing columns: {missing}")
        if self.spec["categorical"] == "native":
            return self._transform_native(frame)
        pieces: list[np.ndarray] = []
        for column in self.numeric_columns_:
            values = _model_numeric(frame, column)
            filled = np.where(np.isfinite(values), values, self.numeric_fill_[column])
            pieces.append(_apply_scale(filled, self.spec["scale"], self.numeric_center_[column], self.numeric_spread_[column]))
        for column in self.categorical_columns_:
            text = _category_text(frame[column], column)
            if self.spec["categorical"] == "target":
                pieces.append(_map_levels(text, self.target_values_[column], self.target_default_))
                continue
            if self.spec["categorical"] == "bin":
                pieces.append(_map_levels(text, self.bin_values_[column], float(self.unseen_bin_)))
                continue
            levels = self.levels_[column]
            codes = pd.Categorical(text, categories=levels).codes
            if self.spec["categorical"] == "ordinal":
                other = float(len(levels))
                encoded = np.where(codes < 0, other, codes).astype(np.float64)
                pieces.append(encoded)
                continue
            if not levels:
                pieces.append(np.ones(len(frame), dtype=np.float64))
                continue
            for index in range(len(levels)):
                pieces.append((codes == index).astype(np.float64))
            pieces.append((codes < 0).astype(np.float64))
        if getattr(self, "include_building_", False):
            pieces.append(_map_levels(_building_key(frame), self.building_values_, self.target_default_))
        if not pieces:
            raise ValueError("the preprocessor has no columns")
        matrix = np.column_stack(pieces)
        return assert_model_matrix(matrix, len(frame), self.output_columns_)

    def _transform_native(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Numeric columns stay floats. Nominal columns stay pandas categories."""
        data: dict[str, pd.Series] = {}
        for column in self.numeric_columns_:
            values = _model_numeric(frame, column)
            filled = np.where(np.isfinite(values), values, self.numeric_fill_[column])
            scaled = _apply_scale(filled, self.spec["scale"], self.numeric_center_[column], self.numeric_spread_[column])
            data[column] = pd.Series(scaled, index=frame.index, dtype="float64")
        for column in self.categorical_columns_:
            text = _category_text(frame[column], column)
            data[column] = pd.Series(
                pd.Categorical(text, categories=self.levels_[column]),
                index=frame.index,
            )
        if getattr(self, "include_building_", False):
            data["BLOCK__building"] = pd.Series(
                _map_levels(_building_key(frame), self.building_values_, self.target_default_),
                index=frame.index,
                dtype="float64",
            )
        table = pd.DataFrame(data, columns=self.output_columns_)
        if len(table) != len(frame):
            raise ValueError(f"native matrix has {len(table)} rows, expected {len(frame)}")
        for column in self.numeric_columns_:
            if not np.isfinite(table[column].to_numpy(dtype=np.float64)).all():
                raise ValueError(f"numeric column {column} has non-finite values")
        if "BLOCK__building" in table.columns and not np.isfinite(table["BLOCK__building"].to_numpy(dtype=np.float64)).all():
            raise ValueError("BLOCK__building has non-finite values")
        return table

    def _required_columns(self) -> list[str]:
        selected = getattr(self, "selected_columns_", None)
        if selected:
            required = list(selected)
        else:
            required = list(self.numeric_columns_) + list(self.categorical_columns_)
        if getattr(self, "include_building_", False):
            required.extend(name for name in ("TOWN", "STREET") if name not in required)
        return required


def assert_model_matrix(matrix, n_rows: int, output_columns: list[str] | None = None) -> np.ndarray:
    """Require the matrix a model is allowed to train or score on."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"model matrix must be 2-D, got shape {values.shape}")
    if values.shape[0] != n_rows:
        raise ValueError(f"model matrix has {values.shape[0]} rows, expected {n_rows}")
    if output_columns is not None and values.shape[1] != len(output_columns):
        raise ValueError(
            f"model matrix has {values.shape[1]} columns, expected {len(output_columns)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("model matrix has non-finite values")
    return values


def assert_fit_inputs(features, target) -> tuple[np.ndarray, np.ndarray]:
    """Refuse a fit whose matrix and target lengths differ."""
    rents = np.asarray(target, dtype=np.float64).reshape(-1)
    matrix = assert_model_matrix(features, len(rents))
    return matrix, rents


def assert_prediction(prediction, n_rows: int) -> np.ndarray:
    """Require a 1-D finite prediction with one value per row."""
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if values.shape != (n_rows,):
        raise ValueError(f"prediction has shape {values.shape}, expected {(n_rows,)}")
    if not np.isfinite(values).all():
        raise ValueError("prediction has non-finite values")
    return values


def _ranked_levels(text: pd.Series) -> list[tuple[str, int]]:
    counts = text.value_counts()
    return sorted(((str(level), int(count)) for level, count in counts.items()), key=lambda item: (-item[1], item[0]))


def _target_map(text: pd.Series, rents: np.ndarray, ranked: list[tuple[str, int]], global_mean: float) -> dict[str, float]:
    mapping = {}
    labels = text.astype("string").to_numpy()
    for level, _count in ranked:
        values = rents[(labels == level) & np.isfinite(rents)]
        count = int(values.size)
        if count == 0:
            mapping[level] = global_mean
            continue
        level_mean = float(np.mean(values))
        mapping[level] = (count * level_mean + TARGET_STRENGTH * global_mean) / (count + TARGET_STRENGTH)
    return mapping


def _frequency_bins(ranked: list[tuple[str, int]]) -> dict[str, int]:
    total = sum(count for _level, count in ranked)
    mapping = {}
    running = 0
    for level, count in ranked:
        fraction = 0.0 if total == 0 else running / total
        mapping[level] = min(BIN_COUNT - 1, int(fraction * BIN_COUNT))
        running += count
    return mapping


def _map_levels(text: pd.Series, mapping: dict, default: float) -> np.ndarray:
    encoded = text.astype("string").map(mapping)
    return encoded.fillna(default).to_numpy(dtype=np.float64)


def _is_categorical(column: str, series: pd.Series) -> bool:
    if column in CATEGORICAL_NAMES or column.endswith(("_code", "_name", "_line")):
        return True
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _model_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column == "RENT_APPROVAL_DATE":
        return _approval_months(frame[column])
    if column == "FURNISHED":
        return _furnished_bit(frame[column])
    return _numeric(frame[column])


def _approval_months(series: pd.Series) -> np.ndarray:
    """Months since 2021-01. ``2024-09`` is 44 and ``2026-07`` is 66."""
    text = series.astype("string").str.strip()
    year = pd.to_numeric(text.str.slice(0, 4), errors="coerce")
    month = pd.to_numeric(text.str.slice(5, 7), errors="coerce")
    values = ((year - DATE_ORIGIN_YEAR) * 12 + (month - DATE_ORIGIN_MONTH)).to_numpy(dtype=np.float64)
    bad = (year.isna() | month.isna() | (month < 1) | (month > 12)).to_numpy()
    values[bad] = np.nan
    return values


def _furnished_bit(series: pd.Series) -> np.ndarray:
    text = series.astype("string").str.strip().str.lower()
    bits = np.full(len(series), np.nan, dtype=np.float64)
    bits[text.eq("yes").to_numpy()] = 1.0
    bits[text.eq("no").to_numpy()] = 0.0
    return bits


def _building_key(frame: pd.DataFrame) -> pd.Series:
    town = _category_text(frame["TOWN"], "TOWN")
    block = _category_text(frame["BLOCK"], "BLOCK")
    street = _category_text(frame["STREET"], "STREET")
    return town + "|" + block + "|" + street


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _fill_value(finite: np.ndarray, method: str) -> float:
    if method == "zero" or finite.size == 0:
        return 0.0
    if method == "mean":
        return float(np.mean(finite))
    if method == "median":
        return float(np.median(finite))
    raise ValueError(f"unknown impute method: {method}")


def _scale_params(filled: np.ndarray, method: str) -> tuple[float, float]:
    if method == "none":
        return 0.0, 1.0
    if method == "standard":
        center = float(np.mean(filled))
        spread = float(np.std(filled))
        return center, spread
    if method == "minmax":
        low = float(np.min(filled))
        spread = float(np.max(filled) - low)
        return low, spread
    raise ValueError(f"unknown scale method: {method}")


def _apply_scale(filled: np.ndarray, method: str, center: float, spread: float) -> np.ndarray:
    if method == "none":
        return filled.astype(np.float64)
    if spread == 0:
        return np.zeros_like(filled, dtype=np.float64)
    return ((filled - center) / spread).astype(np.float64)


def _category_text(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        number = pd.to_numeric(series, errors="coerce")
        text = number.map(_number_token).astype("string")
    else:
        text = series.astype("string").str.strip().str.lower()
    if column == "FLAT_TYPE":
        text = text.str.replace("-", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    return text.fillna("__MISSING__")


def _number_token(value: float) -> str:
    if pd.isna(value):
        return pd.NA
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), "g")


def _output_names(preprocessor: TabularPreprocessor) -> list[str]:
    names = list(preprocessor.numeric_columns_)
    for column in preprocessor.categorical_columns_:
        levels = preprocessor.levels_[column]
        if preprocessor.spec["categorical"] == "native":
            names.append(column)
            continue
        if preprocessor.spec["categorical"] == "ordinal":
            names.append(f"{column}__ordinal")
            continue
        if preprocessor.spec["categorical"] == "target":
            names.append(f"{column}__target")
            continue
        if preprocessor.spec["categorical"] == "bin":
            names.append(f"{column}__bin")
            continue
        if not levels:
            names.append(f"{column}=__OTHER__")
            continue
        names.extend(f"{column}={level}" for level in levels)
        names.append(f"{column}=__OTHER__")
    if getattr(preprocessor, "include_building_", False):
        names.append("BLOCK__building")
    return names

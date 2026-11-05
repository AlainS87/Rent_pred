import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from src.backtest.decomposed import HedonicMonths, MarketTrend, fit_decomposed, require_static_columns
from src.backtest.protocol import TARGET, digest_json, horizon, inner_split, months


def bounded(full):
    train = full.loc[months(full) <= pd.Period("2022-12", freq="M")].copy()
    query = full.loc[months(full) > pd.Period("2022-12", freq="M")].drop(columns=TARGET).copy()
    return train, query


@pytest.mark.training
def test_centering_reconstructs_original_ridge_and_weighted_effect_zero(labeled_fixture, tiny_config):
    train, _ = bounded(labeled_fixture)
    fitted = HedonicMonths("2022-12", tiny_config["decomposed"]).fit(train)
    index = fitted.index_
    assert np.average(index.market_effect, weights=index["count"]) == pytest.approx(0, abs=1e-10)
    assert fitted.property_intercept_ == pytest.approx(fitted.ridge_.intercept_ + fitted.center_shift_)
    assert fitted.reconstruction_max_abs_error_ < 1e-9
    prop = fitted.property_predict(train)
    market = fitted.historical_market(train)
    raw_month = index.set_index("month").raw_effect.reindex(months(train).astype(str)).to_numpy()
    original = fitted.property_matrix(train) @ fitted.ridge_.coef_[:fitted.n_property_] + fitted.ridge_.intercept_ + raw_month
    np.testing.assert_allclose(prop + market, original, rtol=0, atol=1e-9)


def test_declared_property_columns_are_static_and_forbidden_inputs_rejected(tiny_config):
    declared = tiny_config["decomposed"]
    require_static_columns(declared["property_features"])
    require_static_columns(declared["hedonic_categorical"] + declared["hedonic_numeric"])
    assert "building_key" in declared["property_features"]
    for bad in ["RENT_APPROVAL_DATE", "BLOCK", "building_age_years", "flat_age_from_lease", "building_age_x_central",
                "mean_rent_nearest_mrt", "coe_c_premium_lag1", "bank_basket_return_lag1", TARGET]:
        with pytest.raises(ValueError, match="forbidden"):
            require_static_columns(["FLOOR_AREA_SQM", bad])


def test_trend_formula_h1_h16_short_history_and_invalid_values():
    index = pd.DataFrame({"month": ["2022-10", "2022-11", "2022-12"], "count": [10., 20., 30.],
                          "market_effect": [-20., -10., 0.]})
    trend = MarketTrend("damped", "2022-12", {"window": 12, "phi": 0.9}).fit(index)
    expected = [10 * sum(0.9 ** k for k in range(1, h + 1)) for h in [1, 16]]
    np.testing.assert_allclose(trend.predict_horizon(np.array([1, 16])), expected, rtol=0, atol=1e-10)
    assert trend.slope_ == pytest.approx(10)
    flat = MarketTrend("flat", "2022-12", {"window": 3}).fit(index)
    np.testing.assert_allclose(flat.predict_horizon(np.array([1, 16])), np.average(index.market_effect, weights=index["count"]))
    one = MarketTrend("damped", "2022-12", {"window": 12, "phi": 0.9}).fit(index.iloc[-1:])
    np.testing.assert_array_equal(one.predict_horizon(np.array([1, 16])), [0, 0])
    for invalid in [np.array([0]), np.array([17]), np.array([1.5])]:
        with pytest.raises(ValueError, match="horizon"):
            trend.predict_horizon(invalid)
    with pytest.raises(ValueError, match="finite"):
        MarketTrend("flat", "2022-12", {"window": 3}).fit(index.assign(market_effect=[0, np.nan, 1]))
    with pytest.raises(ValueError, match="future"):
        MarketTrend("flat", "2022-11", {"window": 3}).fit(index)


@pytest.mark.training
def test_decomposition_future_perturbation_inner_independence_sharing_and_reload(labeled_fixture, tiny_config, tmp_path):
    train, query = bounded(labeled_fixture)
    first = fit_decomposed(train, query, "2022-12", tiny_config)
    mutated = labeled_fixture.copy()
    mutated.loc[months(mutated) > pd.Period("2022-12", freq="M"), TARGET] = 1e12
    other_train, other_query = bounded(mutated)
    second = fit_decomposed(other_train, other_query, "2022-12", tiny_config)
    for name in ["flat", "damped"]:
        bundle, components = first[name]
        assert bundle.audit == second[name][0].audit
        pd.testing.assert_frame_equal(components, second[name][1])
        path = tmp_path / f"{name}.joblib"
        bundle.save(path)
        restored = joblib.load(path)
        pd.testing.assert_frame_equal(components, restored.predict_components(query))
        cutoff = bundle.audit["inner"]["boundary"]
        assert cutoff == "2022-09"
        inner_history = train.loc[months(train) <= pd.Period(cutoff, freq="M")].reset_index(drop=True)
        independently_fitted = HedonicMonths(cutoff, tiny_config["decomposed"]).fit(inner_history)
        assert bundle.audit["inner"]["index_sha256"] == digest_json(independently_fitted.state())
        assert max(bundle.audit["inner"]["trend"]["months_used"]) <= cutoff
        assert bundle.audit["inner"]["allowed_history_sha256"] != bundle.audit["allowed_history_sha256"]
        # Ignored date-derived ages and supplied future exogenous data cannot affect the property part.
        poisoned_query = query.copy()
        for column in ["building_age_years", "flat_age_from_lease", "building_age_x_central", "coe_c_premium_lag1"]:
            poisoned_query[column] = 9999999
        np.testing.assert_array_equal(components.property_component, bundle.predict_components(poisoned_query).property_component)
        # Identical housing at different future dates has an identical static property component.
        housing = pd.concat([query.iloc[[0]], query.iloc[[0]]], ignore_index=True)
        housing["row_id"] = [900000, 900001]
        housing["RENT_APPROVAL_DATE"] = ["2023-01", "2024-04"]
        pair = bundle.predict_components(housing)
        assert pair.property_component.iloc[0] == pair.property_component.iloc[1]
        assert np.isfinite(pair.to_numpy(dtype=float)).all()
    # Independent single-trend implementation must equal shared preparation exactly.
    separate, separate_pred = fit_decomposed(train, query, "2022-12", tiny_config, ["damped"])["damped"]
    assert separate.audit == first["damped"][0].audit
    pd.testing.assert_frame_equal(separate_pred, first["damped"][1])


@pytest.mark.training
def test_decomposition_source_reader_ignores_unparseable_future_labels(labeled_fixture, tiny_config, tmp_path):
    from src.backtest.data import history_at, query_at
    path = tmp_path / "data/raw"
    path.mkdir(parents=True)
    source = path / "train.csv"
    raw = labeled_fixture.reset_index(drop=True).copy()
    raw.to_csv(source, index=False)
    base = raw.drop(columns=TARGET)
    query = query_at(base, "2022-12")
    first, prediction = fit_decomposed(history_at(base, "2022-12", tmp_path), query, "2022-12", tiny_config, ["flat"])["flat"]
    raw[TARGET] = raw[TARGET].astype(object)
    raw.loc[months(raw) > pd.Period("2022-12", freq="M"), TARGET] = "UNAVAILABLE_FUTURE_TARGET"
    raw.to_csv(source, index=False)
    second, repeated = fit_decomposed(history_at(base, "2022-12", tmp_path), query, "2022-12", tiny_config, ["flat"])["flat"]
    assert first.audit == second.audit
    pd.testing.assert_frame_equal(prediction, repeated)


@pytest.mark.training
def test_inner_validation_label_change_cannot_enter_decomposition_or_static_features(labeled_fixture, tiny_config):
    train, query = bounded(labeled_fixture)
    original = fit_decomposed(train, query, "2022-12", tiny_config, ["flat"])["flat"][0]
    changed = train.copy()
    changed.loc[months(changed) > pd.Period("2022-09", freq="M"), TARGET] = -1e8
    after = fit_decomposed(changed, query, "2022-12", tiny_config, ["flat"])["flat"][0]
    for key in ["index_sha256", "property", "trend", "allowed_history_sha256"]:
        assert original.audit["inner"][key] == after.audit["inner"][key]
    assert original.audit["inner"]["selected_iterations"] != after.audit["inner"]["selected_iterations"]


@pytest.mark.training
def test_negative_final_rent_is_floored_and_nonfinite_predictions_rejected(labeled_fixture, tiny_config, monkeypatch):
    train, query = bounded(labeled_fixture)
    bundle = fit_decomposed(train, query, "2022-12", tiny_config, ["flat"])["flat"][0]
    monkeypatch.setattr(bundle.model, "predict", lambda features: np.full(len(features), -1e9))
    result = bundle.predict_components(query)
    assert (result.unclipped_prediction < 0).all()
    assert (result.prediction == 0).all()
    monkeypatch.setattr(bundle.model, "predict", lambda features: np.full(len(features), np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        bundle.predict(query)

import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest.features import OOFBuildingEncoder, prepare_features, stable_folds
from src.backtest.models import feature_columns, fit_direct, load_bundle, select_iterations
from src.backtest.protocol import TARGET, inner_split, months
from src.features.group_means import GROUP_MEAN_COLUMNS, add_group_mean_columns


def bounded(full, origin="2022-12"):
    cutoff = pd.Period(origin, freq="M")
    history = full.loc[months(full) <= cutoff].copy()
    query = full.loc[(months(full) > cutoff) & (months(full) <= cutoff + 16)].drop(columns=TARGET).copy()
    return history, query


@pytest.mark.parametrize("model_name", ["direct_all93", "direct_legacy40"])
@pytest.mark.training
def test_post_origin_labels_do_not_change_full_fit_selection_or_predictions(labeled_fixture, tiny_config, model_name):
    history, query = bounded(labeled_fixture)
    first, pred = fit_direct(history, query, "2022-12", model_name, tiny_config)
    mutated = labeled_fixture.copy()
    mutated.loc[months(mutated) > pd.Period("2022-12", freq="M"), TARGET] = 999999999.0
    second_history, second_query = bounded(mutated)
    second, second_pred = fit_direct(second_history, second_query, "2022-12", model_name, tiny_config)
    pd.testing.assert_frame_equal(history, second_history)
    assert first.audit == second.audit  # both inner/outer matrices, prior/fills/maps, chosen settings
    np.testing.assert_array_equal(pred, second_pred)
    assert first.audit["selected_iterations"] == second.audit["selected_iterations"]


@pytest.mark.training
def test_inner_validation_targets_affect_stopping_but_not_training_features(labeled_fixture, tiny_config):
    history, _ = bounded(labeled_fixture)
    train, valid_x, valid_y, cutoff = inner_split(history, "2022-12", tiny_config)
    columns = feature_columns("direct_all93", tiny_config)
    selected, original = select_iterations(train, valid_x, valid_y, cutoff, columns, tiny_config)
    changed_selected, changed = select_iterations(train, valid_x, np.full_like(valid_y, -1e8), cutoff, columns, tiny_config)
    # This is an effective negative control: a broken test that ignores y_valid cannot pass.
    assert selected != changed_selected
    for key in ["training_matrix_sha256", "query_matrix_sha256", "preprocessor_sha256"]:
        assert original[key] == changed[key]
    assert TARGET not in valid_x


def test_all_six_means_exact_cutoff_and_far_horizon_missing(labeled_fixture):
    history, query = bounded(labeled_fixture)
    built = add_group_mean_columns(query, history)
    offsets = months(query).asi8 - pd.Period("2022-12", freq="M").ordinal
    assert built.loc[offsets >= 4, GROUP_MEAN_COLUMNS].isna().all().all()
    assert built.loc[offsets <= 3, GROUP_MEAN_COLUMNS].notna().all().all()
    # Every synthetic target's geography groups are the same two towns.
    for offset in [1, 2, 3]:
        row = query.loc[offsets == offset].iloc[[0]]
        target_month = months(row)[0]
        eligible = history.loc[(months(history) < target_month) & (months(history) >= target_month - 3)]
        own_group = eligible.loc[eligible.TOWN == row.TOWN.iloc[0], TARGET]
        expected = (own_group.sum() + 20 * eligible[TARGET].mean()) / (len(own_group) + 20)
        for column in GROUP_MEAN_COLUMNS:
            actual = built.loc[built.row_id == row.row_id.iloc[0], column].iloc[0]
            assert actual == pytest.approx(expected)


def test_inner_cutoff_perturbation_cannot_enter_six_means_prior_or_fill(labeled_fixture, tiny_config):
    history, _ = bounded(labeled_fixture)
    train, valid_x, _, cutoff = inner_split(history, "2022-12", tiny_config)
    a, b, prep, audit = prepare_features(train, valid_x, cutoff, feature_columns("direct_all93", tiny_config), tiny_config)
    poisoned = history.copy()
    poisoned.loc[months(poisoned) > pd.Period(cutoff, freq="M"), TARGET] = -500000
    other_train, other_valid, _, _ = inner_split(poisoned, "2022-12", tiny_config)
    c, d, other_prep, other_audit = prepare_features(other_train, other_valid, cutoff, feature_columns("direct_all93", tiny_config), tiny_config)
    pd.testing.assert_frame_equal(a, c)
    pd.testing.assert_frame_equal(b, d)
    assert prep.state() == other_prep.state()
    assert audit == other_audit


def test_entire_fold_including_prior_is_excluded_and_folds_are_stable(labeled_fixture):
    history, query = bounded(labeled_fixture)
    encoder = OOFBuildingEncoder()
    before = encoder.fit_transform(history)
    fold = stable_folds(history.row_id)
    changed = history.copy()
    changed.loc[fold == fold[0], TARGET] += 10000000
    after = OOFBuildingEncoder().fit_transform(changed)
    np.testing.assert_allclose(before[fold == fold[0]], after[fold == fold[0]], rtol=0, atol=1e-9)
    shuffled = history.sample(frac=1, random_state=8)
    folded = pd.Series(stable_folds(shuffled.row_id), index=shuffled.row_id)
    np.testing.assert_array_equal(fold, folded.loc[history.row_id].to_numpy())
    shuffled_encoding = pd.Series(OOFBuildingEncoder().fit_transform(shuffled), index=shuffled.row_id)
    np.testing.assert_allclose(before, shuffled_encoding.loc[history.row_id].to_numpy(), rtol=0, atol=1e-10)
    unknown = query.iloc[[0]].copy()
    unknown["BLOCK"] = "never_seen"
    np.testing.assert_array_equal(encoder.transform(unknown), [history[TARGET].mean()])


@pytest.mark.training
def test_bundle_reload_and_query_reordering_preserve_row_alignment(labeled_fixture, tiny_config, tmp_path):
    history, query = bounded(labeled_fixture)
    bundle, expected = fit_direct(history, query, "2022-12", "direct_legacy40", tiny_config)
    artifact = tmp_path / "strict.joblib"
    bundle.save(artifact)
    restored = load_bundle(artifact)
    np.testing.assert_array_equal(expected, restored.predict(query))
    shuffled = query.sample(frac=1, random_state=19)
    remapped = pd.Series(restored.predict(shuffled), index=shuffled.row_id)
    np.testing.assert_array_equal(expected, remapped.loc[query.row_id].to_numpy())
    shuffled_fit, shuffled_pred = fit_direct(history.sample(frac=1, random_state=2), query,
                                            "2022-12", "direct_legacy40", tiny_config)
    assert bundle.audit == shuffled_fit.audit
    np.testing.assert_array_equal(expected, shuffled_pred)


@pytest.mark.training
def test_end_to_end_source_reader_selection_and_predictions_ignore_future_csv_labels(labeled_fixture, tiny_config, tmp_path):
    import json
    from src.backtest.data import AUXILIARY, BASE_COLUMNS, history_at, identity_hash, load_cache, query_at, sha256, source_hashes
    data = tmp_path / "data/raw"
    data.mkdir(parents=True)
    source = data / "train.csv"
    raw = labeled_fixture.reset_index(drop=True).copy()
    raw.to_csv(source, index=False)
    base = labeled_fixture.drop(columns=TARGET)
    auxiliary = data / "auxiliary"
    auxiliary.mkdir(parents=True)
    for name in AUXILIARY.values():
        (auxiliary / name).write_text("fixed auxiliary source", encoding="utf-8")
    code = tmp_path / "src/backtest"
    code.mkdir(parents=True)
    (code / "data.py").write_text("fixed cache generation code", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    parquet = cache / "train_base.parquet"
    base = base[BASE_COLUMNS].reset_index(drop=True)
    base.to_parquet(parquet, index=False)
    manifest = {"rows": len(base), "columns": BASE_COLUMNS, "parquet_sha256": sha256(parquet),
                "identity_sha256": identity_hash(base), "source_sha256": source_hashes(tmp_path, "train")}
    (cache / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    first_base = load_cache("train", tmp_path, cache)
    query = query_at(first_base, "2022-12")
    original, pred = fit_direct(history_at(first_base, "2022-12", tmp_path), query, "2022-12", "direct_all93", tiny_config)
    raw[TARGET] = raw[TARGET].astype(object)
    raw.loc[months(raw) > pd.Period("2022-12", freq="M"), TARGET] = "FORBIDDEN_FUTURE_LABEL"
    raw.to_csv(source, index=False)
    second_base = load_cache("train", tmp_path, cache)
    changed, second = fit_direct(history_at(second_base, "2022-12", tmp_path), query, "2022-12", "direct_all93", tiny_config)
    assert original.audit == changed.audit
    np.testing.assert_array_equal(pred, second)


def test_model_rejects_full_labeled_frame_and_precomputed_means(labeled_fixture, tiny_config):
    history, query = bounded(labeled_fixture)
    with pytest.raises(ValueError, match="after its fit boundary"):
        fit_direct(labeled_fixture, query, "2022-12", "direct_all93", tiny_config)
    with pytest.raises(ValueError, match="must not contain target"):
        fit_direct(history, query.assign(MONTHLY_RENT=1), "2022-12", "direct_all93", tiny_config)
    with pytest.raises(ValueError, match="precomputed"):
        prepare_features(history.assign(mean_rent_nearest_mrt=1), query, "2022-12", feature_columns("direct_all93", tiny_config), tiny_config)

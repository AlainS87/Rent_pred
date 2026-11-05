"""Non-training tests must never accidentally invoke a real model fitter."""
import pytest


@pytest.fixture(autouse=True)
def forbid_unmarked_training(request, monkeypatch):
    if request.node.get_closest_marker("training"):
        return

    def forbidden(*args, **kwargs):
        pytest.fail("Real estimator fitting requires @pytest.mark.training")

    # Imputation, category dictionaries and mathematical transforms remain
    # available. Guard the actual model backends used by the core test suite.
    from sklearn.linear_model import Ridge
    from catboost import CatBoost, CatBoostRegressor
    from xgboost import XGBModel, XGBRegressor

    for estimator in (Ridge, CatBoost, CatBoostRegressor, XGBModel, XGBRegressor):
        monkeypatch.setattr(estimator, "fit", forbidden)

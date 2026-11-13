import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

import evaluation as ev
from evaluation import InsufficientGroupsError, inner_cv_for, nested_loco, validate_groups

GRID = [{"model": [Ridge()], "model__alpha": [0.1, 10.0]}]


def _run(df, features, **kw):
    return nested_loco(df, features, seed=0, param_grid=GRID, n_jobs=1,
                       log=lambda *_: None, **kw)


def test_each_outer_fold_holds_out_exactly_one_unseen_city(processed, features):
    res = _run(processed, features)
    cities = sorted(processed["city_name"].unique())
    assert [f["held_out_city"] for f in res.folds] == cities
    for f in res.folds:
        fold_preds = res.predictions[res.predictions["outer_fold"] == f["outer_fold"]]
        assert set(fold_preds["city"]) == {f["held_out_city"]}
        assert f["n_train_cities"] == len(cities) - 1
        assert f["n_train"] == (processed["city_name"] != f["held_out_city"]).sum()


def test_inner_splits_never_share_a_city(processed):
    for held in processed["city_name"].unique():
        train = processed[processed["city_name"] != held]
        g = train["city_name"].to_numpy()
        cv = inner_cv_for(len(np.unique(g)))
        for tr, va in cv.split(train, groups=g):
            assert not set(g[tr]) & set(g[va])
            assert held not in set(g[tr]) | set(g[va])


class _Spy(BaseEstimator, TransformerMixin):
    """Records which city codes each preprocessing fit sees."""
    calls: list = []
    current = {"fold": None, "city": None}

    def fit(self, X, y=None):
        _Spy.calls.append((_Spy.current["fold"], set(X["city_code"].astype(int))))
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_)


def test_held_out_city_never_reaches_any_fit_during_selection(processed, features, monkeypatch):
    real = ev.make_preprocessor
    monkeypatch.setattr(ev, "make_preprocessor",
                        lambda: __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
                            [("spy", _Spy())] + real().steps))
    _Spy.calls = []
    code_of = dict(zip(processed["city_name"], processed["city_code"].astype(int)))

    def on_start(fold, city):
        _Spy.current = {"fold": fold, "city": city}
        on_start.held[fold] = code_of[city]
    on_start.held = {}

    _run(processed, features, on_fold_start=on_start)
    # 2 configs × inner folds + 1 refit per outer fold
    assert len(_Spy.calls) > len(on_start.held)
    for fold, seen in _Spy.calls:
        assert on_start.held[fold] not in seen


def test_outer_test_labels_do_not_change_selection_or_fit(processed, features):
    base = _run(processed, features)
    poisoned = processed.copy()
    mask = poisoned["city_name"] == "City3"
    poisoned.loc[mask, "uhi_intensity"] = poisoned.loc[mask, "uhi_intensity"] * 100 + 50
    pois = _run(poisoned, features)
    f_base = next(f for f in base.folds if f["held_out_city"] == "City3")
    f_pois = next(f for f in pois.folds if f["held_out_city"] == "City3")
    assert f_base["selected_params"] == f_pois["selected_params"]
    assert f_base["inner_cv_rmse"] == pytest.approx(f_pois["inner_cv_rmse"])
    pb = base.predictions.query("city == 'City3'")["predicted_uhi"].to_numpy()
    pp = pois.predictions.query("city == 'City3'")["predicted_uhi"].to_numpy()
    np.testing.assert_allclose(pb, pp)


def test_mean_baseline_uses_training_fold_mean_only(processed, features):
    res = _run(processed, features)
    for f in res.folds:
        p = res.predictions.query(
            "outer_fold == @f['outer_fold'] and model_name == 'mean_baseline'")
        train_mean = processed.loc[processed["city_name"] != f["held_out_city"],
                                   "uhi_intensity"].mean()
        np.testing.assert_allclose(p["predicted_uhi"], train_mean)


def test_too_few_cities_fails_instead_of_row_split(features):
    from conftest import make_processed
    with pytest.raises(InsufficientGroupsError):
        validate_groups(make_processed(n_cities=3))


def test_row_level_identifier_rejected(processed):
    df = processed.copy()
    df["city_name"] = df["observation_id"]
    with pytest.raises(InsufficientGroupsError):
        validate_groups(df.drop(columns=["lat", "lon"]))


def test_missing_group_column_rejected(processed):
    with pytest.raises(InsufficientGroupsError):
        validate_groups(processed.drop(columns=["city_name"]))

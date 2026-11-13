import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from evaluation import (DuplicateFeatureDropper, Winsorizer, make_preprocessor, nested_loco,
                        retained_features)

GRID = [{"model": [Ridge()]}]


def _split(df, city):
    return df[df["city_name"] != city], df[df["city_name"] == city]


def test_imputer_and_scaler_fitted_on_training_fold_only(processed, features):
    df = processed.copy()
    held = "City2"
    # extreme values + missing values only in the held-out city
    df.loc[df["city_name"] == held, "temperature"] = 1e4
    df.loc[df.index[:5], "humidity"] = np.nan
    res = nested_loco(df, features, seed=0, param_grid=GRID, n_jobs=1, keep_fitted=True,
                      log=lambda *_: None)
    fitted = res.fitted[(held, "nested_selected_model")].named_steps["preprocess"]
    train, _ = _split(df, held)
    expected = make_preprocessor().fit(train[features])
    for step in ("impute", "scale"):
        a, b = fitted.named_steps[step], expected.named_steps[step]
        attr = "statistics_" if step == "impute" else "mean_"
        np.testing.assert_allclose(getattr(a, attr), getattr(b, attr))
    leaky = make_preprocessor().fit(df[features])
    assert not np.allclose(fitted.named_steps["scale"].mean_,
                           leaky.named_steps["scale"].mean_)
    np.testing.assert_allclose(fitted.named_steps["winsorize"].upper_,
                               expected.named_steps["winsorize"].upper_)


def test_winsorizer_fences_come_from_training_data():
    tr = pd.DataFrame({"temperature": np.arange(100, dtype=float), "other": 1.0})
    w = Winsorizer(columns=("temperature",)).fit(tr)
    q1, q3 = np.percentile(tr["temperature"], [25, 75])
    assert w.lower_[0] == q1 - 1.5 * (q3 - q1) and w.upper_[0] == q3 + 1.5 * (q3 - q1)
    assert np.isinf(w.upper_[1])
    out = w.transform(pd.DataFrame({"temperature": [1e6, np.nan], "other": [5.0, 5.0]}))
    assert out[0, 0] == w.upper_[0] and np.isnan(out[1, 0]) and out[0, 1] == 5.0


def test_constant_feature_detection_is_fold_local(processed, features):
    df = processed.copy()
    df["only_varies_in_City1"] = 0.0
    m = df["city_name"] == "City1"
    df.loc[m, "only_varies_in_City1"] = np.linspace(0, 1, m.sum())
    feats = features + ["only_varies_in_City1"]
    res = nested_loco(df, feats, seed=0, param_grid=GRID, n_jobs=1, log=lambda *_: None)
    for f in res.folds:
        kept = f["retained_features"]
        assert "hour" not in kept                       # constant everywhere
        if f["held_out_city"] == "City1":
            assert "only_varies_in_City1" not in kept  # constant on training rows
        else:
            assert "only_varies_in_City1" in kept


def test_duplicate_detection_is_fold_local_and_keeps_correlated(processed, features):
    df = processed.copy()
    df["dup_except_City4"] = df["temperature"]
    m = df["city_name"] == "City4"
    df.loc[m, "dup_except_City4"] = df.loc[m, "temperature"] + 1.0
    df["affine_temperature"] = 2 * df["temperature"] + 1   # correlated, not identical
    feats = features + ["dup_except_City4", "affine_temperature"]
    res = nested_loco(df, feats, seed=0, param_grid=GRID, n_jobs=1, log=lambda *_: None)
    for f in res.folds:
        kept = f["retained_features"]
        assert "distance_from_equator" not in kept and "lat_abs" in kept
        assert "affine_temperature" in kept
        if f["held_out_city"] == "City4":
            assert "dup_except_City4" not in kept
        else:
            assert "dup_except_City4" in kept


def test_duplicate_dropper_unit():
    X = np.array([[1, 1, 2, 3], [2, 2, 4, 3], [3, 3, 6, 3]], dtype=float)
    d = DuplicateFeatureDropper().fit(X)
    assert d.support_.tolist() == [True, False, True, True]
    assert d.get_feature_names_out(["a", "b", "c", "d"]).tolist() == ["a", "c", "d"]


def test_retained_feature_names_follow_pipeline(processed, features):
    from evaluation import make_model_pipeline
    pipe = make_model_pipeline(Ridge()).fit(processed[features], processed["uhi_intensity"])
    kept = retained_features(pipe, features)
    assert "hour" not in kept and "distance_from_equator" not in kept
    assert len(kept) == pipe.named_steps["model"].coef_.shape[0]


def test_stateless_preprocessing_is_unaffected_by_other_cities():
    """build_processed_table must not learn anything across cities."""
    from preprocessor import build_processed_table

    def raw(city, lat, temps, lst):
        return pd.DataFrame({
            "name": city, "lat": lat, "lon": 1.0, "temperature": temps, "humidity": 50.0,
            "wind_speed": 2.0, "pressure": np.nan, "clouds": 10.0, "ndvi": 0.3,
            "urban_fraction": 0.7, "urban_lst": [l + 1 for l in lst], "rural_lst": lst,
            "timestamp": [f"2025-01-0{i + 1}T12:00:00" for i in range(len(temps))],
            "lst_date": "2025-01-01", "source": "openmeteo+test",
        })

    a = raw("A", 10.0, [20.0, 22.0, 24.0], [30.0, 31.0, 32.0])
    b = raw("B", 50.0, [100.0, 200.0, 300.0], [0.0, 0.0, 0.0])
    only_a = build_processed_table(a)
    both = build_processed_table(pd.concat([a, b]))
    pd.testing.assert_frame_equal(only_a, both[both["city_name"] == "A"].reset_index(drop=True))
    assert only_a["pressure"].isna().all()  # missing features are NOT imputed here

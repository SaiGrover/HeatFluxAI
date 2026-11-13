import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import OBSERVED_TARGET_PROVENANCE
from evaluation import eligible_rows, nested_loco
from preprocessor import compute_uhi_intensity


def test_target_is_signed_and_never_imputed():
    raw = pd.DataFrame({
        "name": ["A", "A", "A", "B"],
        "urban_lst": [30.0, 25.0, np.nan, 40.0],
        "rural_lst": [28.0, 29.5, 27.0, np.nan],
    })
    out = compute_uhi_intensity(raw)
    assert out["uhi_intensity"].iloc[0] == 2.0
    assert out["uhi_intensity"].iloc[1] == -4.5          # negative kept, not clipped
    assert out["uhi_intensity"].iloc[2:].isna().all()    # no median fill
    assert out["target_provenance"].tolist()[:2] == [OBSERVED_TARGET_PROVENANCE] * 2
    assert (out["target_provenance"].iloc[2:] != OBSERVED_TARGET_PROVENANCE).all()


def test_imputed_and_missing_targets_are_excluded(processed):
    df = processed.copy()
    df.loc[0:4, "target_provenance"] = "imputed_city_median"
    df.loc[5:6, "uhi_intensity"] = np.nan
    elig, info = eligible_rows(df)
    assert len(elig) == len(df) - 7
    assert info["n_excluded"] == 7
    assert sum(info["reasons"].values()) == 7
    assert (elig["target_provenance"] == OBSERVED_TARGET_PROVENANCE).all()


def test_missing_provenance_column_refuses():
    import pytest
    with pytest.raises(ValueError):
        eligible_rows(pd.DataFrame({"uhi_intensity": [1.0]}))


def test_negative_targets_scored_and_residual_signs(processed, features):
    df = processed.copy()
    df.loc[0:9, "target_provenance"] = "imputed_city_median"
    elig, _ = eligible_rows(df)
    res = nested_loco(elig, features, seed=0, param_grid=[{"model": [Ridge()]}], n_jobs=1,
                      log=lambda *_: None)
    p = res.predictions
    assert (p["target_provenance"] == OBSERVED_TARGET_PROVENANCE).all()
    assert not set(df.loc[0:9, "observation_id"]) & set(p["observation_id"])
    assert (p["observed_uhi"] < 0).sum() > 0
    np.testing.assert_allclose(p["residual"], p["observed_uhi"] - p["predicted_uhi"])
    np.testing.assert_allclose(p["absolute_error"], p["residual"].abs())
    # negative predictions are possible (City0 has negative UHI)
    sel = p[(p["model_name"] == "nested_selected_model") & (p["city"] == "City0")]
    assert (sel["predicted_uhi"] < 0).any() or (sel["observed_uhi"] < 0).all()


def test_no_nonnegative_filter_in_pipeline_sources():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ("preprocessor.py", "evaluation.py", "model_trainer.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "clip(lower=0)" not in src.replace("(df[\"temperature\"] - 20).clip(lower=0)", "")
        assert '["uhi_intensity"] >= 0' not in src

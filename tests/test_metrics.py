import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import (aggregate_metrics, city_bootstrap, mae, recompute_from_predictions,
                        rmse, run_full_evaluation)

ROOT = Path(__file__).resolve().parents[1]


def test_mae_rmse_match_hand_computation():
    y, p = [1.0, -2.0, 3.0], [0.0, 0.0, 0.0]
    assert mae(y, p) == pytest.approx(2.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(14 / 3))


def _preds(city_errors: dict) -> pd.DataFrame:
    rows = []
    for city, errs in city_errors.items():
        for i, e in enumerate(errs):
            rows.append({"city": city, "model_name": "m", "observed_uhi": e,
                         "predicted_uhi": 0.0, "observation_id": f"{city}{i}"})
    df = pd.DataFrame(rows)
    df["residual"] = df["observed_uhi"] - df["predicted_uhi"]
    df["absolute_error"] = df["residual"].abs()
    df["squared_error"] = df["residual"] ** 2
    return df


def test_macro_and_pooled_are_different_quantities():
    df = _preds({"A": [10.0], "B": [0.0] * 99})
    a = aggregate_metrics(df, "m")
    assert a["macro_mae"] == pytest.approx(5.0)
    assert a["pooled_mae"] == pytest.approx(0.1)
    assert a["macro_rmse"] == pytest.approx(5.0)        # mean of city RMSEs (10, 0)
    assert a["pooled_rmse"] == pytest.approx(1.0)       # sqrt(100 / 100)


def test_city_bootstrap_reproducible_and_city_level():
    df = _preds({c: list(np.random.default_rng(i).normal(0, i + 1, 20)) for i, c in
                 enumerate("ABCDEFGHIJ")})
    a = city_bootstrap(df, ["m"], 500, seed=1)
    b = city_bootstrap(df, ["m"], 500, seed=1)
    c = city_bootstrap(df, ["m"], 500, seed=2)
    assert a == b and a != c
    lo, hi = a["intervals"]["m"]["macro_rmse"]
    point = aggregate_metrics(df, "m")["macro_rmse"]
    assert lo <= point <= hi
    # with a single city, every resample is that city: zero-width interval
    one = city_bootstrap(_preds({"A": [1.0, -1.0, 2.0]}), ["m"], 100, seed=0)
    lo, hi = one["intervals"]["m"]["pooled_rmse"]
    assert lo == pytest.approx(hi)


def test_metrics_json_agrees_with_prediction_artifact(tmp_path, features):
    from conftest import make_processed
    processed = make_processed(n_cities=8, n_per=20)
    mpath, ppath = tmp_path / "metrics.json", tmp_path / "preds.csv"
    metrics, preds, _, _ = run_full_evaluation(
        processed, features, seed=3, n_boot=300, n_jobs=1, fast=True,
        predictions_path=ppath, metrics_path=mpath, fit_deployment=False,
        log=lambda *_: None)
    saved = pd.read_csv(ppath)
    on_disk = json.loads(mpath.read_text(encoding="utf-8"))
    re = recompute_from_predictions(saved)
    for m, block in on_disk["models"].items():
        for k in ("macro_mae", "macro_rmse", "pooled_mae", "pooled_rmse"):
            assert re[m][k] == pytest.approx(block[k], abs=1e-5)
    for c in on_disk["per_city"]:
        sub = saved[(saved["city"] == c["city"]) & (saved["model_name"] == "nested_selected_model")]
        assert c["n_evaluated"] == len(sub)
        assert c["metrics"]["nested_selected_model"]["rmse"] == pytest.approx(
            rmse(sub["observed_uhi"], sub["predicted_uhi"]), abs=1e-5)
    assert on_disk["aggregate"]["confidence_intervals"]["status"] != "unavailable"
    # deterministic under a fixed seed
    metrics2, *_ = run_full_evaluation(processed, features, seed=3, n_boot=300, n_jobs=1,
                                       fast=True, fit_deployment=False, log=lambda *_: None)
    assert metrics2["aggregate"] == metrics["aggregate"]


def test_ci_unavailable_with_too_few_cities(features):
    from conftest import make_processed
    metrics, *_ = run_full_evaluation(make_processed(n_cities=5, n_per=15), features, seed=0,
                                      n_boot=100, n_jobs=1, fast=True, fit_deployment=False,
                                      log=lambda *_: None)
    assert metrics["aggregate"]["confidence_intervals"]["status"] == "unavailable"


@pytest.mark.skipif(not (ROOT / "evaluation" / "metrics.json").exists(),
                    reason="committed evaluation artifacts not present")
def test_committed_artifacts_are_consistent():
    from eval_artifacts import load_evaluation
    view = load_evaluation(ROOT / "evaluation" / "metrics.json",
                           ROOT / "evaluation" / "heldout_predictions.csv")
    assert view.ok, view.message
    m = view.metrics
    assert m["evaluation"]["method"] == "nested_leave_one_city_out"
    assert m["evaluation"]["test_data_used_for_selection"] is False
    assert m["dataset"]["target_summary"]["n_negative"] > 0
    assert set(view.predictions["city"]) == set(m["dataset"]["cities"])
    assert view.predictions.groupby("model_name").size().nunique() == 1

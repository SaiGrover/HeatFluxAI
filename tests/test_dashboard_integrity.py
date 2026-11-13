import json
from pathlib import Path

import pandas as pd
import pytest

from eval_artifacts import (RESIDUAL_LABELS, load_evaluation, residual_label, summary_rows)
from evaluation import run_full_evaluation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(tmp_path, features):
    from conftest import make_processed
    mpath, ppath = tmp_path / "metrics.json", tmp_path / "heldout_predictions.csv"
    run_full_evaluation(make_processed(n_cities=8, n_per=15), features, seed=0, n_boot=200,
                        n_jobs=1, fast=True, predictions_path=ppath, metrics_path=mpath,
                        fit_deployment=False, log=lambda *_: None)
    return mpath, ppath


def test_labels_are_distinct_and_unknown_sources_rejected():
    assert residual_label("heldout_nested_cv") == "Held-Out City Residuals (Nested Cross-Validation)"
    assert residual_label("training") == "Training Residuals — Not an Independent Test"
    assert "Unavailable" in residual_label("predictions_only")
    assert len(set(RESIDUAL_LABELS.values())) == 3
    with pytest.raises(KeyError):
        residual_label("test")


def test_valid_artifacts_load(artifacts):
    view = load_evaluation(*artifacts)
    assert view.ok and view.residual_source == "heldout_nested_cv"


def test_missing_artifacts_give_message_not_numbers(tmp_path):
    view = load_evaluation(tmp_path / "metrics.json", tmp_path / "p.csv")
    assert not view.ok and "scripts/evaluate.py" in view.message and view.predictions is None


def test_training_style_metrics_are_not_shown_as_heldout(artifacts, tmp_path):
    mpath, ppath = artifacts
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["evaluation"]["method"] = "group_shuffle_split_single_test"
    bad = tmp_path / "legacy.json"
    bad.write_text(json.dumps(m))
    view = load_evaluation(bad, ppath)
    assert not view.ok and view.residual_source is None


def test_tampered_or_stale_predictions_rejected(artifacts):
    mpath, ppath = artifacts
    df = pd.read_csv(ppath)
    df.loc[0, "predicted_uhi"] += 1.0
    df.to_csv(ppath, index=False)
    assert not load_evaluation(mpath, ppath).ok


def test_non_observed_targets_rejected(artifacts):
    mpath, ppath = artifacts
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m.pop("artifacts", None)  # bypass checksum to reach the provenance check
    mpath.write_text(json.dumps(m))
    df = pd.read_csv(ppath)
    df.loc[0, "target_provenance"] = "imputed_city_median"
    df.to_csv(ppath, index=False)
    view = load_evaluation(mpath, ppath)
    assert not view.ok and "non-observed" in view.message


def test_metric_mismatch_rejected(artifacts):
    mpath, ppath = artifacts
    m = json.loads(mpath.read_text(encoding="utf-8"))
    m["models"]["nested_selected_model"]["pooled_rmse"] -= 0.5
    mpath.write_text(json.dumps(m))
    assert not load_evaluation(mpath, ppath).ok


def test_summary_rows_equal_metrics_json(artifacts):
    view = load_evaluation(*artifacts)
    for row in summary_rows(view.metrics):
        block = view.metrics["models"][row["key"]]
        assert row["Macro RMSE"] == block["macro_rmse"]
        assert row["Pooled MAE"] == block["pooled_mae"]


def test_dashboard_source_has_no_misleading_residuals():
    src = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    assert "Test residuals" not in src
    assert 'residual_label("heldout_nested_cv")' in src
    # residuals must not be recomputed from the deployment model on the full table
    assert "model.predict(scaler.transform(Xr))" not in src
    assert "skill_vs_baseline" not in src and "best_rmse" not in src

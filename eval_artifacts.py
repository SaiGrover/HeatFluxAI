"""
Loading and validating evaluation artifacts for the dashboard.

Kept free of Streamlit so the rules can be unit-tested:
  * residuals are only shown as test residuals when they come from the nested
    leave-one-city-out prediction artifact, the artifact matches metrics.json,
    and every scored target is an observed (non-imputed) value;
  * otherwise the dashboard gets an explanatory message instead of a fallback
    statistic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import OBSERVED_TARGET_PROVENANCE

NESTED_METHOD = "nested_leave_one_city_out"

RESIDUAL_LABELS = {
    "heldout_nested_cv": "Held-Out City Residuals (Nested Cross-Validation)",
    "training": "Training Residuals — Not an Independent Test",
    "predictions_only": "Predictions — Residuals Unavailable Without Observed Targets",
}

REQUIRED_PREDICTION_COLUMNS = [
    "observation_id", "city", "outer_fold", "model_name", "observed_uhi",
    "predicted_uhi", "residual", "absolute_error", "squared_error", "target_provenance",
]

MODEL_DISPLAY_NAMES = {
    "mean_baseline": "Mean baseline",
    "latitude_ndvi_baseline": "Latitude + NDVI baseline",
    "nested_selected_model": "Selected ML pipeline (nested)",
}


def residual_label(source: str) -> str:
    """Label for a residual panel.  Unknown sources raise instead of defaulting."""
    if source not in RESIDUAL_LABELS:
        raise KeyError(f"Unknown residual source '{source}'")
    return RESIDUAL_LABELS[source]


@dataclass
class EvaluationView:
    ok: bool
    message: str
    metrics: dict | None = None
    predictions: pd.DataFrame | None = None

    @property
    def residual_source(self) -> str | None:
        return "heldout_nested_cv" if self.ok else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rmse(s: pd.Series) -> float:
    return float((s ** 2).mean() ** 0.5)


def recompute(preds: pd.DataFrame, model: str) -> dict:
    sub = preds[preds["model_name"] == model]
    city = sub.groupby("city")
    return {
        "macro_mae": float(city["absolute_error"].mean().mean()),
        "macro_rmse": float(city["residual"].apply(_rmse).mean()),
        "pooled_mae": float(sub["absolute_error"].mean()),
        "pooled_rmse": _rmse(sub["residual"]),
    }


def load_evaluation(metrics_path: Path, predictions_path: Path,
                    tol: float = 1e-4) -> EvaluationView:
    metrics_path, predictions_path = Path(metrics_path), Path(predictions_path)
    if not metrics_path.exists():
        return EvaluationView(False, f"No evaluation found ({metrics_path.name} missing). "
                                     "Run `python scripts/evaluate.py --seed 42`.")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:  # malformed file
        return EvaluationView(False, f"{metrics_path.name} could not be read: {exc}")
    method = metrics.get("evaluation", {}).get("method")
    if method != NESTED_METHOD:
        return EvaluationView(False, f"{metrics_path.name} was not produced by nested "
                                     f"leave-one-city-out evaluation (method={method!r}); "
                                     "its scores are not shown as held-out results.")
    if not predictions_path.exists():
        return EvaluationView(False, f"Held-out prediction file {predictions_path.name} is "
                                     "missing, so residuals cannot be shown.", metrics)
    expected_sha = metrics.get("artifacts", {}).get("heldout_predictions_sha256")
    if expected_sha and _sha256(predictions_path) != expected_sha:
        return EvaluationView(False, f"{predictions_path.name} does not match the checksum "
                                     "recorded in metrics.json (stale or edited); re-run "
                                     "the evaluation.", metrics)
    preds = pd.read_csv(predictions_path)
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in preds.columns]
    if missing:
        return EvaluationView(False, f"Prediction file lacks columns {missing}.", metrics)
    if not preds["target_provenance"].eq(OBSERVED_TARGET_PROVENANCE).all():
        return EvaluationView(False, "Prediction file contains non-observed targets; "
                                     "residuals are not shown.", metrics)
    for model, block in metrics.get("models", {}).items():
        re = recompute(preds, model)
        for k, v in re.items():
            if abs(v - block[k]) > tol:
                return EvaluationView(False, f"metrics.json {model}.{k}={block[k]} does not "
                                             f"match the predictions ({v:.6f}).", metrics)
    return EvaluationView(True, "ok", metrics, preds)


def summary_rows(metrics: dict) -> list[dict]:
    """Rows for the model-comparison table, straight from metrics.json."""
    ci = metrics.get("aggregate", {}).get("confidence_intervals", {})
    intervals = ci.get("intervals", {}) if ci.get("status", "").startswith(("ok", "unstable")) else {}
    rows = []
    for key, block in metrics["models"].items():
        iv = intervals.get(key, {})
        rows.append({
            "key": key,
            "Model": MODEL_DISPLAY_NAMES.get(key, key),
            "Macro MAE": block["macro_mae"], "Macro RMSE": block["macro_rmse"],
            "Pooled MAE": block["pooled_mae"], "Pooled RMSE": block["pooled_rmse"],
            "Macro RMSE 95% CI": iv.get("macro_rmse"),
            "Pooled RMSE 95% CI": iv.get("pooled_rmse"),
            "Cities": block["n_cities"], "Obs": block["n_observations"],
            "Evaluation": "Held-out cities (nested LOCO)",
        })
    return rows


def per_city_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for c in metrics.get("per_city", []):
        r = {"City": c["city"], "n": c["n_evaluated"], "Selected": c["selected_family"]}
        for key, m in c["metrics"].items():
            short = {"mean_baseline": "Mean", "latitude_ndvi_baseline": "Lat+NDVI",
                     "nested_selected_model": "ML"}.get(key, key)
            r[f"{short} MAE"] = m["mae"]
            r[f"{short} RMSE"] = m["rmse"]
        rows.append(r)
    return pd.DataFrame(rows)

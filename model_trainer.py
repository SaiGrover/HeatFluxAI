"""
UHI Prediction System - Model Training

Thin wrapper around ``evaluation.run_full_evaluation``:

  1. nested leave-one-city-out evaluation (unbiased cross-city estimate) of the
     mean baseline, the latitude+NDVI baseline and the inner-CV-selected ML
     pipeline  →  evaluation/metrics.json + evaluation/heldout_predictions.csv
  2. a separately selected *deployment* pipeline (grouped CV on all eligible
     cities, refit on all of them)  →  models/best_model.pkl

The previous implementation picked the "best" model by its RMSE on the test
cities and reported that same score, fitted the scaler/imputer on all rows
before splitting, and cross-validated already-tuned models on the full data.
All of that is gone.  See docs/EVALUATION_METHODOLOGY.md.
"""

from __future__ import annotations

import json
import pickle

import pandas as pd

from config import (
    BEST_MODEL_PATH, FEATURES_PATH, HELDOUT_PREDICTIONS_PATH, METRICS_PATH,
    PROCESSED_DATA_PATH, RANDOM_STATE, RAW_DATA_PATH,
)
from logger import get_logger

log = get_logger("model_trainer")


def train(force: bool = False, fast: bool = False, seed: int = RANDOM_STATE,
          n_jobs: int = -1, raw_path=None) -> dict:
    """Run the nested evaluation, fit the deployment model, save artifacts."""
    if METRICS_PATH.exists() and BEST_MODEL_PATH.exists() and not force:
        log.info("Model artefacts exist. Loading metrics ...")
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)

    from evaluation import run_full_evaluation
    from preprocessor import candidate_features

    log.info("=" * 60)
    log.info("STEP 3 – NESTED LEAVE-ONE-CITY-OUT EVALUATION + DEPLOYMENT FIT")
    log.info("=" * 60)
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")
    processed = pd.read_csv(PROCESSED_DATA_PATH)
    features = candidate_features()

    metrics, _, _, deploy_model = run_full_evaluation(
        processed, features, seed=seed, n_jobs=n_jobs, fast=fast,
        raw_path=raw_path or RAW_DATA_PATH,
        predictions_path=HELDOUT_PREDICTIONS_PATH, metrics_path=METRICS_PATH,
        log=log.info,
    )
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(deploy_model, f)
    with open(FEATURES_PATH, "w") as f:
        json.dump(features, f)
    sel = metrics["models"]["nested_selected_model"]
    log.info(f"  Nested LOCO selected pipeline: macro RMSE {sel['macro_rmse']:.3f} °C, "
             f"pooled RMSE {sel['pooled_rmse']:.3f} °C")
    log.info(f"  Saved metrics → {METRICS_PATH}")
    log.info(f"  Saved deployment pipeline → {BEST_MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    m = train(force=True)
    for name in ("mean_baseline", "latitude_ndvi_baseline", "nested_selected_model"):
        a = m["models"][name]
        print(f"{name:26s} macro MAE {a['macro_mae']:.3f}  macro RMSE {a['macro_rmse']:.3f}  "
              f"pooled RMSE {a['pooled_rmse']:.3f}")

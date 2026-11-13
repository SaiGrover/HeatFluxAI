"""
Reproduce HeatFluxAI's published evaluation artifacts.

    python scripts/evaluate.py --seed 42

Reads the raw snapshot (default: data/snapshot_modis_pc_2025H1.csv), builds the
modelling table with stateless preprocessing only, runs nested leave-one-city-out
evaluation, and writes:

    evaluation/metrics.json              (committed)
    evaluation/heldout_predictions.csv   (committed)
    models/best_model.pkl                (deployment pipeline, not committed)
    data/raw_data.csv, data/processed_data.csv  (dashboard inputs, not committed)
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    BEST_MODEL_PATH, BOOTSTRAP_ITERATIONS, FEATURES_PATH, HELDOUT_PREDICTIONS_PATH,
    METRICS_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, SNAPSHOT_DATA_PATH,
)
from evaluation import (  # noqa: E402
    InsufficientGroupsError, MODEL_ORDER, recompute_from_predictions, run_full_evaluation,
)
from preprocessor import build_processed_table, candidate_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Nested leave-one-city-out UHI evaluation")
    ap.add_argument("--data", type=Path, default=SNAPSHOT_DATA_PATH,
                    help="raw collected CSV (default: committed snapshot)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ITERATIONS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--fast", action="store_true", help="reduced model grid")
    ap.add_argument("--no-deploy", action="store_true", help="skip deployment-model fit")
    ap.add_argument("--metrics", type=Path, default=METRICS_PATH)
    ap.add_argument("--predictions", type=Path, default=HELDOUT_PREDICTIONS_PATH)
    a = ap.parse_args()

    if not a.data.exists():
        print(f"Raw data not found: {a.data}", file=sys.stderr)
        return 2
    t0 = time.time()
    raw = pd.read_csv(a.data)
    processed = build_processed_table(raw)
    features = candidate_features()

    try:
        metrics, preds, _, deploy = run_full_evaluation(
            processed, features, seed=a.seed, n_boot=a.bootstrap, n_jobs=a.n_jobs,
            fast=a.fast, raw_path=a.data, predictions_path=a.predictions,
            metrics_path=a.metrics, fit_deployment=not a.no_deploy,
        )
    except InsufficientGroupsError as exc:
        print(f"Evaluation refused: {exc}", file=sys.stderr)
        return 3

    # Self-check: aggregates must be recomputable from the saved predictions
    saved = pd.read_csv(a.predictions)
    re = recompute_from_predictions(saved)
    for m in MODEL_ORDER:
        for k in ("macro_mae", "macro_rmse", "pooled_mae", "pooled_rmse"):
            assert abs(re[m][k] - metrics["models"][m][k]) < 1e-5, (m, k)

    # Dashboard inputs (gitignored)
    if a.data.resolve() != RAW_DATA_PATH.resolve():
        shutil.copyfile(a.data, RAW_DATA_PATH)
    processed.to_csv(PROCESSED_DATA_PATH, index=False)
    FEATURES_PATH.write_text(json.dumps(features))
    if deploy is not None:
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(deploy, f)

    print("\nModel                       macro MAE  macro RMSE  pooled MAE  pooled RMSE")
    for m in MODEL_ORDER:
        r = metrics["models"][m]
        print(f"{m:26s}  {r['macro_mae']:9.3f}  {r['macro_rmse']:10.3f}  "
              f"{r['pooled_mae']:10.3f}  {r['pooled_rmse']:11.3f}")
    ci = metrics["aggregate"]["confidence_intervals"]
    print(f"\nCI status: {ci['status']}")
    print(f"Wrote {a.metrics} and {a.predictions}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
UHI Prediction System - Model Training
Trains 7-9 regression models with:
  - GroupKFold train/test split  (Fix #3 — no city in both train and test)
  - GridSearchCV hyperparameter tuning with GroupKFold inner CV
  - MAE + RMSE + R²  (Fix #8)
  - Baseline comparison  (Fix #8)
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import (
    GroupShuffleSplit, GroupKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    PROCESSED_DATA_PATH, BEST_MODEL_PATH, METRICS_PATH, FEATURES_PATH, SCALER_PATH,
)
from logger import get_logger

warnings.filterwarnings("ignore")
log = get_logger("model_trainer")


# ─── Optional heavy libraries ─────────────────────────────────────────────────

def _get_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    except ImportError:
        return None


def _get_lightgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    except ImportError:
        return None


# ─── Model registry ───────────────────────────────────────────────────────────

def get_base_models() -> dict:
    models = {
        "Linear Regression":   LinearRegression(),
        "Ridge Regression":    Ridge(),
        "Lasso Regression":    Lasso(max_iter=5000),
        "Decision Tree":       DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest":       RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingRegressor(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }
    xgb = _get_xgboost()
    if xgb:
        models["XGBoost"] = xgb
        log.info("  XGBoost available")
    lgbm = _get_lightgbm()
    if lgbm:
        models["LightGBM"] = lgbm
        log.info("  LightGBM available")
    return models


# ─── GridSearch parameter grids ───────────────────────────────────────────────

PARAM_GRIDS = {
    "Ridge Regression":    {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression":    {"alpha": [0.001, 0.01, 0.1, 1.0]},
    "Decision Tree":       {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 2, 5]},
    "Random Forest":       {"n_estimators": [100, 200], "max_depth": [5, 8, None],
                            "min_samples_leaf": [1, 2]},
    "Gradient Boosting":   {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                            "max_depth": [3, 5]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "XGBoost":             {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                            "max_depth": [4, 6], "subsample": [0.8, 1.0]},
    "LightGBM":            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                            "max_depth": [4, 6]},
}


# ─── Feature importance ───────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> dict:
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
        else:
            return {}
        imp = imp / (imp.sum() + 1e-9)
        return dict(zip(feature_names, imp.round(5).tolist()))
    except Exception:
        return {}


# ─── Training pipeline ───────────────────────────────────────────────────────

def train(force: bool = False) -> dict:
    """
    Train all models, evaluate, pick the best, save artefacts.
    Returns the full metrics dict.
    """
    if METRICS_PATH.exists() and BEST_MODEL_PATH.exists() and not force:
        log.info("Model artefacts exist. Loading metrics ...")
        with open(METRICS_PATH) as f:
            return json.load(f)

    log.info("=" * 60)
    log.info("STEP 3 – MODEL TRAINING")
    log.info("=" * 60)

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    log.info(f"  Loaded {len(df)} rows")

    # ── Load feature names & group labels ─────────────────────────────────
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH) as f:
            feature_names = json.load(f)
    else:
        feature_names = [c for c in FEATURE_COLUMNS if c in df.columns]

    feature_names = [f for f in feature_names if f in df.columns]

    # city_name column is saved for GroupKFold but is NOT a model feature
    groups = df["city_name"].values if "city_name" in df.columns else None
    n_cities = len(np.unique(groups)) if groups is not None else "?"
    log.info(f"  Features: {len(feature_names)}  |  Samples: {len(df)}  |  Cities: {n_cities}")

    X = df[feature_names].values
    y = df[TARGET_COLUMN].values

    # ── Scale features ────────────────────────────────────────────────────
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    # ── GroupShuffleSplit: no city in both train and test (Fix #3) ─────────
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X_scaled, y, groups=groups))
        X_train, X_test       = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test       = y[train_idx],         y[test_idx]
        train_groups          = groups[train_idx]
        test_cities           = np.unique(groups[test_idx])
        train_cities          = np.unique(groups[train_idx])
        log.info(f"  Train cities ({len(train_cities)}): {list(train_cities)}")
        log.info(f"  Test  cities ({len(test_cities)}):  {list(test_cities)}")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        train_groups = None

    log.info(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # ── Baseline: always predict training mean (Fix #8) ───────────────────
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_mae  = float(mean_absolute_error(y_test, baseline_pred))
    baseline_r2   = float(r2_score(y_test, baseline_pred))
    log.info(
        f"  Baseline (mean predictor) → "
        f"RMSE={baseline_rmse:.4f}  MAE={baseline_mae:.4f}  R²={baseline_r2:.4f}"
    )

    # ── Train all models ──────────────────────────────────────────────────
    # Inner CV uses GroupKFold so tuning also respects city boundaries
    inner_cv = (
        GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(train_groups))))
        if train_groups is not None
        else CV_FOLDS
    )

    all_metrics = {}
    best_name   = None
    best_rmse   = float("inf")
    best_model  = None

    for name, base_model in get_base_models().items():
        log.info(f"  Training: {name} ...")
        try:
            # GridSearchCV with group-aware inner CV
            if name in PARAM_GRIDS:
                gs = GridSearchCV(
                    base_model,
                    PARAM_GRIDS[name],
                    cv=inner_cv,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    refit=True,
                )
                fit_kwargs = {"groups": train_groups} if train_groups is not None else {}
                gs.fit(X_train, y_train, **fit_kwargs)
                model = gs.best_estimator_
                best_params = gs.best_params_
                log.info(f"    Best params: {best_params}")
            else:
                model = base_model
                model.fit(X_train, y_train)
                best_params = {}

            y_pred = model.predict(X_test)
            rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae    = float(mean_absolute_error(y_test, y_pred))          # Fix #8
            r2     = float(r2_score(y_test, y_pred))

            # Outer GroupKFold CV on full scaled dataset
            outer_cv = (
                GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(groups))))
                if groups is not None else CV_FOLDS
            )
            cv_kw = {"groups": groups} if groups is not None else {}
            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=outer_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                **cv_kw,
            )
            cv_rmse = float(-cv_scores.mean())
            cv_std  = float(cv_scores.std())

            # Skill score vs baseline (> 0 means better than predicting the mean)
            skill = round(1.0 - (rmse / (baseline_rmse + 1e-9)), 4)

            all_metrics[name] = {
                "rmse":               round(rmse, 4),
                "mae":                round(mae, 4),
                "r2":                 round(r2, 4),
                "cv_rmse":            round(cv_rmse, 4),
                "cv_std":             round(cv_std, 4),
                "skill_vs_baseline":  skill,
                "best_params":        best_params,
                "feature_importance": get_feature_importance(model, feature_names),
            }

            log.info(
                f"    RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
                f"CV={cv_rmse:.4f}±{cv_std:.4f}  skill={skill:.3f}"
            )

            if rmse < best_rmse:
                best_rmse  = rmse
                best_name  = name
                best_model = model

        except Exception as e:
            log.error(f"    Failed: {name}: {e}")

    if best_model is None:
        raise RuntimeError("All models failed to train!")

    # ── Save best model ───────────────────────────────────────────────────
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    log.info(f"\n  Best model: {best_name}  RMSE={best_rmse:.4f}  "
             f"vs baseline RMSE={baseline_rmse:.4f}")

    # ── Save full metrics ─────────────────────────────────────────────────
    output = {
        "best_model":       best_name,
        "best_rmse":        round(best_rmse, 4),
        "feature_names":    feature_names,
        "baseline_rmse":    round(baseline_rmse, 4),
        "baseline_mae":     round(baseline_mae, 4),
        "train_cities":     list(train_cities) if groups is not None else [],
        "test_cities":      list(test_cities)  if groups is not None else [],
        "models":           all_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"  Saved metrics → {METRICS_PATH}")

    return output


if __name__ == "__main__":
    metrics = train(force=True)
    best = metrics["best_model"]
    print(f"\nBest: {best}  RMSE={metrics['best_rmse']}  "
          f"(baseline={metrics['baseline_rmse']})")

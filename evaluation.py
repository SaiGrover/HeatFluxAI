"""
HeatFluxAI — leakage-free evaluation

Nested leave-one-city-out (LOCO) evaluation of UHI-intensity models.

    Outer loop : LeaveOneGroupOut over real city identifiers.  The held-out
                 city is used exactly once, to score the already-selected and
                 refitted pipeline.  It never influences preprocessing,
                 hyper-parameters or the choice of algorithm.
    Inner loop : GroupKFold over the remaining cities (no city is split across
                 inner train/validation).  GridSearchCV jointly chooses the
                 algorithm family and its hyper-parameters by mean inner-fold
                 RMSE, then refits on all outer-training cities.

Every data-dependent preprocessing step (winsorisation, median imputation,
constant-feature removal, duplicate-feature removal, standardisation) is a
step of the sklearn Pipeline, so it is re-fitted on the training part of every
inner and outer fold.

Baselines evaluated on the *same* outer folds:
    mean_baseline           DummyRegressor(strategy="mean") — training-fold mean
    latitude_ndvi_baseline  LinearRegression on [lat, ndvi] (fold-local median
                            imputation)

Only rows whose target provenance is ``observed_modis_lst_difference`` are
used for training, tuning and scoring.  Signed (negative) targets are kept.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import (
    BOOTSTRAP_ITERATIONS, CV_FOLDS, GROUP_COLUMN, MIN_CITIES_FOR_CI,
    MIN_CITIES_FOR_NESTED_CV, OBSERVED_TARGET_PROVENANCE, TARGET_COLUMN,
    TARGET_PROVENANCE_COLUMN,
)

METHOD_NAME = "nested_leave_one_city_out"
METHOD_VERSION = "2.0.0"

MEAN_BASELINE = "mean_baseline"
LAT_NDVI_BASELINE = "latitude_ndvi_baseline"
SELECTED_MODEL = "nested_selected_model"
MODEL_ORDER = [MEAN_BASELINE, LAT_NDVI_BASELINE, SELECTED_MODEL]

WINSORIZE_COLUMNS = ("temperature", "humidity", "wind_speed", "pressure", "clouds", "ndvi")


class InsufficientGroupsError(ValueError):
    """Raised instead of silently falling back to row-wise splitting."""


# ══════════════════════════════════════════════════════════════════════════════
# Fold-local transformers
# ══════════════════════════════════════════════════════════════════════════════

def _names_of(X) -> list[str] | None:
    return [str(c) for c in X.columns] if hasattr(X, "columns") else None


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip selected columns to Tukey fences [Q1 − k·IQR, Q3 + k·IQR] learned
    from the training data of the current fold.  NaNs pass through."""

    def __init__(self, columns=WINSORIZE_COLUMNS, k: float = 1.5):
        self.columns = columns
        self.k = k

    def fit(self, X, y=None):
        names = _names_of(X)
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        A = np.asarray(X, dtype=float)
        self.n_features_in_ = A.shape[1]
        self.lower_ = np.full(A.shape[1], -np.inf)
        self.upper_ = np.full(A.shape[1], np.inf)
        targets = set(self.columns or ())
        for j in range(A.shape[1]):
            if names is None or names[j] not in targets:
                continue
            col = A[:, j][~np.isnan(A[:, j])]
            if col.size == 0:
                continue
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            self.lower_[j], self.upper_[j] = q1 - self.k * iqr, q3 + self.k * iqr
        return self

    def transform(self, X):
        A = np.asarray(X, dtype=float)
        return np.clip(A, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class DuplicateFeatureDropper(BaseEstimator, TransformerMixin):
    """Drop columns whose training-fold values are identical (within ``atol``)
    to an earlier column.  Only exact duplicates are removed — merely
    correlated, physically distinct variables are kept."""

    def __init__(self, atol: float = 1e-12):
        self.atol = atol

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.n_features_in_ = A.shape[1]
        keep: list[int] = []
        self.duplicate_of_: dict[int, int] = {}
        for j in range(A.shape[1]):
            match = next(
                (k for k in keep
                 if np.allclose(A[:, j], A[:, k], rtol=0.0, atol=self.atol, equal_nan=True)),
                None,
            )
            if match is None:
                keep.append(j)
            else:
                self.duplicate_of_[j] = match
        self.support_ = np.zeros(A.shape[1], dtype=bool)
        self.support_[keep] = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)[self.support_]


def make_preprocessor() -> Pipeline:
    """Data-dependent preprocessing; fitted only on a fold's training rows."""
    return Pipeline([
        ("winsorize", Winsorizer()),
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold(threshold=1e-12)),
        ("drop_duplicate", DuplicateFeatureDropper()),
        ("scale", StandardScaler()),
    ])


def make_model_pipeline(estimator=None) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", estimator if estimator is not None else LinearRegression()),
    ])


def make_mean_baseline() -> Pipeline:
    return Pipeline([("model", DummyRegressor(strategy="mean"))])


def make_lat_ndvi_baseline() -> Pipeline:
    return Pipeline([
        ("select", ColumnTransformer([("lat_ndvi", "passthrough", ["lat", "ndvi"])])),
        ("impute", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])


def retained_features(fitted_pipeline: Pipeline, input_features: list[str]) -> list[str]:
    pre = fitted_pipeline.named_steps.get("preprocess")
    if pre is None:
        return list(input_features)
    return [str(f) for f in pre.get_feature_names_out(input_features)]


# ══════════════════════════════════════════════════════════════════════════════
# Candidate search space
# ══════════════════════════════════════════════════════════════════════════════

def _xgb(seed):
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=1, verbosity=0, tree_method="hist")
    except ImportError:
        return None


def _lgbm(seed):
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, deterministic=True,
                             force_row_wise=True)
    except ImportError:
        return None


def candidate_param_grid(seed: int, fast: bool = False) -> list[dict]:
    """GridSearchCV grid over the ``model`` step: algorithm + hyper-parameters."""
    grid = [
        {"model": [LinearRegression()]},
        {"model": [Ridge()], "model__alpha": [0.1, 1.0, 10.0, 100.0]},
        {"model": [Lasso(max_iter=20000)], "model__alpha": [0.001, 0.01, 0.1]},
        {"model": [DecisionTreeRegressor(random_state=seed)],
         "model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 20]},
        {"model": [RandomForestRegressor(random_state=seed, n_estimators=200, n_jobs=1)],
         "model__max_depth": [5, 10, None], "model__min_samples_leaf": [2, 10]},
        {"model": [GradientBoostingRegressor(random_state=seed)],
         "model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [3]},
    ]
    xgb, lgbm = _xgb(seed), _lgbm(seed)
    if xgb is not None:
        grid.append({"model": [xgb], "model__n_estimators": [200],
                     "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5],
                     "model__subsample": [0.8]})
    if lgbm is not None:
        grid.append({"model": [lgbm], "model__n_estimators": [200],
                     "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 5],
                     "model__num_leaves": [15]})
    if fast:  # cloud-safe reduced grid (dashboard live refresh)
        grid = [
            {"model": [Ridge()], "model__alpha": [1.0, 10.0]},
            {"model": [RandomForestRegressor(random_state=seed, n_estimators=100, n_jobs=1)],
             "model__max_depth": [8], "model__min_samples_leaf": [5]},
        ]
    return grid


def describe_grid(grid: list[dict]) -> list[dict]:
    out = []
    for g in grid:
        est = g["model"][0]
        out.append({
            "family": type(est).__name__,
            "fixed_params": {k: v for k, v in est.get_params().items()
                             if k in ("random_state", "n_estimators", "max_iter")
                             and not any(p == f"model__{k}" for p in g)},
            "searched": {k.replace("model__", ""): v for k, v in g.items() if k != "model"},
        })
    return out


def _family(est) -> str:
    return type(est).__name__


def _params_of(best_params: dict) -> dict:
    out = {}
    for k, v in best_params.items():
        if k == "model":
            continue
        out[k.replace("model__", "")] = v if isinstance(v, (int, float, str, type(None))) else str(v)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Data eligibility and group validation
# ══════════════════════════════════════════════════════════════════════════════

def eligible_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rows usable for supervised learning + exclusion counts with reasons."""
    prov = df.get(TARGET_PROVENANCE_COLUMN)
    if prov is None:
        raise ValueError(
            f"Column '{TARGET_PROVENANCE_COLUMN}' missing: target provenance cannot be "
            "established, so no score can be presented as observed-target performance."
        )
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    observed = prov.eq(OBSERVED_TARGET_PROVENANCE)
    ok = observed & y.notna() & np.isfinite(y)
    reasons = {}
    for reason, mask in [
        ("target_not_observed:" + "|".join(sorted(prov[~observed].astype(str).unique())),
         ~observed),
        ("observed_provenance_but_target_missing", observed & ~(y.notna() & np.isfinite(y))),
    ]:
        n = int(mask.sum())
        if n:
            reasons[reason] = n
    return df.loc[ok].copy(), {"n_excluded": int((~ok).sum()), "reasons": reasons}


def validate_groups(df: pd.DataFrame, group_col: str = GROUP_COLUMN) -> dict:
    """Ensure groups are real geographic cities, not row-level surrogates."""
    if group_col not in df.columns:
        raise InsufficientGroupsError(
            f"No '{group_col}' column: cannot run leave-one-city-out evaluation. "
            "Refusing to fall back to random row splits."
        )
    g = df[group_col]
    if g.isna().any():
        raise InsufficientGroupsError("Some rows have no city identifier.")
    n_groups, n_rows = g.nunique(), len(df)
    if n_groups < MIN_CITIES_FOR_NESTED_CV:
        raise InsufficientGroupsError(
            f"Only {n_groups} distinct cities with observed targets; nested "
            f"leave-one-city-out CV needs ≥ {MIN_CITIES_FOR_NESTED_CV} (1 held-out "
            "city + ≥ 3 cities for inner grouped CV). Not falling back to row splits."
        )
    if n_groups > n_rows / 2:
        raise InsufficientGroupsError(
            f"{n_groups} groups for {n_rows} rows looks like a row-level identifier, "
            "not a city. Refusing to present this as leave-one-city-out evaluation."
        )
    if {"lat", "lon"} <= set(df.columns):
        coords = df.groupby(group_col)[["lat", "lon"]].nunique()
        multi = coords[(coords["lat"] > 1) | (coords["lon"] > 1)].index.tolist()
        if multi:
            raise InsufficientGroupsError(
                f"Cities with more than one coordinate pair: {multi[:5]} — the "
                "group column does not identify a single city."
            )
        shared = df.groupby(["lat", "lon"])[group_col].nunique()
        if (shared > 1).any():
            raise InsufficientGroupsError("One coordinate pair maps to several city names.")
    return {"n_groups": int(n_groups), "rows_per_group_min": int(g.value_counts().min()),
            "rows_per_group_max": int(g.value_counts().max())}


# ══════════════════════════════════════════════════════════════════════════════
# Nested leave-one-city-out
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NestedResult:
    predictions: pd.DataFrame
    folds: list[dict]
    features: list[str]
    grid: list[dict]
    fitted: dict = field(default_factory=dict)   # (city, model) → fitted pipeline


def inner_cv_for(n_train_groups: int) -> GroupKFold:
    if n_train_groups < 3:
        raise InsufficientGroupsError(
            f"Only {n_train_groups} training cities in an outer fold; inner grouped "
            "CV needs ≥ 3."
        )
    return GroupKFold(n_splits=min(CV_FOLDS, n_train_groups))


def nested_loco(
    df: pd.DataFrame,
    features: list[str],
    seed: int = 42,
    param_grid: list[dict] | None = None,
    n_jobs: int = -1,
    keep_fitted: bool = False,
    on_fold_start: Callable[[int, str], None] | None = None,
    fast: bool = False,
    log: Callable[[str], None] = print,
) -> NestedResult:
    """Run nested LOCO on *eligible* rows (call ``eligible_rows`` first)."""
    df = df.reset_index(drop=True)
    validate_groups(df)
    X = df[features]
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    groups = df[GROUP_COLUMN].to_numpy()
    grid = param_grid if param_grid is not None else candidate_param_grid(seed, fast=fast)

    pred_frames, folds, fitted = [], [], {}
    outer = LeaveOneGroupOut()
    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        city = str(groups[te][0])
        assert set(groups[te]) == {city}, "outer test fold must be exactly one city"
        assert city not in set(groups[tr]), "held-out city leaked into outer training"
        if on_fold_start:
            on_fold_start(fold, city)

        Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
        Xte, yte = X.iloc[te], y[te]
        inner = inner_cv_for(len(np.unique(gtr)))

        gs = GridSearchCV(
            make_model_pipeline(), grid, cv=inner,
            scoring="neg_root_mean_squared_error", n_jobs=n_jobs,
            refit=True, error_score="raise",
        )
        gs.fit(Xtr, ytr, groups=gtr)
        selected = gs.best_estimator_

        mean_bl = clone(make_mean_baseline()).fit(Xtr, ytr)
        latndvi = clone(make_lat_ndvi_baseline()).fit(Xtr, ytr)

        sel_family = _family(selected.named_steps["model"])
        fold_info = {
            "outer_fold": fold,
            "held_out_city": city,
            "n_train": int(len(tr)),
            "n_train_cities": int(len(np.unique(gtr))),
            "n_test": int(len(te)),
            "inner_cv": f"GroupKFold(n_splits={inner.get_n_splits()})",
            "selected_family": sel_family,
            "selected_params": _params_of(gs.best_params_),
            "inner_cv_rmse": float(-gs.best_score_),
            "retained_features": retained_features(selected, features),
            "train_target_mean": float(ytr.mean()),
            "lat_ndvi_coefficients": dict(zip(
                ["lat", "ndvi"], latndvi.named_steps["model"].coef_.round(6).tolist())),
        }
        folds.append(fold_info)
        log(f"  fold {fold:2d}  held-out {city:<14s} n={len(te):4d}  "
            f"selected {sel_family} {fold_info['selected_params']}  "
            f"inner RMSE {fold_info['inner_cv_rmse']:.3f}")

        for name, est in [(MEAN_BASELINE, mean_bl), (LAT_NDVI_BASELINE, latndvi),
                          (SELECTED_MODEL, selected)]:
            p = est.predict(Xte)
            pred_frames.append(pd.DataFrame({
                "observation_id": df["observation_id"].iloc[te].to_numpy(),
                "city": city,
                "date": df["date"].iloc[te].to_numpy() if "date" in df else None,
                "outer_fold": fold,
                "model_name": name,
                "estimator": sel_family if name == SELECTED_MODEL else
                             ("DummyRegressor(mean)" if name == MEAN_BASELINE
                              else "LinearRegression[lat,ndvi]"),
                "observed_uhi": yte,
                "predicted_uhi": p,
                "target_provenance": df[TARGET_PROVENANCE_COLUMN].iloc[te].to_numpy(),
                "lst_date": df["lst_date"].iloc[te].to_numpy() if "lst_date" in df else None,
            }))
            if keep_fitted:
                fitted[(city, name)] = est

    preds = pd.concat(pred_frames, ignore_index=True)
    preds["residual"] = preds["observed_uhi"] - preds["predicted_uhi"]
    preds["absolute_error"] = preds["residual"].abs()
    preds["squared_error"] = preds["residual"] ** 2
    preds["model_name"] = pd.Categorical(preds["model_name"], MODEL_ORDER, ordered=True)
    preds = preds.sort_values(["model_name", "city", "date", "observation_id"]).reset_index(drop=True)
    preds["model_name"] = preds["model_name"].astype(str)
    return NestedResult(preds, folds, list(features), grid, fitted)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def per_city_metrics(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = preds[preds["model_name"] == model]
    rows = []
    for city, g in sub.groupby("city", sort=True):
        rows.append({
            "city": city, "n": int(len(g)),
            "mae": mae(g["observed_uhi"], g["predicted_uhi"]),
            "rmse": rmse(g["observed_uhi"], g["predicted_uhi"]),
            "bias_mean_residual": float(g["residual"].mean()),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(preds: pd.DataFrame, model: str) -> dict:
    """macro_* = unweighted mean over cities (each city counts once);
    pooled_* = computed over all observations (large cities weigh more)."""
    pc = per_city_metrics(preds, model)
    sub = preds[preds["model_name"] == model]
    return {
        "n_cities": int(len(pc)),
        "n_observations": int(len(sub)),
        "macro_mae": float(pc["mae"].mean()),
        "macro_rmse": float(pc["rmse"].mean()),
        "pooled_mae": mae(sub["observed_uhi"], sub["predicted_uhi"]),
        "pooled_rmse": rmse(sub["observed_uhi"], sub["predicted_uhi"]),
    }


def city_bootstrap(preds: pd.DataFrame, models: list[str], n_boot: int, seed: int,
                   level: float = 0.95) -> dict:
    """
    Percentile CIs from resampling *cities* with replacement (each sampled city
    brings all of its held-out observations).  The same city draws are used for
    every model, so paired differences are also reported.
    """
    cities = sorted(preds["city"].unique())
    C = len(cities)
    stats = {}
    for m in models:
        pc = per_city_metrics(preds, m).set_index("city").loc[cities]
        sub = preds[preds["model_name"] == m].groupby("city")
        stats[m] = {
            "city_mae": pc["mae"].to_numpy(), "city_rmse": pc["rmse"].to_numpy(),
            "sae": sub["absolute_error"].sum().loc[cities].to_numpy(),
            "sse": sub["squared_error"].sum().loc[cities].to_numpy(),
            "n": sub.size().loc[cities].to_numpy().astype(float),
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, C, size=(n_boot, C))
    draws = {}
    for m, s in stats.items():
        n = s["n"][idx].sum(axis=1)
        draws[m] = {
            "macro_mae": s["city_mae"][idx].mean(axis=1),
            "macro_rmse": s["city_rmse"][idx].mean(axis=1),
            "pooled_mae": s["sae"][idx].sum(axis=1) / n,
            "pooled_rmse": np.sqrt(s["sse"][idx].sum(axis=1) / n),
        }
    a = (1 - level) / 2 * 100

    def ci(v):
        lo, hi = np.percentile(v, [a, 100 - a])
        return [float(lo), float(hi)]

    out = {m: {k: ci(v) for k, v in d.items()} for m, d in draws.items()}
    diffs = {}
    for m in models:
        if m == SELECTED_MODEL or SELECTED_MODEL not in draws:
            continue
        key = f"{SELECTED_MODEL} - {m}"
        diffs[key] = {k: ci(draws[SELECTED_MODEL][k] - draws[m][k]) for k in draws[m]}
    return {"intervals": out, "paired_differences": diffs}


# ══════════════════════════════════════════════════════════════════════════════
# Deployment model (separate from evaluation)
# ══════════════════════════════════════════════════════════════════════════════

def fit_deployment_model(df: pd.DataFrame, features: list[str], seed: int,
                         n_jobs: int = -1, fast: bool = False):
    """
    Choose and fit the model the dashboard uses for predictions.  Selection uses
    grouped CV over ALL eligible cities; its CV score is a *selection* score
    (optimistically biased by the selection itself) and is not a test estimate.
    The unbiased estimate of this procedure is the nested LOCO result.
    """
    df = df.reset_index(drop=True)
    groups = df[GROUP_COLUMN].to_numpy()
    cv = GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(groups))))
    gs = GridSearchCV(make_model_pipeline(), candidate_param_grid(seed, fast=fast), cv=cv,
                      scoring="neg_root_mean_squared_error", n_jobs=n_jobs,
                      refit=True, error_score="raise")
    gs.fit(df[features], df[TARGET_COLUMN].to_numpy(float), groups=groups)
    best = gs.best_estimator_
    kept = retained_features(best, features)
    model = best.named_steps["model"]
    imp = {}
    if hasattr(model, "feature_importances_"):
        v = np.asarray(model.feature_importances_, float)
    elif hasattr(model, "coef_"):
        v = np.abs(np.asarray(model.coef_, float)).ravel()
    else:
        v = None
    if v is not None and v.sum() > 0:
        imp = {f: round(float(x / v.sum()), 6) for f, x in zip(kept, v)}
    info = {
        "selection_procedure": (
            f"GridSearchCV over the same candidate grid, GroupKFold(n_splits={cv.get_n_splits()}) "
            "on all eligible cities, selection metric = mean inner-fold RMSE; refit on all "
            "eligible rows"),
        "selected_family": _family(model),
        "selected_params": _params_of(gs.best_params_),
        "selection_cv_rmse": float(-gs.best_score_),
        "selection_cv_rmse_note": (
            "Validation score of the winning configuration during selection. It is "
            "optimistically biased by the selection and is NOT a held-out test estimate; "
            "use the nested leave-one-city-out results for generalisation."),
        "n_train_observations": int(len(df)),
        "n_train_cities": int(len(np.unique(groups))),
        "input_features": list(features),
        "retained_features": kept,
        "feature_importance": imp,
    }
    return best, info


# ══════════════════════════════════════════════════════════════════════════════
# Full run → artifacts
# ══════════════════════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, float_format="%.6f").encode()).hexdigest()


def _round(obj, nd=6):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def software_versions() -> dict:
    import sklearn
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    for mod in ("xgboost", "lightgbm", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def code_fingerprint() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in ("evaluation.py", "preprocessor.py", "config.py"):
        h.update((here / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


LIMITATIONS = [
    "25 cities is a small number of independent geographic units; city-bootstrap "
    "intervals are wide and assume the cities are exchangeable draws from a wider "
    "population of cities, which a hand-picked list of megacities is not.",
    "One 180-day window (2025-01-01 to 2025-06-29): results may not transfer to other "
    "seasons, years, or to the opposite hemisphere's summer.",
    "The target is a MODIS daytime land-SURFACE temperature difference (urban pixel vs "
    "median of 12 rural ring points at 55–110 km), not an air-temperature UHI. Ring "
    "points can include water, mountains or other urban areas.",
    "Each 8-day LST composite is shared by up to 8 consecutive daily weather rows, so "
    "observations within a city are strongly autocorrelated; the effective sample size "
    "is the number of (city, composite) pairs, not the number of rows.",
    "Static city attributes (lat, lon, NDVI, veg class) are constant within a city, so "
    "they only identify cities; cross-city generalisation from 24 training cities is "
    "inherently limited.",
    "temp_anomaly is computed from each city's own weather over the window (no target "
    "information, no cross-city information), which is transductive for the held-out "
    "city.",
    "Leave-one-city-out does not remove spatial dependence between nearby or climatically "
    "similar cities (e.g. Delhi/Karachi, Kolkata/Dhaka) or the shared time window.",
    "The published snapshot uses the Planetary Computer MODIS mirror (MOD13A1 500 m NDVI "
    "instead of MOD13A2 1 km); the GEE collector produces a different dataset and must be "
    "re-evaluated before its numbers are quoted.",
]


def run_full_evaluation(
    processed: pd.DataFrame,
    features: list[str],
    seed: int = 42,
    n_boot: int = BOOTSTRAP_ITERATIONS,
    n_jobs: int = -1,
    fast: bool = False,
    raw_path: Path | None = None,
    predictions_path: Path | None = None,
    metrics_path: Path | None = None,
    fit_deployment: bool = True,
    log: Callable[[str], None] = print,
):
    eligible, excl = eligible_rows(processed)
    groups_info = validate_groups(eligible)
    log(f"Eligible observed targets: {len(eligible)}  excluded: {excl['n_excluded']}  "
        f"cities: {groups_info['n_groups']}")

    res = nested_loco(eligible, features, seed=seed, n_jobs=n_jobs, fast=fast, log=log)
    preds = res.predictions

    aggregates = {m: aggregate_metrics(preds, m) for m in MODEL_ORDER}
    per_city = {m: per_city_metrics(preds, m).set_index("city") for m in MODEL_ORDER}
    n_cities = groups_info["n_groups"]
    if n_cities < MIN_CITIES_FOR_CI:
        ci_block = {"status": "unavailable",
                    "reason": f"only {n_cities} cities (< {MIN_CITIES_FOR_CI}); "
                              "city-level bootstrap would be unstable"}
    else:
        boot = city_bootstrap(preds, MODEL_ORDER, n_boot, seed)
        ci_block = {
            "status": "ok" if n_cities >= 20 else "unstable_small_number_of_cities",
            "method": "percentile city-level bootstrap: resample cities with replacement, "
                      "each sampled city keeps all its held-out observations; macro and "
                      "pooled metrics recomputed per resample; same draws for all models "
                      "(paired differences)",
            "unit_of_resampling": "city",
            "n_bootstrap": int(n_boot), "seed": int(seed), "level": 0.95,
            "intervals": boot["intervals"],
            "paired_differences": boot["paired_differences"],
            "interpretation": "Descriptive uncertainty over cities; not a significance test.",
        }

    fam_counts = pd.Series([f["selected_family"] for f in res.folds]).value_counts()
    all_retained = [set(f["retained_features"]) for f in res.folds]
    always = sorted(set.intersection(*all_retained), key=features.index)
    sometimes = sorted(set.union(*all_retained) - set(always), key=features.index)
    never = [f for f in features if f not in set.union(*all_retained)]

    per_city_list = []
    for city in sorted(per_city[SELECTED_MODEL].index):
        fold = next(f for f in res.folds if f["held_out_city"] == city)
        per_city_list.append({
            "city": city,
            "outer_fold": fold["outer_fold"],
            "n_evaluated": int(per_city[SELECTED_MODEL].loc[city, "n"]),
            "selected_family": fold["selected_family"],
            "selected_params": fold["selected_params"],
            "inner_cv_rmse": fold["inner_cv_rmse"],
            "metrics": {m: {"mae": float(per_city[m].loc[city, "mae"]),
                            "rmse": float(per_city[m].loc[city, "rmse"]),
                            "bias_mean_residual": float(per_city[m].loc[city, "bias_mean_residual"])}
                        for m in MODEL_ORDER},
        })

    y_ev = eligible[TARGET_COLUMN]
    metrics = {
        "schema_version": "2.0",
        "evaluation": {
            "method": METHOD_NAME, "version": METHOD_VERSION,
            "group_column": GROUP_COLUMN,
            "outer_cv": "LeaveOneGroupOut (one city held out per fold)",
            "inner_cv": "GroupKFold over outer-training cities, n_splits = min(5, #training cities)",
            "selection_metric": "rmse",
            "selection_rule": "GridSearchCV(scoring='neg_root_mean_squared_error'): highest mean "
                              "inner-fold score (lowest mean inner RMSE) over algorithm × "
                              "hyper-parameters; ties → first in grid order; refit on all "
                              "outer-training cities",
            "test_data_used_for_selection": False,
            "preprocessing": "fold-local Pipeline: Winsorizer(IQR k=1.5 on weather+ndvi) → "
                             "SimpleImputer(median) → VarianceThreshold(1e-12) → "
                             "DuplicateFeatureDropper(exact) → StandardScaler",
            "random_seed": int(seed),
            "fast_grid": bool(fast),
            "code_fingerprint_sha256": code_fingerprint(),
        },
        "dataset": {
            "raw_file": str(raw_path.name) if raw_path else None,
            "raw_file_sha256": sha256_file(raw_path) if raw_path and raw_path.exists() else None,
            "eligible_table_fingerprint_sha256": frame_fingerprint(
                eligible[["observation_id", GROUP_COLUMN, TARGET_COLUMN] + features]),
            "n_rows_total": int(len(processed)),
            "n_cities": int(n_cities),
            "cities": sorted(eligible[GROUP_COLUMN].unique().tolist()),
            "n_evaluated_observations": int(len(eligible)),
            "n_excluded_targets": excl["n_excluded"],
            "excluded_target_reasons": excl["reasons"],
            "n_unique_city_lst_composites": int(
                eligible[[GROUP_COLUMN, "lst_date"]].drop_duplicates().shape[0])
                if "lst_date" in eligible else None,
            "date_range": [str(eligible["date"].min()), str(eligible["date"].max())]
                if "date" in eligible else None,
            "target_summary": {
                "mean": float(y_ev.mean()), "std": float(y_ev.std()),
                "min": float(y_ev.min()), "max": float(y_ev.max()),
                "n_negative": int((y_ev < 0).sum()),
                "fraction_negative": float((y_ev < 0).mean()),
            },
        },
        "target_policy": {
            "definition": "uhi_intensity = urban_LST − median(rural ring LST), °C, MODIS "
                          "MOD11A2 LST_Day_1km 8-day composite covering the observation date",
            "eligible_provenance": [OBSERVED_TARGET_PROVENANCE],
            "imputed_targets": "none — rows without an observed LST pair are excluded from "
                               "training, tuning and scoring",
            "negative_values": "kept (urban surface cooler than rural reference)",
            "clipping": "none",
        },
        "features": {
            "candidates": list(features),
            "definitional_drops": {"urban_fraction": "defined as clip(1 − ndvi, 0, 1); a "
                                   "re-expression of NDVI, dropped before modelling"},
            "fold_local_selection": "constant (variance ≤ 1e-12) and exact-duplicate columns "
                                    "are detected on each fold's training rows only",
            "retained_in_all_outer_folds": always,
            "retained_in_some_outer_folds": sometimes,
            "dropped_in_all_outer_folds": never,
        },
        "models": {
            MEAN_BASELINE: {"description": "DummyRegressor(strategy='mean'): predicts the "
                            "outer-training-fold mean UHI", **aggregates[MEAN_BASELINE]},
            LAT_NDVI_BASELINE: {"description": "LinearRegression on [lat (signed), ndvi] with "
                                "fold-local median imputation", "features": ["lat", "ndvi"],
                                **aggregates[LAT_NDVI_BASELINE]},
            SELECTED_MODEL: {"description": "per outer fold, the algorithm + hyper-parameters "
                             "chosen by inner grouped CV RMSE, refit on all outer-training "
                             "cities", "search_space": describe_grid(res.grid),
                             "family_selection_counts": {str(k): int(v) for k, v in fam_counts.items()},
                             **aggregates[SELECTED_MODEL]},
        },
        "per_city": per_city_list,
        "aggregate": {
            "definitions": {
                "macro": "unweighted mean of per-city metric (each held-out city counts once)",
                "pooled": "metric over all held-out observations (cities weighted by #obs)",
            },
            **{m: {k: aggregates[m][k] for k in
                   ("macro_mae", "macro_rmse", "pooled_mae", "pooled_rmse")} for m in MODEL_ORDER},
            "confidence_intervals": ci_block,
        },
        "outer_folds": [{k: v for k, v in f.items() if k != "retained_features"}
                        for f in res.folds],
        "software": software_versions(),
        "limitations": LIMITATIONS,
    }

    deploy_model = None
    if fit_deployment:
        deploy_model, deploy_info = fit_deployment_model(eligible, features, seed,
                                                         n_jobs=n_jobs, fast=fast)
        metrics["deployment_model"] = deploy_info
        log(f"Deployment model: {deploy_info['selected_family']} "
            f"{deploy_info['selected_params']}  selection-CV RMSE "
            f"{deploy_info['selection_cv_rmse']:.3f}")

    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(predictions_path, index=False, float_format="%.6f",
                     lineterminator="\n")
        metrics["artifacts"] = {
            "heldout_predictions": predictions_path.name,
            "heldout_predictions_sha256": sha256_file(predictions_path),
            "heldout_predictions_columns": list(preds.columns),
        }
    metrics = _round(metrics)
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8", newline="\n")
    return metrics, preds, res, deploy_model


def recompute_from_predictions(preds: pd.DataFrame) -> dict:
    """Independent recomputation used by tests / verification."""
    return {m: aggregate_metrics(preds, m) for m in preds["model_name"].unique()}

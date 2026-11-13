"""
UHI Prediction System - Data Preprocessing

Turns raw collected rows into the modelling table.  Only *stateless*,
row-wise or definitional operations happen here; every data-dependent step
(imputation, outlier capping, scaling, constant/duplicate-feature removal) is
learned inside each cross-validation fold by the pipeline in ``evaluation.py``.

What this module does NOT do any more (and why):
  * no global median imputation of features      → fold-local SimpleImputer
  * no global IQR winsorisation                   → fold-local Winsorizer
  * no global StandardScaler / scaler.pkl          → fold-local StandardScaler
  * no imputation of the UHI target                → rows without an observed
    LST pair keep target = NaN and provenance "missing_lst"
  * no clipping of the UHI target at 0             → negative UHI (urban
    surface cooler than its rural reference) is a valid observation
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS, TARGET_COLUMN, GROUP_COLUMN,
    TARGET_PROVENANCE_COLUMN, OBSERVED_TARGET_PROVENANCE,
    DEFINITIONAL_DUPLICATE_FEATURES,
    RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURES_PATH, MODEL_DIR,
)
from logger import get_logger

log = get_logger("preprocessor")

# Columns described by the (descriptive-only) IQR outlier report
_OUTLIER_CAP_COLS = ["temperature", "humidity", "wind_speed", "pressure", "clouds", "ndvi"]
OUTLIER_STATS_PATH = MODEL_DIR / "outlier_stats.json"

EXTRA_FEATURES = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "lat_abs", "lon_sin", "lon_cos", "heat_index",
]

MISSING_LST_PROVENANCE = "missing_lst_no_covering_composite"


def candidate_features() -> list[str]:
    """Candidate model inputs, before fold-local constant/duplicate removal."""
    seen, out = set(), []
    for c in FEATURE_COLUMNS + EXTRA_FEATURES:
        if c in DEFINITIONAL_DUPLICATE_FEATURES or c in seen:
            continue
        out.append(c)
        seen.add(c)
    return out


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ─── Descriptive IQR outlier report (NOT applied to the data) ─────────────────

def describe_outliers(df: pd.DataFrame) -> dict:
    """
    Tukey-fence (Q1 − 1.5·IQR, Q3 + 1.5·IQR) counts per column, for display.
    Capping itself is done fold-locally by evaluation.Winsorizer so that
    held-out cities never influence the fences.
    """
    stats = {}
    for col in [c for c in _OUTLIER_CAP_COLS if c in df.columns]:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_low, n_high = int((s < lo).sum()), int((s > hi).sum())
        stats[col] = {
            "q1": round(float(q1), 3), "q3": round(float(q3), 3),
            "iqr": round(float(iqr), 3),
            "lower_fence": round(float(lo), 3), "upper_fence": round(float(hi), 3),
            "n_low": n_low, "n_high": n_high, "n_capped": n_low + n_high,
            "pct": round(100 * (n_low + n_high) / max(len(s), 1), 2),
            "note": "descriptive, full dataset; capping is applied fold-locally",
        }
    return stats


# Backward-compatible name used by older code/notebooks
def cap_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deprecated: returns the data UNCHANGED plus descriptive stats."""
    return df.copy(), describe_outliers(df)


# ─── UHI target (observed only, signed) ───────────────────────────────────────

def compute_uhi_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    UHI intensity (°C) = urban_LST − rural_LST, signed.

    Rows with both LST values get the satellite-derived target and provenance
    ``observed_modis_lst_difference``.  All other rows keep target = NaN and
    provenance ``missing_lst_no_covering_composite``; they remain available
    for mapping/inference but are excluded from training, tuning and scoring.
    Negative values are kept: they mean the urban surface was cooler than the
    rural reference ring in that composite.
    """
    df = df.copy()
    if "urban_lst" not in df.columns or "rural_lst" not in df.columns:
        raise ValueError("urban_lst and rural_lst columns are required.")

    urban = pd.to_numeric(df["urban_lst"], errors="coerce")
    rural = pd.to_numeric(df["rural_lst"], errors="coerce")
    valid = urban.notna() & rural.notna()

    df[TARGET_COLUMN] = np.where(valid, (urban - rural).round(3), np.nan)
    df[TARGET_PROVENANCE_COLUMN] = np.where(
        valid, OBSERVED_TARGET_PROVENANCE, MISSING_LST_PROVENANCE
    )
    obs = df.loc[valid, TARGET_COLUMN]
    log.info(
        f"  Target: {int(valid.sum())} observed, {int((~valid).sum())} without LST "
        f"(left missing).  Observed range [{obs.min():.2f}, {obs.max():.2f}] °C, "
        f"{int((obs < 0).sum())} negative values kept"
        if len(obs) else "  Target: no observed LST pairs"
    )
    return df


# ─── Feature engineering (row-wise / within-city, no target use) ──────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features.  Missing inputs propagate as NaN."""
    df = df.copy()

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = ts.dt.hour.astype(float)
        df["month"] = ts.dt.month.astype(float)
    else:
        df["hour"] = np.nan
        df["month"] = np.nan

    df["is_daytime"] = np.where(df["hour"].isna(), np.nan,
                                ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(float))
    df["is_night"] = np.where(df["hour"].isna(), np.nan,
                              ((df["hour"] < 6) | (df["hour"] > 18)).astype(float))

    ndvi = df["ndvi"].clip(-1, 1)
    df["veg_class"] = pd.cut(ndvi, bins=[-1.0, 0.2, 0.5, 1.0], labels=[0, 1, 2],
                             include_lowest=True).astype(float)

    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100
    df["wind_cooling_effect"] = df["wind_speed"] * (df["temperature"] - 20).clip(lower=0)
    df["heat_retention"] = (
        df["urban_fraction"] * df["temperature"] / (df["wind_speed"] + 1)
    ).round(3)

    # Within-city temperature anomaly: uses only that city's own *weather*
    # (never the target, never other cities).  For a held-out city it is
    # computed from that city's own feature rows — see docs on limitations.
    if "name" in df.columns:
        df["temp_anomaly"] = (
            df["temperature"] - df.groupby("name")["temperature"].transform("mean")
        ).round(3)
    else:
        df["temp_anomaly"] = np.nan

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["lat_abs"] = df["lat"].abs()
    df["lon_sin"] = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"] = np.cos(np.deg2rad(df["lon"]))
    df["distance_from_equator"] = df["lat"].abs()

    df["heat_index"] = (
        -8.78469475556
        + 1.61139411 * df["temperature"]
        + 2.33854883889 * df["humidity"] / 100
        - 0.14611605 * df["temperature"] * df["humidity"] / 100
    ).round(3)
    return df


# ─── Full preprocessing ───────────────────────────────────────────────────────

_PHYSICAL_RANGES = {
    "temperature": (-60, 60), "humidity": (0, 100), "wind_speed": (0, 100),
    "lat": (-90, 90), "lon": (-180, 180),
}


def build_processed_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function: raw rows → modelling table (all rows, target may be NaN)."""
    df = df.copy()
    if "source" in df.columns:
        n0 = len(df)
        df = df[df["source"].astype(str).str.contains("openweather|openmeteo", na=False)]
        log.info(f"  Source filter: {n0} → {len(df)} rows")
    if df.empty:
        raise ValueError("No real-data rows after source filter. Re-run collection.")

    for col in ["temperature", "humidity", "wind_speed", "pressure", "clouds",
                "ndvi", "urban_fraction", "lat", "lon", "urban_lst", "rural_lst"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stateless validity rule: drop rows whose *present* value is physically
    # impossible; missing values are kept for fold-local imputation.
    n0 = len(df)
    for col, (lo, hi) in _PHYSICAL_RANGES.items():
        df = df[df[col].isna() | df[col].between(lo, hi)]
    log.info(f"  Removed {n0 - len(df)} physically invalid rows")

    if df["name"].isna().any():
        raise ValueError("Every row needs a city 'name' (used as CV group).")

    df = engineer_features(df)
    df = compute_uhi_intensity(df)

    feats = candidate_features()
    ts = pd.to_datetime(df.get("timestamp"), errors="coerce")
    out = pd.DataFrame({
        "observation_id": [f"{slugify(n)}_{d}" for n, d in
                           zip(df["name"], ts.dt.strftime("%Y-%m-%d"))],
        GROUP_COLUMN: df["name"].values,
        "date": ts.dt.strftime("%Y-%m-%d").values,
        "lst_date": df["lst_date"].values if "lst_date" in df.columns else None,
    }, index=df.index)
    for c in feats:
        out[c] = df[c].values if c in df.columns else np.nan
    for c in ["urban_fraction", "urban_lst", "rural_lst"]:
        out[c] = df[c].values
    out[TARGET_COLUMN] = df[TARGET_COLUMN].values
    out[TARGET_PROVENANCE_COLUMN] = df[TARGET_PROVENANCE_COLUMN].values

    if out["observation_id"].duplicated().any():
        dups = out.loc[out["observation_id"].duplicated(), "observation_id"].head(5).tolist()
        raise ValueError(f"Duplicate (city, date) observations: {dups}")
    return out.sort_values([GROUP_COLUMN, "date"]).reset_index(drop=True)


def preprocess(df: pd.DataFrame | None = None, force: bool = False,
               raw_path=None) -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists() and not force and df is None:
        log.info("Processed data exists. Loading ...")
        return pd.read_csv(PROCESSED_DATA_PATH)

    log.info("=" * 60)
    log.info("STEP 2 – PREPROCESSING (stateless only; fitted steps live in CV folds)")
    log.info("=" * 60)

    if df is None:
        path = raw_path or RAW_DATA_PATH
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found at {path}")
        df = pd.read_csv(path)
    log.info(f"  Input: {df.shape[0]} rows × {df.shape[1]} cols")

    with open(OUTLIER_STATS_PATH, "w") as f:
        json.dump(describe_outliers(df), f, indent=2)

    out = build_processed_table(df)
    with open(FEATURES_PATH, "w") as f:
        json.dump(candidate_features(), f)
    out.to_csv(PROCESSED_DATA_PATH, index=False)
    log.info(f"  Saved {len(out)} rows → {PROCESSED_DATA_PATH}")
    return out


if __name__ == "__main__":
    result = preprocess(force=True)
    print(result.describe())

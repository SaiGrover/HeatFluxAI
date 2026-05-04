"""
UHI Prediction System - Data Preprocessing
Cleans raw data, engineers features, and prepares the ML-ready dataset.
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS, TARGET_COLUMN,
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    SCALER_PATH, FEATURES_PATH,
    MODEL_DIR,
)
from logger import get_logger

log = get_logger("preprocessor")

# Columns eligible for IQR outlier capping (weather / satellite)
# Excludes LST, lat/lon, binary/categorical, and the target itself
_OUTLIER_CAP_COLS = ["temperature", "humidity", "wind_speed", "pressure", "clouds", "ndvi"]
OUTLIER_STATS_PATH = MODEL_DIR / "outlier_stats.json"


# ─── IQR outlier capping ──────────────────────────────────────────────────────

def cap_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Winsorise each weather/satellite column to [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
    Returns the capped dataframe and a stats dict for dashboard display.
    """
    df = df.copy()
    stats = {}
    avail = [c for c in _OUTLIER_CAP_COLS if c in df.columns]

    for col in avail:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        n_low  = int((df[col] < lo).sum())
        n_high = int((df[col] > hi).sum())
        n_total = n_low + n_high
        if n_total > 0:
            df[col] = df[col].clip(lower=lo, upper=hi)
        stats[col] = {
            "q1":       round(float(q1),  3),
            "q3":       round(float(q3),  3),
            "iqr":      round(float(iqr), 3),
            "lower_fence": round(float(lo), 3),
            "upper_fence": round(float(hi), 3),
            "n_low":    n_low,
            "n_high":   n_high,
            "n_capped": n_total,
            "pct":      round(100 * n_total / max(len(df), 1), 2),
        }
        if n_total > 0:
            log.info(
                f"  Capped '{col}': {n_total} outliers "
                f"({n_low} below {lo:.2f}, {n_high} above {hi:.2f})"
            )
    total_capped = sum(s["n_capped"] for s in stats.values())
    log.info(f"  Total outliers capped: {total_capped} across {len(avail)} columns")
    return df, stats


# ─── UHI intensity computation (LST-based, dynamic) ──────────────────────────

def compute_uhi_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    UHI intensity (°C) = urban_LST − rural_LST (clipped ≥ 0).

    Rows with valid LST pairs get the real satellite-derived target.
    Rows without LST (cloud cover, missed composite) get the city-level
    median UHI as a fallback so dataset size stays stable. (Fix #6)
    """
    df = df.copy()

    if "urban_lst" not in df.columns or "rural_lst" not in df.columns:
        raise ValueError("urban_lst and rural_lst columns are required.")

    valid = df["urban_lst"].notna() & df["rural_lst"].notna()
    invalid_count = (~valid).sum()

    df[TARGET_COLUMN] = np.nan
    df.loc[valid, TARGET_COLUMN] = (
        (df.loc[valid, "urban_lst"] - df.loc[valid, "rural_lst"]).clip(lower=0)
    )

    if invalid_count > 0:
        # Fill per-city median for rows without LST — keeps rows, avoids bias
        df[TARGET_COLUMN] = df.groupby("name")[TARGET_COLUMN].transform(
            lambda s: s.fillna(s.median())
        )
        # Global median as last-resort for cities where ALL rows lack LST
        global_median = df.loc[valid, TARGET_COLUMN].median()
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(global_median)
        log.info(
            f"  {valid.sum()} rows used real LST target; "
            f"{invalid_count} rows filled with city/global median"
        )

    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(lower=0).round(3)
    log.info(
        f"  UHI range: [{df[TARGET_COLUMN].min():.2f}, "
        f"{df[TARGET_COLUMN].max():.2f}] °C  "
        f"(mean {df[TARGET_COLUMN].mean():.2f} °C)"
    )
    return df


# ─── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived features. Requires 'name' column to be present."""
    df = df.copy()

    # ── Temporal (from real API timestamps) ───────────────────────────────
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"]  = df["timestamp"].dt.hour.fillna(12).astype(int)
        df["month"] = df["timestamp"].dt.month.fillna(6).astype(int)
    else:
        df["hour"]  = 12
        df["month"] = 6

    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    # Night flag — UHI dynamics differ at night (Fix #4)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 18)).astype(int)

    # ── Satellite / land cover ────────────────────────────────────────────
    # Vegetation class: reduces direct NDVI→LST leakage (Fix #2 Option B)
    df["veg_class"] = pd.cut(
        df["ndvi"].clip(-1, 1),
        bins=[-1.0, 0.2, 0.5, 1.0],
        labels=[0, 1, 2],
    ).astype(int)

    # ── Interaction features ───────────────────────────────────────────────
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100
    df["wind_cooling_effect"]       = (
        df["wind_speed"] * (df["temperature"] - 20).clip(lower=0)
    )
    # Heat retention: built-up area + heat + calm air = UHI driver (Fix #10)
    df["heat_retention"] = (
        df["urban_fraction"] * df["temperature"] / (df["wind_speed"] + 1)
    ).round(3)

    # ── Temperature anomaly — city-relative departure (Fix #4) ───────────
    # Captures "unusually hot day for THIS city" independently of absolute T
    if "name" in df.columns:
        df["temp_anomaly"] = (
            df["temperature"]
            - df.groupby("name")["temperature"].transform("mean")
        ).round(3)
    else:
        df["temp_anomaly"] = 0.0

    # ── Trigonometric time encoding ────────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Geographic features ────────────────────────────────────────────────
    df["lat_abs"]               = df["lat"].abs()
    df["lon_sin"]               = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]               = np.cos(np.deg2rad(df["lon"]))
    df["distance_from_equator"] = df["lat"].abs()

    # ── Atmospheric stability proxy ────────────────────────────────────────
    df["heat_index"] = (
        -8.78469475556
        + 1.61139411    * df["temperature"]
        + 2.33854883889 * df["humidity"] / 100
        - 0.14611605   * df["temperature"] * df["humidity"] / 100
    ).round(3)

    return df


# ─── Full preprocessing pipeline ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """
    Full pipeline:  load → filter → impute → validate → engineer →
                    compute target → select features → scale → save.
    Saves 'city_name' alongside features so model_trainer can use GroupKFold.
    """
    if PROCESSED_DATA_PATH.exists() and not force:
        log.info(f"Processed data exists. Loading ...")
        return pd.read_csv(PROCESSED_DATA_PATH)

    log.info("=" * 60)
    log.info("STEP 2 – PREPROCESSING")
    log.info("=" * 60)

    if df is None:
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH)

    log.info(f"  Input: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── 1. Keep only real API data ─────────────────────────────────────────
    if "source" in df.columns:
        n_before = len(df)
        df = df[df["source"].str.contains("openweather|openmeteo", na=False)].copy()
        log.info(f"  Source filter: {n_before} → {len(df)} rows")

    if len(df) == 0:
        raise ValueError("No real-data rows after source filter. Re-run collection.")

    # ── 2. Ensure required columns ─────────────────────────────────────────
    defaults = {
        "temperature": 20.0, "humidity": 60.0, "wind_speed": 2.0,
        "pressure": 1013.0, "clouds": 50, "ndvi": 0.3,
        "urban_fraction": 0.5, "lat": 0.0, "lon": 0.0,
        "urban_lst": None, "rural_lst": None,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # ── 3. Impute weather columns; leave LST as NaN (handled in step 6) ───
    lst_cols  = {"urban_lst", "rural_lst"}
    imp_cols  = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in lst_cols
    ]
    missing   = df[imp_cols].isnull().sum().sum()
    df[imp_cols] = df[imp_cols].fillna(df[imp_cols].median())
    log.info(f"  Imputed {missing} missing weather values (LST columns untouched)")

    # ── 4. Remove physically invalid rows ─────────────────────────────────
    n_before = len(df)
    df = df[df["temperature"].between(-60, 60)]
    df = df[df["humidity"].between(0, 100)]
    df = df[df["wind_speed"].between(0, 100)]
    df = df[df["lat"].between(-90, 90)]
    df = df[df["lon"].between(-180, 180)]
    log.info(f"  Removed {n_before - len(df)} physically invalid rows")

    # ── 4b. IQR outlier capping (winsorisation) ───────────────────────────
    log.info("  Running IQR outlier capping ...")
    df, outlier_stats = cap_outliers(df)
    with open(OUTLIER_STATS_PATH, "w") as f:
        json.dump(outlier_stats, f, indent=2)
    log.info(f"  Outlier stats saved → {OUTLIER_STATS_PATH}")

    # ── 5. Feature engineering ────────────────────────────────────────────
    df = engineer_features(df)
    log.info("  Feature engineering complete")

    # ── 6. Compute LST-based UHI target (fill missing with median) ────────
    df = compute_uhi_intensity(df)

    # ── 7. Select and order final feature set ─────────────────────────────
    extended = FEATURE_COLUMNS + [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "lat_abs", "lon_sin", "lon_cos", "heat_index",
    ]
    seen, available = set(), []
    for c in extended:
        if c in df.columns and c not in seen:
            available.append(c)
            seen.add(c)

    # ── 8. Fit and save scaler ────────────────────────────────────────────
    df_final = df[available + [TARGET_COLUMN]].copy()

    scaler = StandardScaler()
    df_scaled = df_final.copy()
    df_scaled[available] = scaler.fit_transform(df_final[available])

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURES_PATH, "w") as f:
        json.dump(available, f)

    log.info(f"  Scaler saved → {SCALER_PATH}")
    log.info(f"  Features ({len(available)}): {available}")

    # ── 9. Save processed data with city_name for GroupKFold ──────────────
    df_final["city_name"] = df["name"].values
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    log.info(f"  Saved {len(df_final)} rows → {PROCESSED_DATA_PATH}")

    return df_final


if __name__ == "__main__":
    result = preprocess(force=True)
    print(result.describe())

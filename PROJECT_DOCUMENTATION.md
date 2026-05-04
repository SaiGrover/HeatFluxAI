# Project Documentation — Urban Heat Island Estimation System

**Subject:** Artificial Intelligence & Machine Learning (PBL)
**Topic:** Data-driven UHI Estimation using Weather + Satellite Proxies

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Data Sources](#5-data-sources)
6. [File-by-File Description](#6-file-by-file-description)
7. [Data Pipeline — End to End](#7-data-pipeline--end-to-end)
8. [Feature Engineering](#8-feature-engineering)
9. [Target Variable — UHI Intensity](#9-target-variable--uhi-intensity)
10. [Machine Learning Models](#10-machine-learning-models)
11. [Model Evaluation](#11-model-evaluation)
12. [Dashboard](#12-dashboard)
13. [Design Decisions & Scientific Validity](#13-design-decisions--scientific-validity)
14. [Limitations](#14-limitations)
15. [How to Run](#15-how-to-run)
16. [Evolution Log — What Was Fixed and Why](#16-evolution-log--what-was-fixed-and-why)

---

## 1. Project Overview

The Urban Heat Island (UHI) Estimation System is a complete, production-quality machine-learning pipeline that:

- Collects **180 days of real ERA5-reanalysis weather** for 25 major global cities (Open-Meteo, free)
- Fetches **real satellite NDVI** from NASA MODIS MOD13A2 via Google Earth Engine
- Fetches **real 8-day LST time series** from NASA MODIS MOD11A2 via Google Earth Engine
- Computes a **dynamic, satellite-validated UHI target** that varies by city AND by date
- Engineers **26 scientifically meaningful features** (weather, land cover, temporal, geographic, interaction)
- Trains **8–9 regression models** with full GridSearchCV hyperparameter tuning
- Uses **GroupKFold splitting** to ensure no city appears in both training and test — measuring real geographic generalisation
- Tracks **RMSE, MAE, R², and skill vs baseline** so model quality can be honestly assessed
- Presents all results in an **interactive 6-tab Streamlit dashboard**

Running `python main.py` executes the entire pipeline automatically.  
**Zero synthetic data. Zero random values. Every number comes from a real source.**

---

## 2. Problem Statement

Cities are significantly warmer than their surrounding rural areas — a phenomenon known as the **Urban Heat Island (UHI) effect**. Key consequences include:

- Increased energy demand (air conditioning in summer)
- Worsened air quality and heat-related health impacts
- Amplified effects of climate change in densely populated areas
- Disproportionate burden on low-income communities in urban cores

Traditional UHI quantification requires dense in-situ weather station networks, which are expensive, sparse outside wealthy cities, and non-existent for most of the global south.

**Research question:** Can freely available satellite imagery and reanalysis weather data be used to estimate UHI intensity at city scale, without any ground-based instruments?

---

## 3. Objectives

1. Build a reproducible, automated data pipeline using only free APIs
2. Compute a scientifically valid, dynamic UHI target from MODIS Land Surface Temperature
3. Engineer features that are physically meaningful and free from target leakage
4. Train and compare multiple regression models fairly using group-aware cross-validation
5. Use GridSearchCV to find optimal hyperparameters for each model
6. Benchmark every model against a trivial baseline (predict the training mean)
7. Present results in an accessible, interactive dashboard suitable for viva demonstration

---

## 4. System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                       main.py  (orchestrator)                     │
│                                                                   │
│  Step 1               Step 2               Step 3    Step 4      │
│  data_collector.py → preprocessor.py → model_trainer.py → dashboard.py │
│                                                                   │
│  raw_data.csv   →  processed_data.csv  →  best_model.pkl  → :8505 │
│                     city_name column       metrics.json           │
│                     (GroupKFold)           skill vs baseline       │
└───────────────────────────────────────────────────────────────────┘

External services:
  Open-Meteo archive-api  →  180 days daily weather  (ERA5, free, no key)
  GEE MODIS MOD13A2       →  annual mean NDVI 2023   (scale-retry for masked pixels)
  GEE MODIS MOD11A2       →  8-day LST timeseries    (server-side batch, 13 pts/city)
  OpenWeatherMap          →  current weather fallback (free tier, rarely used)
```

Each step checks whether its output file already exists. If it does, that step is skipped — re-run any step by deleting its output file.

---

## 5. Data Sources

### 5.1 Open-Meteo Archive API
- **URL:** `https://archive-api.open-meteo.com/v1/archive`
- **What it is:** A free, open-source API serving ERA5 reanalysis data from ECMWF (European Centre for Medium-Range Weather Forecasts). ERA5 is the global standard reanalysis product used in thousands of peer-reviewed climate studies.
- **What is fetched:** For each of 25 cities, 180 days of daily values: mean temperature (°C), mean relative humidity (%), max wind speed (km/h → converted to m/s), mean surface pressure (hPa), mean cloud cover (%).
- **Why 180 days:** Provides ~4,500 total rows, enough for meaningful train/test splits across cities and time. ERA5 archive goes back to 1940 and is available with ~1 day lag.
- **Why this over OpenWeatherMap historical:** OpenWeatherMap One Call 3.0 timemachine requires a paid subscription. Open-Meteo provides higher-quality, reanalysis-validated data for free.

### 5.2 Google Earth Engine — MODIS MOD13A2 (NDVI)
- **Collection ID:** `MODIS/061/MOD13A2`
- **What it is:** NASA's MODIS Vegetation Indices product. Provides 16-day composite NDVI (Normalised Difference Vegetation Index) at 500 m resolution globally.
- **What is fetched:** Annual mean NDVI for 2023 at each city's coordinates.
- **Scale factor:** Raw MODIS NDVI values are in the range -10,000 to +10,000; divided by 10,000 to recover the standard -1 to 1 range.
- **Masked pixel handling:** Dense urban core pixels and water-edge pixels can be masked. The fetcher retries at 500 m → 1 km → 2 km → 5 km scales before giving up. This fixed a bug where New York was being skipped entirely.
- **How it is used:** As the `ndvi` feature directly; to derive `urban_fraction = 1 − NDVI`; to compute the `veg_class` categorical feature.

### 5.3 Google Earth Engine — MODIS MOD11A2 (Land Surface Temperature)
- **Collection ID:** `MODIS/061/MOD11A2`
- **What it is:** NASA's MODIS Land Surface Temperature and Emissivity product. Provides 8-day daytime LST composites at 1 km resolution globally.
- **What is fetched:** The full 8-day LST timeseries over the 180-day collection window, sampled simultaneously at 13 points per city: 1 urban point (city centre) and 12 rural reference points at 0.5° (~55 km) and 1.0° (~110 km) offsets in 8 compass directions.
- **Scale factor:** Raw values × 0.02 − 273.15 converts MODIS digital numbers to degrees Celsius.
- **Rural reference strategy:** All 12 surrounding points are sampled. MODIS automatically masks water pixels (null return), so coastal/island cities do not accidentally pick up sea surface temperatures. The **median** of all valid non-null rural values is used as the rural LST reference — more stable than minimum or a single fixed direction.
- **Why not annual mean:** The original system used annual mean LST (constant per city) making the target non-varying within a city. The current system fetches the full 8-day composite timeseries so the target changes both between cities and across time, giving the model real temporal signal to learn from.
- **Null composite handling:** Rows where no 8-day composite falls within ±8 days of the weather date (e.g., extended cloud cover) receive the city-level median UHI as a fallback.

### 5.4 OpenWeatherMap (fallback only)
- **URL:** `https://api.openweathermap.org/data/2.5/weather`
- **When used:** Only if Open-Meteo fails for a given city. Returns current weather conditions. Free tier supported. In practice, almost never triggered.

---

## 6. File-by-File Description

---

### `main.py` — Pipeline Orchestrator
**What it is:** The single entry point. `python main.py` runs the entire system.

**What it does:**
- Prints a startup banner with project name and timestamp
- Calls each of the four pipeline steps in order
- Each step function checks for its output file — skips the step if the file exists, runs it if not
- Launches the Streamlit dashboard as a subprocess on port 8501 after training completes
- Catches `KeyboardInterrupt` cleanly; logs unexpected errors before re-raising

**Key design:** Incremental by default. Deleting a single output file causes only that step and downstream steps to re-run. You never re-collect data just to re-train a model.

---

### `config.py` — Central Configuration
**What it is:** The single source of truth for all constants. Every other module imports from here rather than defining its own magic numbers.

**Contents:**
- **Directory paths:** `BASE_DIR`, `DATA_DIR`, `MODEL_DIR`, `LOG_DIR`, `CACHE_DIR` — all auto-created on import
- **API credentials:** `GEE_PROJECT_ID`, `OPENWEATHER_API_KEY`, `OPENWEATHER_BASE_URL`
- **Cities list:** 25 global cities with lat/lon (Delhi, Mumbai, New York, Los Angeles, London, Tokyo, Shanghai, São Paulo, Cairo, Lagos, Jakarta, Mexico City, Karachi, Beijing, Dhaka, Bangkok, Kolkata, Chicago, Paris, Istanbul, Sydney, Toronto, Singapore, Berlin, Seoul)
- **Feature columns:** Ordered list of 26 features passed to the model
- **Target column:** `"uhi_intensity"`
- **ML settings:** `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`, `CV_FOLDS = 5`
- **Output file paths:** Exact `Path` objects for every artefact
- **UI colours:** Hex palette used by the dashboard

---

### `logger.py` — Logging Utility
**What it is:** A thin wrapper around Python's `logging` module providing consistent log formatting across all modules.

**What it does:**
- Creates a named logger with two handlers: console (INFO) and file (DEBUG)
- Log files: `logs/uhi_YYYYMMDD.log` — new file each day
- Format: `YYYY-MM-DD HH:MM:SS | LEVEL | module_name | message`
- Every other module gets its logger via `log = get_logger("module_name")`

**Why it matters:** Every API call, row count, model RMSE, LST composite count, and error goes to the log. The log file is the first debugging tool when a pipeline run fails.

---

### `data_collector.py` — Data Collection
**What it is:** Fetches all raw data from external sources. Contains zero fabricated or randomly generated values.

**Initialisation:** Calls `ee.Initialize(project=GEE_PROJECT_ID)` at import time. Authenticates if necessary.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `fetch_ndvi(lat, lon)` | Annual mean NDVI 2023 from MOD13A2. Retries at 500 m / 1 km / 2 km / 5 km scales for masked pixels. Caches result for 24 h. Returns `None` (city skipped) only if all scales fail. |
| `fetch_lst_timeseries(lat, lon, start, end)` | Builds a FeatureCollection of 13 sample points (1 urban + 12 rural offsets). Calls `sampleRegions` server-side across the entire 8-day composite collection in a **single GEE batch call** per city. Parses the returned features into a `{date: {urban_lst, rural_lst}}` dict. Rural LST = median of valid surrounding pixels. Caches for 30 days. |
| `match_to_nearest_lst(date_str, lst_ts)` | Given a daily weather date and the city's LST timeseries, finds the nearest available 8-day composite. Returns `(None, None)` if the gap exceeds 8 days (one composite period). |
| `fetch_openmeteo_history(city, start, end)` | Fetches 180 days of daily ERA5 weather from Open-Meteo. Converts wind from km/h to m/s. Caches for 30 days. |
| `fetch_openweather_current(city)` | Last-resort fallback: current weather from OpenWeatherMap. Caches for 1 hour. |
| `collect_data(force=False)` | Main loop: for each city → NDVI → LST timeseries → weather rows → match LST per row → append. Saves to `raw_data.csv`. |

**Collection window:** Ends 30 days before today (MODIS processing lag), starts 180 days before that end date. This guarantees full MODIS LST coverage for every weather row.

**Caching:** All API responses are written as JSON to `cache/`. Keys are MD5 hashes of the request parameters. The GEE server-side batch approach means one cache file per (city, date-range) holds the entire LST timeseries — not one file per composite.

**Output:** `data/raw_data.csv` — approximately 4,500 rows. Columns: `name, lat, lon, temperature, humidity, wind_speed, pressure, clouds, timestamp, ndvi, urban_fraction, urban_lst, rural_lst, source`.

---

### `preprocessor.py` — Preprocessing & Feature Engineering
**What it is:** Transforms raw_data.csv into a clean, feature-rich, ML-ready dataset.

**Pipeline (in order):**

1. **Source filter** — keeps only rows where `source` contains `"openweather"` or `"openmeteo"`. Guards against any synthetic data contaminating the dataset.
2. **Default column injection** — adds any missing columns (e.g., `pressure`, `clouds`) with sensible defaults.
3. **Imputation** — fills missing values in weather columns with column medians. `urban_lst` and `rural_lst` are explicitly excluded from imputation — they remain NaN and are handled separately in step 6.
4. **Physical validation** — removes rows outside valid ranges: temperature (−60 to +60°C), humidity (0–100%), wind speed (0–100 m/s), lat/lon bounds.
4b. **IQR outlier capping (winsorisation)** — for `temperature`, `humidity`, `wind_speed`, `pressure`, `clouds`, `ndvi`: values outside `[Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]` are clipped to the fence value. Values are **capped, not dropped** — dataset size is preserved. Per-column stats (Q1, Q3, IQR, fences, count and % of capped values) are saved to `models/outlier_stats.json` and visualised in the Preprocessing tab.
5. **Feature engineering** — creates all 26 model features (see Section 8).
6. **UHI target computation** — `uhi_intensity = max(0, urban_lst − rural_lst)`. Rows with valid LST get the real satellite value. Rows without LST (NaN) get the city-level median UHI; if the whole city has no LST, the global median is used. No rows are dropped.
7. **Feature selection** — selects the 26 features from `FEATURE_COLUMNS` plus trig/geographic extensions.
8. **Scaling** — fits `StandardScaler` on feature columns; saves scaler to `models/scaler.pkl` and feature list to `models/feature_names.json`.
9. **Save** — writes `data/processed_data.csv` with features + `uhi_intensity` + `city_name`. The `city_name` column is not a model feature — it exists solely for GroupKFold in the model trainer.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `cap_outliers(df)` | IQR winsorisation on weather/satellite columns; returns capped df + stats dict |
| `compute_uhi_intensity(df)` | Dynamic LST-based target; city/global median fallback for missing LST |
| `engineer_features(df)` | All 26 features including new: `veg_class`, `temp_anomaly`, `is_night`, `heat_retention` |
| `preprocess(df, force)` | Full pipeline — load, clean, cap outliers, engineer, target, scale, save |

---

### `model_trainer.py` — Model Training & Selection
**What it is:** Trains 8–9 regression models with full hyperparameter tuning, evaluates them with honest group-aware metrics, and saves the best.

**Pipeline (in order):**

1. **Load** processed data; extract `city_name` as the group variable; load or fit scaler; apply to features.
2. **GroupShuffleSplit** — assigns entire cities to train or test with 80/20 city ratio and `random_state=42`. A city in the test set is completely unseen during training. Train/test city lists are logged and saved to `metrics.json`.
3. **Baseline** — computes predictions of constant `mean(y_train)` on the test set. Establishes RMSE, MAE, R² that a trivial predictor achieves. All model metrics are interpreted relative to this.
4. **Per-model loop:**
   - Wraps model in `GridSearchCV` with `GroupKFold(n_splits=5)` as the inner CV — hyperparameter search also respects city boundaries.
   - Calls `gs.fit(X_train, y_train, groups=train_city_names)`.
   - Uses `gs.best_estimator_` for test-set evaluation.
   - Computes RMSE, MAE, R² on held-out test cities.
   - Runs outer `GroupKFold` CV on full dataset for CV-RMSE ± Std.
   - Computes `skill_vs_baseline = 1 − (model_RMSE / baseline_RMSE)`.
5. **Select** model with lowest test RMSE.
6. **Save** best model to `models/best_model.pkl`; full metrics to `models/metrics.json`.

**GridSearch parameter grids:**

| Model | Parameters tuned |
|-------|-----------------|
| Ridge Regression | `alpha`: [0.01, 0.1, 1, 10, 100] |
| Lasso Regression | `alpha`: [0.001, 0.01, 0.1, 1] |
| Decision Tree | `max_depth`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| K-Nearest Neighbors | `n_neighbors`, `weights` |
| XGBoost | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| LightGBM | `n_estimators`, `learning_rate`, `max_depth` |

**Key functions:**

| Function | Purpose |
|----------|---------|
| `get_base_models()` | Returns dict of uninitialised model instances |
| `get_feature_importance(model, names)` | Extracts and normalises feature importances |
| `train(force=False)` | Full training pipeline |

---

### `dashboard.py` — Streamlit Interactive Dashboard  ✦ Elite Dark Edition ✦
**What it is:** A 6-tab interactive web application visualising every stage of the pipeline in a bespoke dark-mode UI.

**Technology:** Streamlit (framework), Plotly (all interactive charts), custom CSS + animations injected via `st.markdown()`, Google Fonts (Inter, Space Grotesk, JetBrains Mono).

**Port:** `8505` (configured in `config.py → DASHBOARD_PORT`).

**UI design highlights:**
- Full dark theme with custom colour palette (see `config.py → COLORS`)
- Animated hero banner with gradient title text
- Animated live-stats ticker strip showing global UHI stats from loaded data
- KPI cards with per-card accent colour, hover lift animation, and top gradient bar
- Glass-morphism section cards with neon border glow on hover
- Animated scrollbar, custom tab styling, button overrides, expander styling

**Helper functions:**

| Function | Purpose |
|----------|---------|
| `kpi(col, icon, value, label, sub, color)` | Renders one animated KPI metric card |
| `sec(title, sub)` | Renders a section heading + subtitle |
| `sfig(fig)` | Applies the dark Plotly theme dict `_PL` to any figure |
| `hex_rgba(hex, alpha)` | Converts `#RRGGBB` → `rgba(R,G,B,alpha)` for Plotly fill colours |
| `build_input(...)` | Assembles the full 26-feature vector from slider inputs, applying all engineering transformations |
| `render_ticker(proc_df, metrics, raw_df)` | Generates the animated scrolling live-stats strip |
| `render_severity_table()` | Renders the UHI severity reference table with coloured indicators |
| `render_feature_contributions(vec, feat_names)` | Renders horizontal scaled-value bars for top-12 features |

**Tabs:**

| Tab | Key content |
|-----|------------|
| **Overview** | Live-stats animated ticker · 6 KPI cards · 4-step pipeline status cards · UHI intensity histogram (mean + median lines) · city ranking bar chart with std error bars · **UHI severity reference table** with 5 severity levels · **violin plot** (top-8 cities, with box + mean line) · global scatter mapbox |
| **Data Explorer** | 4 sub-tabs: Raw Data (metrics, source pie, feature box plots, table + download) · Processed (metrics, correlation heatmap, table + download) · Time Series (multi-city UHI line chart, monthly average chart) · Correlations (Pearson r bar chart, scatter vs top correlator with OLS trendline) |
| **Preprocessing** | Missing value bar chart · before/after shape comparison · **IQR outlier capping** (summary metric cards per capped column, full statistics table, raw vs capped box plots with fence lines, stacked outlier count bar) · engineered feature grid by category · feature distribution raw vs processed comparison |
| **Models** | **🥇🥈🥉 ranked leaderboard cards** (top-4) · full comparison table (best model highlighted green) · RMSE + MAE bar charts with baseline reference line · skill score bar · model comparison radar (5 axes, normalised) · R² vs RMSE bubble chart · predicted vs actual scatter (city-coloured, OLS + perfect-prediction line) · **residual analysis** (histogram + normal fit overlay, residuals vs predicted scatter, per-city MAE bar) · feature importance selector · GroupKFold CV error bars |
| **Heatmap** | Scatter or density mapbox · UHI range slider · base map style selector · summary card · hottest 10 / coolest 10 bar charts · seasonal city × month heatmap |
| **Prediction** | 12 input sliders (weather + land cover + location + time) · city quick-load preset selector · live predicted UHI with animated severity badge · gauge chart (0–8°C with colour zones) · radar chart of 6 driving factors · **scaled feature contribution bars** (top 12, coloured by direction) · sensitivity analysis line chart with current-value marker · **3D response surface** (Temperature × Urban Fraction → UHI, interactive with current position marker) · **city similarity finder** (Euclidean distance in feature space to all training cities, top-5 as flag pills with % similarity + comparison card) · **mitigation insights** (4 evidence-based suggestions dynamically generated from slider values) |

**Cached loaders:** All file reads use `@st.cache_data` or `@st.cache_resource` with TTLs. The prediction and sensitivity analysis recompute on every slider change; all data loading is cached independently.

---

### `requirements.txt` — Python Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.32.0 | Dashboard web framework |
| `pandas` | ≥ 2.0.0 | DataFrames, CSV I/O |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `scikit-learn` | ≥ 1.3.0 | Models, GridSearchCV, GroupKFold, StandardScaler |
| `plotly` | ≥ 5.18.0 | All interactive charts |
| `requests` | ≥ 2.31.0 | HTTP calls to Open-Meteo and OpenWeatherMap |
| `xgboost` | ≥ 2.0.0 | XGBoost gradient boosting |
| `lightgbm` | ≥ 4.0.0 | LightGBM gradient boosting |
| `earthengine-api` | ≥ 0.1.390 | Google Earth Engine Python client |
| `google-auth` | ≥ 2.0.0 | GEE OAuth 2.0 authentication |

---

### `data/raw_data.csv` — Raw Collected Data

One row per (city, day). Approximately 4,500 rows total.

| Column | Type | Description |
|--------|------|-------------|
| `name` | string | City name |
| `lat` | float | Latitude |
| `lon` | float | Longitude |
| `temperature` | float | Daily mean temperature (°C, ERA5) |
| `humidity` | float | Daily mean relative humidity (%, ERA5) |
| `wind_speed` | float | Daily max wind speed (m/s, converted from km/h) |
| `pressure` | float | Daily mean surface pressure (hPa, ERA5) |
| `clouds` | int | Daily mean cloud cover (%) |
| `timestamp` | string | ISO 8601 datetime, T12:00:00 (midday representative) |
| `ndvi` | float | Annual mean NDVI 2023, rounded to 4 d.p. |
| `urban_fraction` | float | 1 − NDVI, rounded to 3 d.p. |
| `urban_lst` | float or NaN | 8-day composite daytime LST at city centre (°C) matched to this row's date |
| `rural_lst` | float or NaN | Median 8-day LST across 12 rural directional offsets (°C) |
| `source` | string | `"openmeteo+gee"` for all real rows |

---

### `data/processed_data.csv` — Processed Feature Matrix

Output of `preprocessor.py`. Human-readable (un-scaled). Contains:
- All 26 engineered features
- `uhi_intensity` target column
- `city_name` column (for GroupKFold in model_trainer; not a model feature)

---

### `models/best_model.pkl`
Pickled best-performing trained model after GridSearch tuning. Loaded by the dashboard prediction tab.

---

### `models/scaler.pkl`
Pickled `StandardScaler` fitted on training features. Must be applied to any new input before calling the model. Loaded by the dashboard automatically.

---

### `models/metrics.json`
Full evaluation results for every model. Structure:

```json
{
  "best_model": "XGBoost",
  "best_rmse": 1.45,
  "baseline_rmse": 2.10,
  "baseline_mae": 1.72,
  "train_cities": ["Delhi", "Tokyo", ...],
  "test_cities": ["Seoul", "Lagos", "Paris", "Chicago", "Sydney"],
  "feature_names": ["temperature", "humidity", ...],
  "models": {
    "XGBoost": {
      "rmse": 1.45,
      "mae": 1.10,
      "r2": 0.63,
      "cv_rmse": 1.58,
      "cv_std": 0.21,
      "skill_vs_baseline": 0.31,
      "best_params": {"learning_rate": 0.1, "max_depth": 6, ...},
      "feature_importance": {"temp_anomaly": 0.18, "urban_fraction": 0.15, ...}
    }
  }
}
```

`skill_vs_baseline` > 0 means the model beats the trivial mean predictor. This is the minimum bar for a useful model.

---

### `models/feature_names.json`
JSON array of feature names in the exact order the trained model expects. Used by the dashboard to correctly assemble the feature vector from slider inputs.

---

### `models/outlier_stats.json`
Written by `preprocessor.cap_outliers()` during step 4b. Contains per-column winsorisation statistics for the six capped features.

```json
{
  "temperature": {
    "q1": 18.5, "q3": 32.1, "iqr": 13.6,
    "lower_fence": -1.9, "upper_fence": 52.5,
    "n_low": 3, "n_high": 7, "n_capped": 10, "pct": 0.22
  },
  ...
}
```

Used by the Preprocessing tab to display outlier summary cards, the full IQR statistics table, raw vs capped box plots, and the stacked outlier count chart.

---

### `logs/uhi_YYYYMMDD.log`
Rotating daily log files. Sample line:
```
2026-05-03 14:22:01 | INFO  | data_collector | Delhi           180 rows  LST coverage 178/180
```
First place to look when a pipeline run fails or produces unexpected results.

---

### `cache/`
MD5-keyed JSON files caching all API responses. TTLs:
- GEE NDVI: 24 hours
- GEE LST timeseries: 30 days (historical data is immutable)
- Open-Meteo historical: 30 days
- OpenWeatherMap current: 1 hour

Delete all `cache/cache_*.json` files to force fresh API calls on the next run.

---

## 7. Data Pipeline — End to End

```
config.py: 25 global cities
        │
        ▼
[data_collector.py]
  For each city:
    ① GEE MOD13A2 → NDVI (scale retry: 500m/1km/2km/5km)
       If all scales fail → skip city
    ② GEE MOD11A2 → 8-day LST timeseries for 180-day window
       Single server-side sampleRegions call: 1 urban + 12 rural offsets
       rural_lst per composite = median of valid surrounding pixels
    ③ Open-Meteo → 180 daily weather rows
       urban_fraction = 1 − NDVI  (no random)
    ④ For each daily row → match to nearest 8-day composite (±8 days max)
       Attach urban_lst, rural_lst (or None if no nearby composite)
        │
        ▼ raw_data.csv  (~4500 rows, 14 columns)
        │
[preprocessor.py]
  Source filter (openweather|openmeteo)
  Impute weather columns with medians (LST columns left as NaN)
  Remove physically invalid rows
  Engineer 26 features:
    temporal: hour, month, is_daytime, is_night, sin/cos encodings
    land: veg_class, urban_fraction, ndvi
    interaction: temp_humidity, wind_cooling, temp_anomaly, heat_retention
    geographic: lat_abs, lon_sin/cos, distance_from_equator
    atmospheric: heat_index
  Compute UHI = urban_lst − rural_lst
    (city median fallback for rows without LST — no rows dropped)
  Fit & save StandardScaler
  Save city_name alongside features (for GroupKFold)
        │
        ▼ processed_data.csv + scaler.pkl + feature_names.json
        │
[model_trainer.py]
  Load processed data; extract city_name as groups
  Apply scaler → X_scaled
  GroupShuffleSplit → train cities / test cities (no overlap)
  Compute baseline: predict mean(y_train) always → RMSE_baseline
  For each of 8-9 models:
    GridSearchCV with GroupKFold(5) inner CV on training cities
    Evaluate on held-out test cities: RMSE, MAE, R²
    Outer GroupKFold CV on full data: CV-RMSE ± Std
    Compute skill = 1 − RMSE / RMSE_baseline
  Select best by test RMSE
  Save best_model.pkl + metrics.json
        │
        ▼ best_model.pkl + metrics.json
        │
[dashboard.py]
  Load all artefacts (cached)
  Render 6-tab Streamlit app
  Prediction: sliders → engineer features → scale → predict → display
```

---

## 8. Feature Engineering

All features are computed in `preprocessor.engineer_features()`. The `name` column must be present for `temp_anomaly`.

| Feature | Formula / Source | Scientific rationale |
|---------|-----------------|---------------------|
| `temperature` | Open-Meteo daily mean (°C) | Primary driver of sensible heat flux |
| `humidity` | Open-Meteo daily mean (%) | Affects latent heat and evaporative cooling |
| `wind_speed` | Open-Meteo daily max (m/s) | Higher wind disperses urban heat faster |
| `pressure` | Open-Meteo daily mean (hPa) | Proxy for synoptic weather regime |
| `clouds` | Open-Meteo daily mean (%) | Cloud cover modulates solar radiation reaching surface |
| `ndvi` | GEE MOD13A2 annual mean | Vegetation cover index; dense vegetation cools via evapotranspiration |
| `urban_fraction` | `1 − NDVI` | Proxy for impervious surface fraction; more built-up = more heat retention |
| `veg_class` | `pd.cut(ndvi, [-1, 0.2, 0.5, 1], labels=[0,1,2])` | Categorical bins: 0=sparse, 1=moderate, 2=dense. Reduces direct numeric NDVI→LST leakage |
| `lat` | Raw coordinate | Geographic baseline |
| `lon` | Raw coordinate | Geographic baseline |
| `distance_from_equator` | `abs(lat)` | Equatorial cities have different baseline temperatures and UHI profiles |
| `hour` | From timestamp `.dt.hour` | Hour of day drives solar loading and wind pattern |
| `month` | From timestamp `.dt.month` | Seasonal variation in baseline temperature and UHI |
| `is_daytime` | `1 if 6 ≤ hour ≤ 18` | Boolean indicator for daytime solar forcing |
| `is_night` | `1 if hour < 6 or hour > 18` | Night UHI can be stronger than daytime due to slow rural cooling |
| `temp_humidity_interaction` | `temperature × humidity / 100` | Heat index proxy; humid heat is retained differently |
| `wind_cooling_effect` | `wind_speed × max(0, temperature − 20)` | Wind cooling is more effective when ambient temperature is high |
| `temp_anomaly` | `temperature − city_mean_temperature` | City-relative departure; captures "unusually hot day" independently of absolute T |
| `heat_retention` | `urban_fraction × temperature / (wind_speed + 1)` | Combines built-up land, ambient heat, and wind damping — core UHI mechanism |
| `hour_sin` | `sin(2π × hour / 24)` | Smooth cyclical encoding; avoids discontinuity at 23→0 |
| `hour_cos` | `cos(2π × hour / 24)` | Paired with hour_sin for full 24-h cycle |
| `month_sin` | `sin(2π × month / 12)` | Smooth seasonal encoding |
| `month_cos` | `cos(2π × month / 12)` | Paired with month_sin |
| `lat_abs` | `abs(lat)` | Absolute latitude for symmetric hemisphere treatment |
| `lon_sin` | `sin(radians(lon))` | Continuous east-west encoding |
| `lon_cos` | `cos(radians(lon))` | Paired with lon_sin for full longitude circle |
| `heat_index` | Steadman approximation: `−8.78 + 1.61×T + 2.34×H − 0.15×T×H` | Atmospheric stability proxy; captures humid-heat synergy beyond raw T and H |

**Permanently removed: `urban_heat_proxy`** (= `urban_fraction × (1 − NDVI)`). This was `(1 − ndvi) × (1 − ndvi)` — a squared NDVI term. Since the UHI target is derived from MODIS LST which itself correlates with land cover (NDVI), this feature was a direct data leakage path. Removed in full.

---

## 9. Target Variable — UHI Intensity

### Definition
```
UHI intensity (°C) = max(0,  urban_LST(t)  −  rural_LST(t))
```

### Dynamic (8-day) LST — Why it matters

| Property | Old system (annual mean) | Current system (8-day timeseries) |
|----------|--------------------------|----------------------------------|
| Target varies across cities | Yes | Yes |
| Target varies within a city | No (same value all 180 days) | Yes (changes with LST composite) |
| Features vary | Yes (daily weather) | Yes (daily weather) |
| Model learns | Nothing temporal | Temporal + spatial patterns |

With a constant per-city target, the model had no incentive to learn from daily weather variation. Now the features and target carry independent temporal signal across 22–23 composites per city.

### Rural reference — Why median of 12 directions

The original system used a fixed +0.45° northeast offset. Discovered bugs:

- Mumbai's offset landed in hot, semi-arid inland Maharashtra → rural_lst > urban_lst → UHI = 0
- Istanbul, Jakarta, Singapore, Sydney offsets landed on water → MODIS null → LST missing for 150 rows, then wrongly imputed with dataset median

**Fix:** Sample all 12 compass directions at 0.5° and 1.0°. MODIS automatically null-masks water pixels. Take the **median** of the valid (non-null) values. This is physically correct — the median of surrounding land pixels is a stable rural background. It avoids the extremes that the minimum (coolest = possibly anomalous cold pixel) or the maximum (hottest = hot desert) would introduce.

### Missing LST fallback

Rows where no 8-day composite falls within ±8 days (extended cloudy periods, MODIS data gaps) receive the city-level median UHI. This keeps dataset size at ~4,500 rows rather than shrinking with each gap. Cities where the entire time window has no LST data receive the global dataset median.

---

## 10. Machine Learning Models

### Models

| Model | Type | Notes |
|-------|------|-------|
| Linear Regression | Linear | Unregularised baseline |
| Ridge Regression | Linear + L2 | Controls multicollinearity; α tuned |
| Lasso Regression | Linear + L1 | Sparse solution; implicit feature selection; α tuned |
| Decision Tree | Non-linear | Interpretable splits; prone to overfitting without depth limit |
| Random Forest | Ensemble (bagging) | Variance reduction via averaging; 3 params tuned |
| Gradient Boosting | Ensemble (boosting) | Iterative residual fitting; 3 params tuned |
| K-Nearest Neighbors | Instance-based | Non-parametric; no training phase; k + weight tuned |
| XGBoost | Optimised boosting | Regularised gradient boosting; state-of-the-art on tabular data; 4 params tuned |
| LightGBM | Fast boosting | Histogram-based leaf-wise growth; efficient for larger datasets; 3 params tuned |

### Hyperparameter tuning — GroupKFold aware

All models except Linear Regression use `GridSearchCV`:
- **Inner CV:** `GroupKFold(n_splits=5)` — hyperparameter selection never sees test cities
- **Scoring:** `neg_root_mean_squared_error`
- **`n_jobs=-1`:** Parallelised across all CPU cores
- **`refit=True`:** Automatically refits best parameters on full training set

Best parameters per model are saved under `"best_params"` in `metrics.json`.

---

## 11. Model Evaluation

### Train / test integrity — GroupShuffleSplit

`GroupShuffleSplit(test_size=0.2)` assigns complete cities to train or test. With 25 cities and `test_size=0.2`, approximately 5 cities go to test and 20 to train. The exact city lists are logged and saved in `metrics.json` for reproducibility.

This is the correct evaluation protocol for spatial generalisation. It answers: *"Can this model estimate UHI for a city it has never seen?"* — which is the real use case.

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **RMSE** | `sqrt(mean((pred − true)²))` | Average error in °C. Penalises large errors more than MAE. |
| **MAE** | `mean(abs(pred − true))` | Average absolute error in °C. More interpretable than RMSE for reporting to non-specialists. |
| **R²** | `1 − SS_residual / SS_total` | Fraction of UHI variance explained. 1.0 = perfect; 0.0 = predicting the mean. |
| **CV-RMSE ± Std** | GroupKFold 5-fold | Generalisation stability. High std = sensitive to which cities train vs test. |
| **Skill vs Baseline** | `1 − RMSE / RMSE_baseline` | Relative improvement over a trivial mean predictor. Must be > 0 for the model to be useful. |

### Baseline comparator

The baseline always predicts `mean(y_train)` regardless of input. Its RMSE on the test set is the minimum bar any model must clear. A model with `skill_vs_baseline ≤ 0` is worse than predicting the mean — it should be rejected.

### Expected result range

With a dynamic LST-based target and GroupKFold evaluation:
- RMSE: ~1.0–2.5 °C (realistic for spatial generalisation, city-level UHI spans 0–10°C)
- R²: ~0.4–0.7 (lower than the old 0.92 which was an artefact of leakage — this is honest)
- Skill vs baseline: ~0.2–0.5 for good ensemble models

---

## 12. Dashboard

### Navigation

```
Sidebar navigation
  ├── Overview       — Live ticker · KPIs · pipeline health · UHI distribution · severity guide
  ├── Data Explorer  — Raw / processed data · time series · correlations
  ├── Preprocessing  — Missing values · IQR capping · feature engineering · distributions
  ├── Models         — Leaderboard · comparison · residuals · feature importance
  ├── Heatmap        — Global interactive UHI map · seasonal heatmap
  └── Prediction     — Live prediction · 3D surface · city similarity · mitigation insights
```

### Overview tab — key components

- **Animated ticker** — scrolling strip showing live global mean UHI, peak UHI, hottest city, best model RMSE, and skill score, populated from the loaded dataset at render time.
- **KPI cards** — raw samples, processed rows, feature count, model count, best RMSE, skill score. Each card has a coloured top bar and lift animation on hover.
- **Pipeline status cards** — four step cards showing collection → preprocessing → training → dashboard, with ✓/… completion indicators.
- **UHI histogram** — mean + median reference lines, 55-bin Plotly histogram.
- **City ranking bar chart** — horizontal bars with std error, coloured by UHI magnitude (green→red).
- **UHI severity reference table** — five bands (< 1, 1–2, 2–3, 3–5, > 5 °C) with colour-coded severity labels and plain-English health/infrastructure implications.
- **Violin plot** — top-8 hottest cities, each with box + mean line overlay. Country flag emojis from `CITY_EMOJI` map.
- **Global scatter mapbox** — sized + coloured by mean city UHI, carto-darkmatter basemap.

### Prediction tab — full mechanics

1. User adjusts 12 sliders: temperature, wind speed, pressure, humidity, cloud cover, NDVI, urban fraction, lat, lon, hour, month, plus a quick-load city preset.
2. On every slider change, Streamlit re-renders the output column.
3. From slider values, all 26 engineered features are recomputed: `veg_class`, `temp_anomaly`, `is_night`, `heat_retention`, trig encodings, `heat_index`, etc.
4. Feature vector is assembled in the exact order from `feature_names.json`.
5. Vector is passed through the saved `StandardScaler`.
6. `best_model.predict()` returns predicted UHI in °C.
7. The following update live:
   - **Prediction box** — large animated gradient value + severity badge
   - **Gauge chart** — 0–8 °C range with colour-zone background steps
   - **Radar chart** — 6 qualitative driving factors normalised 0→1
   - **Feature contribution bars** — top-12 features by scaled magnitude, colour-coded positive (blue) / negative (red)
   - **Sensitivity analysis** — one parameter swept 0→max while others are fixed; current position marked as dashed line + dot
   - **3D response surface** — 35×35 grid over Temperature × Urban Fraction, surface coloured by UHI; current slider position plotted as a "← You" marker
   - **City similarity** — Euclidean distance in scaled feature space to all 25 training city centroids; top-5 shown as flag pills with similarity %; closest city mean UHI compared to prediction
   - **Mitigation insights** — 4 dynamically generated cards based on which slider values are in concerning ranges (high urban fraction, low NDVI, low wind, etc.)

---

## 13. Design Decisions & Scientific Validity

### Comprehensive validity table

| Aspect | Invalid approach (removed) | Current valid approach |
|--------|---------------------------|----------------------|
| UHI target | `temperature − rural_avg + noise` (circular, fabricated) | `urban_LST(t) − rural_LST(t)` from MODIS 8-day composites |
| Target temporality | Constant per city (annual mean) | Dynamic — varies by 8-day period |
| Rural reference | Single fixed +0.45° NE offset | Median of 12 compass directions at 0.5° and 1.0° |
| Rural pixel quality | Would pick hot inland/water pixels | MODIS null-masks water; median avoids outlier directions |
| NDVI fallback | `random.uniform(0.2, 0.5)` | Skip city if all GEE scales fail |
| Urban fraction | `random.uniform(0.4, 0.9)` | `1 − NDVI` (physically grounded) |
| `urban_heat_proxy` feature | Present (= squared NDVI, leakage) | Permanently removed |
| Missing LST rows | Imputed with dataset median (wrong target) | City-level median UHI fallback |
| Dataset size | 25 rows → too small | ~4,500 rows (25 cities × 180 days) |
| Timestamps | All identical (collection time) | Real daily timestamps from Open-Meteo |
| Train/test split | Random — same city in train and test | GroupShuffleSplit — entire cities to one side |
| Hyperparameter CV | Regular KFold (leaks city info) | GroupKFold — CV also city-aware |
| Evaluation metrics | RMSE + R² only | RMSE + MAE + R² + skill vs baseline |
| Baseline comparator | None | Always-predict-mean baseline computed first |
| New York city | Missing (NDVI masked pixel, no retry) | NDVI scale retry (500m→5km) recovers the city |
| Istanbul/Singapore LST | NaN (offset on water, imputed wrong) | 12-direction search finds land pixels |
| Data leakage (veg) | Raw NDVI directly correlated with LST target | `veg_class` bins break the numeric continuity |

---

## 14. Limitations

1. **25 cities is a small spatial sample.** Geographic generalisation to unseen cities (especially in under-represented regions like central Africa, central Asia, or Oceania beyond Sydney) is uncertain. Adding more cities from diverse climate zones would significantly improve robustness.

2. **urban_fraction = 1 − NDVI is a proxy.** Low NDVI can indicate desert, bare rock, or snow — not just urban land. A proper impervious surface dataset (e.g., ESA WorldCover 10 m) would be more accurate. The proxy is reasonable for tropical and temperate cities but may mislead for arid cities like Cairo or Karachi.

3. **Open-Meteo/ERA5 is reanalysis.** ERA5 is a model output constrained by observations, not direct measurements. It smooths out hyper-local urban microclimate signals. Weather data from urban stations would capture the city heat dome directly, but such data is not freely available globally.

4. **MODIS LST at 1 km resolution.** At 1 km, a city-centre pixel averages across a large area including parks, rivers, and roads. Higher-resolution thermal imagery (e.g., Landsat TIRS at 100 m, Sentinel-3 SLSTR) would better isolate the hottest urban cores.

5. **Rural reference is still geometrically defined.** Even with 12 compass directions, the rural reference is defined by distance from the city, not by land-cover classification. A rural pixel is ideally classified as non-urban by a land-cover product and has NDVI > 0.3. Future improvement would filter candidate rural pixels by NDVI threshold before taking the median.

6. **8-day LST composites coarsen temporal resolution.** MODIS provides at most one 8-day composite per period. On days that fall inside the same composite, the target is identical for all weather rows. This limits the day-to-day learning signal within a composite period.

---

## 15. How to Run

### Prerequisites

```bash
# Python 3.10 or newer required
pip install -r requirements.txt

# Authenticate GEE once (opens browser for Google account login)
earthengine authenticate
```

### Full automated pipeline

```bash
python main.py
```

Runs all four steps in order. Steps whose output files already exist are skipped. Dashboard opens automatically at `http://localhost:8501`.

### Force re-run individual steps (Windows)

```bash
del data\raw_data.csv              # triggers step 1 re-run
del data\processed_data.csv        # triggers step 2 re-run
del models\best_model.pkl          # triggers step 3 re-run
del models\metrics.json            # triggers step 3 re-run
del cache\cache_*.json             # clears API cache (fresh GEE + weather calls)
```

### Force re-run individual steps (macOS / Linux)

```bash
rm data/raw_data.csv
rm data/processed_data.csv
rm models/best_model.pkl models/metrics.json
rm cache/cache_*.json
```

### Run individual modules directly

```bash
python data_collector.py    # collection only
python preprocessor.py      # preprocessing only
python model_trainer.py     # training only
streamlit run dashboard.py  # dashboard only (port 8501)
```

### Common issues and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `EEException: Not signed up for Earth Engine` | GEE account not activated | Visit earthengine.google.com, sign up, then re-run `earthengine authenticate` |
| City skipped with `NDVI unavailable` | All 4 NDVI scales returned null — rare, usually a transient GEE issue | Delete NDVI cache for that city and re-run |
| `LST timeseries: 0 composites` | MODIS not yet processed for recent dates (GEE lag) | The collection window ends 30 days ago by design; if fresh data is needed adjust `end_dt` in `collect_data()` |
| `No data collected` | GEE initialisation failed or quota exceeded | Check GEE console for quota; wait 24 h or use a different GEE project |
| Dashboard shows stale results after code changes | Old artefacts loaded from disk | Delete `data/processed_data.csv` and `models/` files, re-run `python main.py` |
| `ModuleNotFoundError: earthengine` | Package not installed | `pip install earthengine-api google-auth` |
| Very slow training | Large GridSearch grids on slow CPU | Reduce `PARAM_GRIDS` grid sizes in `model_trainer.py` or set `CV_FOLDS=3` in `config.py` |

---

## 16. Evolution Log — What Was Fixed and Why

This section documents the major changes made during development, explaining the reasoning so the project history is transparent.

### Round 1 — Initial corrections

| Change | File | Reason |
|--------|------|--------|
| Replaced `temperature − rural_avg + noise` target with `urban_LST − rural_LST` | `preprocessor.py` | Old target was circular (rural avg derived from same dataset) and had fabricated noise |
| Removed `random.uniform()` for NDVI and urban_fraction | `data_collector.py` | Random values have no scientific basis; replaced with real GEE NDVI and 1−NDVI proxy |
| Skip city if NDVI unavailable (no fallback) | `data_collector.py` | Fabricated NDVI would propagate incorrect features and target |
| Switched historical weather to Open-Meteo (ERA5) | `data_collector.py` | OpenWeatherMap timemachine requires paid subscription; Open-Meteo is free and higher quality |
| Changed 30 days to 180-day collection window | `data_collector.py` | 25 rows → ~4500 rows; model needs sufficient samples for meaningful cross-validation |
| Removed `urban_heat_proxy` feature | `preprocessor.py`, `config.py` | Direct leakage: `(1−NDVI)²` is a function of the target variable |
| Added `distance_from_equator` feature | `preprocessor.py` | Explicit geographic signal; latitude strongly predicts climate regime |
| Changed CV_FOLDS from 5 to 3 | `config.py` | Appropriate for smaller datasets; later reverted to 5 after dataset grew |
| Added `shuffle=True` to train_test_split | `model_trainer.py` | Data ordered city-by-city; without shuffle test set contains only last cities |
| Added GridSearchCV for all models | `model_trainer.py` | Fixed hyperparameters are suboptimal; tuning improves fairness of comparison |
| Honest dashboard title | `dashboard.py` | "Real-time prediction system" was misleading; changed to "Data-driven estimation" |

### Round 2 — Data quality fixes

| Change | File | Reason |
|--------|------|--------|
| Added NDVI scale retry (500m→5km) | `data_collector.py` | New York (dense urban core) had masked NDVI pixel at 500m — was silently dropped |
| Replaced single +0.45° rural offset with 12-direction search | `data_collector.py` | Fixed offset put Mumbai's rural reference in hot inland Maharashtra → rural_lst > urban_lst → UHI=0 |
| Changed rural LST selection from `min` to `median` | `data_collector.py` | Minimum was unstable (one anomalously cold pixel could dominate); median is robust |
| Excluded `urban_lst`/`rural_lst` from median imputation | `preprocessor.py` | Imputing NaN LST with dataset median silently gave Istanbul/Jakarta/Sydney fabricated targets |
| Deleted stale `processed_data.csv` (24 rows) | — | Old file from pre-180-day pipeline; preprocessor was skipping reprocessing due to existence check |

### Round 3 — Scientific rigour upgrade

| Change | File | Reason |
|--------|------|--------|
| Dynamic 8-day LST target (replaces annual mean) | `data_collector.py` | Annual mean = constant per city → model had no temporal signal to learn; 8-day composites give target that varies over the 180-day window |
| Server-side GEE batch sampling (`sampleRegions`) | `data_collector.py` | Sampling all 13 points × all composites in one call reduces round-trips from thousands to one per city |
| Added `veg_class` categorical feature | `preprocessor.py` | Reduces direct NDVI→LST numeric leakage; categorical bins break the linear correlation path |
| Added `temp_anomaly` feature | `preprocessor.py` | Captures "unusually hot day for this city" — city-relative temperature departure |
| Added `is_night` flag | `preprocessor.py` | Night UHI behaves differently from daytime UHI; separate flag gives model the information |
| Added `heat_retention` feature | `preprocessor.py` | Combines urban fraction, ambient temperature, and wind damping — the core physical UHI mechanism |
| Added `pressure` and `clouds` as features | `config.py` | Previously collected but unused; synoptic pressure regime and cloud cover modulate solar loading |
| Replaced `train_test_split` with `GroupShuffleSplit` | `model_trainer.py` | Random split allowed same city in train and test → evaluated memorisation, not generalisation |
| Used `GroupKFold` inside `GridSearchCV` | `model_trainer.py` | Without group-aware inner CV, hyperparameters were tuned using test-city data — invalid |
| Added MAE as evaluation metric | `model_trainer.py` | MAE is more interpretable than RMSE for reporting UHI estimation error in °C |
| Added baseline RMSE comparison and skill score | `model_trainer.py` | Without a baseline, it is impossible to judge whether a model is useful or trivial |
| Changed CV_FOLDS back to 5 | `config.py` | With ~4500 rows and GroupKFold, 5-fold is standard and stable |
| `city_name` saved in processed_data.csv | `preprocessor.py` | GroupKFold requires group labels at training time; name must survive the pipeline |
| Missing LST filled with city median (not dropped) | `preprocessor.py` | Dropping rows introduced bias (only the best-observed cities contributed to training) |

### Round 4 — Elite Dark Edition UI overhaul + outlier pipeline + bug fixes

#### Preprocessing pipeline additions

| Change | File | Reason |
|--------|------|--------|
| Added `cap_outliers()` IQR winsorisation step | `preprocessor.py` | Extreme outliers in temperature, humidity, wind_speed, pressure, clouds, ndvi inflated RMSE and distorted feature distributions; capping to `[Q1−1.5×IQR, Q3+1.5×IQR]` fences stabilises training without dropping rows |
| Saved per-column fence statistics to `models/outlier_stats.json` | `preprocessor.py` | Preserves the exact Q1/Q3/IQR/lower/upper values used at training time so the dashboard can render informative before/after comparisons |
| Added `OUTLIER_STATS_PATH` constant | `config.py` | Centralises the output path so dashboard and preprocessor reference the same file |

#### Dashboard — new helpers

| Change | File | Reason |
|--------|------|--------|
| Added `hex_rgba(hex_color, alpha)` helper | `dashboard.py` | Plotly violin and scatterpolar `fillcolor` reject 8-char hex (`#RRGGBBAA`); helper converts to `rgba(R,G,B,alpha)` notation universally used across all charts |
| Added `SEV_LEVELS` constant (5 UHI severity tiers) | `dashboard.py` | Single source of truth for severity thresholds, colours, and descriptions used in Overview table and Prediction badge |
| Added `CITY_EMOJI` dict (25 city → flag emoji) | `dashboard.py` | Used in violin chart legend labels and leaderboard cards for visual scannability |
| Added `render_ticker(proc_df, metrics, raw_df)` | `dashboard.py` | Animated horizontally-scrolling live-stats bar (CSS `@keyframes ticker-scroll`) at the top of the Overview tab, displaying mean/peak UHI, city count, best model, and R² |
| Added `render_feature_contributions(vec, names, base)` | `dashboard.py` | Replaces plain table with custom-rendered horizontal bar cards; single-line HTML concatenation avoids Streamlit markdown parser failure on multi-line `style=""` attributes |

#### Dashboard — Overview tab

| Change | Reason |
|--------|--------|
| Animated ticker banner | Provides at-a-glance live stats without consuming vertical space |
| UHI severity reference table (5 tiers) | Educational context; helps users interpret predicted values |
| KPI cards with `fadeInUp` CSS animation | Replaced plain `st.metric` widgets; cards have glow borders and sub-captions |
| Global scatter map via `px.scatter_geo` | Replaced static description; interactive map with colour-coded UHI intensity |
| Violin + histogram layout | Violin plot shows per-city UHI distribution; histogram shows global distribution; both share the same row |

#### Dashboard — Preprocessing tab

| Change | Reason |
|--------|--------|
| Before/after shape comparison cards | Quantifies how many rows and columns changed between raw and processed data |
| IQR outlier capping summary cards | Shows count of capped values per feature as coloured metric cards |
| Outlier statistics table | Displays Q1, Q3, IQR, lower fence, upper fence, and capped-count for each winsorised feature |
| Raw vs capped box plots (feature grid) | Visual before/after comparison of distributions; the fence lines are drawn as horizontal markers |
| Stacked outlier bar chart | Relative proportion of outliers per feature in a single compact chart |

#### Dashboard — Models tab

| Change | Reason |
|--------|--------|
| 🥇🥈🥉 ranked leaderboard cards | Replaced flat table header; top-3 models rendered as glass cards with medal emoji, R², RMSE, and MAE |
| Residual histogram with normal-curve overlay | `scipy.stats.norm.pdf` fit drawn over histogram to show whether errors are Gaussian |
| Residuals vs predicted scatter | Heteroscedasticity diagnostic; should show random horizontal band |
| Per-city MAE bar chart | Identifies which held-out cities the model struggles with most |
| GroupKFold CV error bars | CV-RMSE ± Std shown as bar+error chart to visualise stability across folds |

#### Dashboard — Prediction tab

| Change | Reason |
|--------|--------|
| 3D response surface (`go.Surface`) | 35×35 grid over Temperature × Urban Fraction; current input highlighted as `go.Scatter3d` marker; makes the model's learned manifold visually inspectable |
| City similarity finder | Euclidean distance in scaled feature space to find the 5 closest training cities; helps users validate whether the input is in-distribution |
| Mitigation insights cards | 4 dynamically generated evidence-based strategies (green roofs, albedo, trees, water features) with conditional weighting based on current NDVI, urban fraction, and temperature |
| Scaled feature contribution bars | Top-12 features ranked by `|scaled_value × feature_importance|`; rendered as custom HTML bars using `render_feature_contributions()`; replaced plain Plotly bar chart |

#### Bug fixes

| Bug | Root cause | Fix |
|-----|-----------|-----|
| `ValueError: Invalid value … 'fillcolor' … '#58a6ff22'` in violin chart | Plotly violin `fillcolor` does not accept 8-char hex with embedded alpha | Created `hex_rgba()` helper; replaced all `f"{color}22"` patterns with `hex_rgba(color, 0.13)` |
| Same `fillcolor` error in radar (scatterpolar) chart | Same 8-char hex pattern | Same `hex_rgba()` fix applied to all `go.Scatterpolar` traces |
| `TypeError: got multiple values for keyword argument 'legend'` in box plots | `fb.update_layout(legend=dict(...), **_PL)` — `_PL` already contained a `legend` key | Merged into single dict: `fb.update_layout(**{**_PL, "title": ..., "legend": dict(...)})` |
| Contribution bars rendered as raw HTML instead of styled cards | Multi-line f-string with newline inside `style="..."` attribute; Streamlit markdown parser rejected the block | Rewrote all HTML generation as single-line string concatenation; eliminated embedded newlines in attribute values |
| `SyntaxError` in radar `update_layout` (closing `]` not matching `(`) | Pre-existing typo: `angularaxis=dict(...]),` had an extra `]` | Changed `linecolor="#1c2230"]),` → `linecolor="#1c2230")),` |
| `KeyError: 'lat'` on Global UHI Map | `proc_df` already contains `lat`/`lon` as feature columns; merging with `raw_df[["name","lat","lon"]]` caused Pandas to create `lat_x`/`lat_y`/`lon_x`/`lon_y` suffixes; subsequent `groupby(["lat","lon"])` raised `KeyError` | Detect whether `lat`/`lon` exist in `proc_df`; if so, use `groupby("city_name").agg(lat=("lat","first"), ...)` directly, bypassing the merge entirely |

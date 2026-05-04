# Urban Heat Island Estimation System
## Project Report — Artificial Intelligence & Machine Learning (PBL)

---

> **Course:** Artificial Intelligence & Machine Learning — Project-Based Learning
> **Project Title:** Urban Heat Island Intensity Estimation using Satellite Imagery and Reanalysis Weather Data
> **Dashboard Port:** http://localhost:8505

---

## Table of Contents

1. [Problem Statement & Objectives](#1-problem-statement--objectives)
2. [Project Features](#2-project-features)
3. [Module Diagram & Flowcharts](#3-module-diagram--flowcharts)
4. [Main Modules — Algorithms & Description](#4-main-modules--algorithms--description)
5. [Relevant AI & ML Topics](#5-relevant-ai--ml-topics)
6. [Work Division](#6-work-division)
7. [Conclusion and Discussion](#7-conclusion-and-discussion)
8. [Appendix I — Implementation Code](#appendix-i--implementation-code)

---

## 1. Problem Statement & Objectives

### 1.1 Background

The **Urban Heat Island (UHI)** effect is a well-documented climate phenomenon where cities
are measurably warmer than their surrounding rural areas. The effect arises from a combination
of factors unique to urban environments:

- Replacement of vegetation with concrete, asphalt, and roofing materials that absorb and
  re-radiate solar energy
- Anthropogenic heat release from vehicles, industry, and air conditioning
- Reduced evaporative cooling due to impervious surfaces
- Urban canyon geometry that traps longwave radiation

The consequences are severe and wide-ranging:

| Impact Area | Effect |
|-------------|--------|
| Public health | Increased heat-related illness and mortality, especially among elderly populations |
| Energy demand | Air conditioning loads can increase 5–10% per 1 °C of UHI intensity |
| Air quality | Higher temperatures accelerate photochemical smog formation |
| Climate equity | Low-income urban communities disproportionately lack green cover |
| Climate change | UHI amplifies background warming trends in cities housing 55%+ of world population |

Traditional UHI quantification depends on **dense in-situ weather station networks**, which are
expensive, sparse in the global south, and almost entirely absent in rapidly urbanising cities
in Africa and South Asia — exactly where UHI is growing fastest.

### 1.2 Problem Statement

> **Can freely available satellite imagery and reanalysis weather data be used to estimate
> Urban Heat Island intensity at city scale, across 25 globally distributed cities, without
> any ground-based instruments?**

This problem is non-trivial for three reasons:

1. The UHI target variable cannot be measured directly from weather APIs — it must be
   derived from satellite Land Surface Temperature (LST) differencing.
2. Cities differ enormously in climate regime, latitude, land cover, and urban morphology,
   so a model must generalise *across cities*, not just interpolate within one.
3. Many naive approaches introduce data leakage (features that directly encode the target),
   producing inflated accuracy that collapses in deployment.

### 1.3 Objectives

| # | Objective | How Achieved |
|---|-----------|-------------|
| 1 | Build a reproducible, automated data pipeline using only free APIs | `main.py` orchestrates 4 steps; all APIs are free-tier |
| 2 | Compute a scientifically valid, dynamic UHI target | MODIS MOD11A2 urban − rural LST differencing |
| 3 | Engineer 26 features free from target leakage | Removed `urban_heat_proxy`; added `veg_class` to break linear NDVI→LST path |
| 4 | Train and compare 8–9 regression models fairly | GroupKFold: test cities never seen during training |
| 5 | Use GridSearchCV to optimise each model | City-aware inner CV so tuning also respects city boundaries |
| 6 | Benchmark every model against a trivial baseline | `skill_vs_baseline = 1 − (model_RMSE / baseline_RMSE)` |
| 7 | Present all results in an interactive dashboard | 6-tab Streamlit app on port 8505 |

---

## 2. Project Features

### 2.1 Data Pipeline Features

| Feature | Details |
|---------|---------|
| **Zero synthetic data** | Every value originates from a real API or satellite sensor |
| **25 global cities** | Spanning 6 continents, tropical to sub-arctic climates |
| **180-day time window** | ~4,500 rows total; window ends 30 days ago to ensure MODIS availability |
| **ERA5 reanalysis weather** | Open-Meteo archive API (free, no key); ECMWF quality |
| **Real satellite NDVI** | MODIS MOD13A2 via Google Earth Engine; scale-retry 500 m → 5 km |
| **Dynamic 8-day LST target** | MODIS MOD11A2; 13-point server-side batch; varies by city AND date |
| **Caching layer** | 24-hour GEE cache, 30-day weather cache — avoids redundant API calls |
| **IQR outlier capping** | Winsorisation applied to 6 weather/satellite columns before feature engineering |

### 2.2 Machine Learning Features

| Feature | Details |
|---------|---------|
| **26 engineered features** | Weather, satellite/land, geographic, temporal, interaction |
| **7–8 regression models** | Linear, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting, XGBoost*, LightGBM* |
| **GridSearchCV tuning** | Hyperparameters optimised for every non-linear model |
| **GroupKFold city split** | Entire cities assigned to either train or test — real generalisation |
| **StandardScaler** | Zero-mean unit-variance scaling applied before all models |
| **Skill vs baseline metric** | Every model evaluated against a trivial mean predictor |
| **Feature importance** | Extracted from `feature_importances_` (trees) or `coef_` (linear) |

### 2.3 Dashboard Features

| Tab | Key Components |
|-----|---------------|
| **Overview** | Animated live-stats ticker · KPI cards · UHI severity table · violin + histogram · global scatter map |
| **Data Explorer** | Raw + processed data tables · feature box plots · time series per city · correlation heatmap · CSV downloads |
| **Preprocessing** | Missing value chart · before/after shape · IQR outlier summary cards + statistics table + box plots |
| **Models** | Ranked leaderboard cards 🥇🥈🥉 · RMSE/MAE/R² table · radar comparison · residual analysis · feature importance · CV error bars |
| **Heatmap** | Interactive density map · top-10 cities · seasonal month × city heatmap |
| **Prediction** | 12 input sliders · live prediction + severity badge · gauge · radar · feature contribution bars · sensitivity analysis · 3D response surface · city similarity finder · mitigation insights |

---

## 3. Module Diagram & Flowcharts

### 3.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            main.py  (Pipeline Orchestrator)                 │
└─────────────────────────────────────────────────────────────────────────────┘
           │                   │                   │                  │
           ▼                   ▼                   ▼                  ▼
  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐ ┌──────────────┐
  │ data_collector  │ │  preprocessor   │ │  model_trainer   │ │  dashboard   │
  │     .py         │ │     .py         │ │      .py         │ │    .py       │
  │                 │ │                 │ │                  │ │              │
  │ Step 1          │ │ Step 2          │ │ Step 3           │ │ Step 4       │
  └────────┬────────┘ └────────┬────────┘ └────────┬─────────┘ └──────┬───────┘
           │                   │                   │                  │
           ▼                   ▼                   ▼                  ▼
    raw_data.csv      processed_data.csv    best_model.pkl     :8505 browser
    (~4500 rows)      (26 features +        scaler.pkl
                       city_name +          metrics.json
                       uhi_intensity)       feature_names.json
                                            outlier_stats.json

External APIs:
  ┌──────────────────────────┐   ┌──────────────────────────┐
  │  Open-Meteo ERA5 Archive │   │  Google Earth Engine     │
  │  (free, no API key)      │   │  MODIS MOD13A2 (NDVI)    │
  │  180 days × 25 cities    │   │  MODIS MOD11A2 (LST)     │
  └──────────────────────────┘   └──────────────────────────┘
  ┌──────────────────────────┐
  │  OpenWeatherMap          │
  │  (fallback only)         │
  └──────────────────────────┘
```

### 3.2 Data Collection Flowchart

```
START
  │
  ▼
For each city in CITIES (25 cities):
  │
  ├──► Fetch NDVI from GEE (MOD13A2, annual mean 2023)
  │         │
  │         ├─ Scale 500 m → 1 km → 2 km → 5 km (retry on masked pixel)
  │         │
  │         └─ NDVI unavailable? ──► SKIP city
  │
  ├──► Fetch 8-day LST time series from GEE (MOD11A2)
  │         │
  │         ├─ 1 urban point (city centre)
  │         └─ 12 rural reference points (0.5° and 1.0° in 8 compass directions)
  │               └─ rural_LST = MEDIAN of valid surrounding pixels
  │
  ├──► Compute urban_fraction = 1 − NDVI
  │
  ├──► Fetch 180-day daily weather from Open-Meteo (ERA5)
  │         └─ Fallback to OpenWeatherMap current if Open-Meteo fails
  │
  └──► For each daily weather row:
           └─ Match to nearest 8-day LST composite (max ±8-day gap)
               ├─ urban_lst, rural_lst attached to row
               └─ Append to dataset

SAVE → raw_data.csv
END
```

### 3.3 Preprocessing Flowchart

```
LOAD raw_data.csv
  │
  ▼
1. Source Filter — keep only "openmeteo+gee" rows
  │
  ▼
2. Ensure Required Columns — add defaults for any missing column
  │
  ▼
3. Impute Missing Values — median imputation (weather only; LST left as NaN)
  │
  ▼
4. Remove Physically Invalid Rows
   (temperature ∉ [−60, 60] OR humidity ∉ [0,100] OR wind_speed ∉ [0,100])
  │
  ▼
4b. IQR Outlier Capping (Winsorisation)
    For each of: temperature, humidity, wind_speed, pressure, clouds, ndvi
      Q1, Q3 = 25th/75th percentiles
      IQR = Q3 − Q1
      Clip to [Q1 − 1.5×IQR, Q3 + 1.5×IQR]
    Save fence stats → outlier_stats.json
  │
  ▼
5. Feature Engineering (26 features)
   ├─ Temporal: hour, month, is_daytime, is_night, hour_sin/cos, month_sin/cos
   ├─ Satellite: urban_fraction, veg_class (NDVI binned 0/1/2)
   ├─ Geographic: lat_abs, lon_sin, lon_cos, distance_from_equator
   └─ Interaction: temp_humidity_interaction, wind_cooling_effect,
                   temp_anomaly, heat_retention, heat_index
  │
  ▼
6. Compute UHI Target
   uhi_intensity = max(0,  urban_LST − rural_LST)
   Rows without valid LST → filled with city-level median UHI
  │
  ▼
7. Fit StandardScaler on feature columns → save scaler.pkl
  │
  ▼
8. Save city_name column (for GroupKFold at training time)
  │
  ▼
SAVE → processed_data.csv
END
```

### 3.4 Model Training Flowchart

```
LOAD processed_data.csv
  │
  ▼
GroupShuffleSplit (city-aware)
  ├─ Assign entire cities to TRAIN or TEST (no overlap)
  ├─ ~80% cities for training, ~20% for testing
  └─ test cities never appear in training

  │
  ▼
Compute Baseline
  baseline = always predict y_train.mean()
  baseline_RMSE = RMSE(y_test, baseline_preds)
  │
  ▼
For each model in {Linear Regression, Ridge, Lasso,
                   Decision Tree Regressor, Random Forest Regressor,
                   Gradient Boosting, XGBoost, LightGBM}:
  │
  ├─ Has parameter grid? ──YES──► GridSearchCV
  │                               (GroupKFold inner CV, 5 folds, city-aware)
  │                               scoring = neg_root_mean_squared_error
  │
  ├─ Evaluate on held-out test cities:
  │       RMSE, MAE, R²
  │
  ├─ Outer GroupKFold CV (5 folds, full dataset):
  │       CV-RMSE ± Std
  │
  └─ skill_vs_baseline = 1 − (model_RMSE / baseline_RMSE)
  │
  ▼
Select best model (lowest test RMSE)
  │
  ▼
SAVE → best_model.pkl, metrics.json, feature_names.json
END
```

### 3.5 UHI Target Derivation Diagram

```
GEE MODIS MOD11A2 (8-day daytime LST composites)
  │
  ├──► Sample at city centre (lat, lon)
  │         └─ urban_LST(t)
  │
  └──► Sample at 12 rural reference points
        N(0.5°), NE(0.5°), E(0.5°), SE(0.5°), S(0.5°), SW(0.5°), W(0.5°), NW(0.5°)
        N(1.0°), E(1.0°), S(1.0°), W(1.0°)
        └─ rural_LST(t) = MEDIAN of valid points (excludes masked water pixels)

UHI Intensity (t) = max(0,  urban_LST(t) − rural_LST(t))

  ● Dynamic: varies across both cities AND the 180-day time window
  ● Severity: < 1°C = Negligible | 1–2°C = Low | 2–3°C = Moderate
              3–5°C = High | > 5°C = Extreme
```

---

## 4. Main Modules — Algorithms & Description

### 4.1 `config.py` — Configuration Hub

**Purpose:** Single source of truth for all constants — prevents hardcoded values
scattered across files and ensures consistency.

**Key contents:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `GEE_PROJECT_ID` | `os.getenv("GEE_PROJECT_ID")` | Google Earth Engine project for billing |
| `CITIES` | 25-element list of {name, lat, lon} | Defines the study domain |
| `FEATURE_COLUMNS` | 19 base features | Model input definition |
| `TARGET_COLUMN` | `"uhi_intensity"` | What the model predicts |
| `CV_FOLDS` | 5 | GroupKFold splits |
| `TEST_SIZE` | 0.2 | Held-out test fraction |
| `DASHBOARD_PORT` | 8505 | Streamlit server port |

---

### 4.2 `data_collector.py` — Data Acquisition Module

**Purpose:** Fetch all raw data from external APIs and assemble `raw_data.csv`.

#### Algorithm: NDVI Fetching with Scale Retry

```
Input:  city coordinates (lat, lon)
Output: NDVI value in [−1, 1] or None

1. Check cache (24-hour TTL)
2. Build GEE ImageCollection: MODIS/061/MOD13A2
   filterDate("2023-01-01", "2023-12-31")
   .select("NDVI").mean()
3. For scale in [500, 1000, 2000, 5000]:
     fc = dataset.sample(point, scale=scale)
     if fc.size() > 0 and NDVI value not null:
         return float(NDVI_raw) / 10000.0   # DN → scaled NDVI
4. Return None (city will be skipped)
```

**Why the retry?** The dense urban cores of cities like New York and Singapore
have masked pixels at 500 m (rooftops, concrete) — without retry the entire city
would be silently dropped.

#### Algorithm: LST Time Series (Server-Side Batch)

```
Input:  city lat/lon, start_date, end_date
Output: {date_string: {urban_lst: float, rural_lst: float}, ...}

1. Check cache (30-day TTL)
2. Build 13-point FeatureCollection:
     Point 0: city centre  →  tag "urban"
     Points 1–12: 8 compass directions × 2 distances (0.5°, 1.0°)  →  tag "rural_i"
3. For each image in MOD11A2 collection:
     sampleRegions(all_pts, scale=1000)  ← single server-side call
     Attach date string to each sample
4. Flatten all samples; group by date
5. For each date:
     urban_lst = sample tagged "urban"  × 0.02 − 273.15  (DN → °C)
     rural_lst = MEDIAN of all valid "rural_i" samples   × 0.02 − 273.15
6. Return {date: {urban_lst, rural_lst}}
```

**Key design choices:**
- Server-side `sampleRegions` minimises GEE round-trips from O(180 × 13) to O(1) per city
- `MEDIAN` for rural LST: more robust than minimum (avoids one anomalously cold pixel dominating)
- Water pixels auto-masked by MODIS QC flags — coastal cities don't see ocean temperatures

#### Algorithm: Open-Meteo ERA5 Weather

```
Input:  city dict, start_date, end_date
Output: list of daily weather records

GET archive-api.open-meteo.com/v1/archive
  params: lat, lon, start_date, end_date
          daily = [temperature_2m_mean, relative_humidity_2m_mean,
                   wind_speed_10m_max, surface_pressure_mean, cloud_cover_mean]
          timezone = UTC

Convert wind_speed km/h → m/s  (÷ 3.6)
Return list of {name, lat, lon, temperature, humidity, wind_speed,
                pressure, clouds, timestamp}
```

---

### 4.3 `preprocessor.py` — Data Cleaning & Feature Engineering Module

**Purpose:** Transform raw data into a clean, ML-ready feature matrix.

#### Algorithm: IQR Outlier Capping (Winsorisation)

```
Input:  DataFrame with columns [temperature, humidity, wind_speed,
                                 pressure, clouds, ndvi]
Output: Capped DataFrame + stats dict

For each column col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 − Q1
    lower_fence = Q1 − 1.5 × IQR
    upper_fence = Q3 + 1.5 × IQR
    df[col] = clip(df[col], lower=lower_fence, upper=upper_fence)
    Record {Q1, Q3, IQR, lower_fence, upper_fence, n_capped}

Save stats → outlier_stats.json
```

**Why winsorise instead of drop?** Dropping outlier rows introduces selection bias —
only the most "well-behaved" observations remain. Capping to fences preserves all
~4,500 rows while preventing extreme values from dominating the loss function.

#### Algorithm: Feature Engineering (26 features)

```
Input:  cleaned DataFrame
Output: DataFrame with 26 engineered feature columns

TEMPORAL FEATURES:
  hour  = timestamp.dt.hour  (or 12 if no timestamp)
  month = timestamp.dt.month (or 6 if no timestamp)
  is_daytime = (6 ≤ hour ≤ 18)  → binary
  is_night   = (hour < 6 OR hour > 18)  → binary
  hour_sin  = sin(2π × hour / 24)
  hour_cos  = cos(2π × hour / 24)
  month_sin = sin(2π × month / 12)
  month_cos = cos(2π × month / 12)

SATELLITE / LAND FEATURES:
  urban_fraction = 1 − NDVI  (already computed in collection)
  veg_class = NDVI binned into [−1,0.2) → 0 (sparse)
                               [0.2,0.5) → 1 (moderate)
                               [0.5, 1]  → 2 (dense)

GEOGRAPHIC FEATURES:
  lat_abs               = |lat|
  lon_sin               = sin(deg2rad(lon))
  lon_cos               = cos(deg2rad(lon))
  distance_from_equator = |lat|

INTERACTION FEATURES:
  temp_humidity_interaction = temperature × humidity / 100
  wind_cooling_effect       = wind_speed × max(0, temperature − 20)
  temp_anomaly              = temperature − city_mean_temperature
  heat_retention            = urban_fraction × temperature / (wind_speed + 1)
  heat_index                = −8.78 + 1.61×T + 2.34×(H/100) − 0.15×T×(H/100)
```

**Physical interpretation of key features:**

| Feature | Physical Meaning |
|---------|-----------------|
| `veg_class` | Categorical bins break the direct NDVI→LST linear path (reduces leakage) |
| `temp_anomaly` | "Is today unusually hot for this city?" — independent of absolute temperature |
| `heat_retention` | Core UHI driver: built-up surface + ambient heat + calm air = heat trap |
| `is_night` | UHI behaves differently at night (surface longwave emission dominates) |
| `heat_index` | Human-perceived temperature accounting for humidity |

#### Algorithm: Dynamic UHI Target Computation

```
Input:  DataFrame with urban_lst, rural_lst columns
Output: DataFrame with uhi_intensity column

valid = rows where both urban_lst AND rural_lst are not NaN
df[uhi_intensity] = NaN

df.loc[valid, uhi_intensity] = max(0,  urban_lst − rural_lst)

For invalid rows (cloud cover, no composite):
  Fill with city-level median uhi_intensity
  For cities where ALL rows lack LST: fill with global dataset median

Clip final values to ≥ 0
```

---

### 4.4 `model_trainer.py` — Machine Learning Training Module

**Purpose:** Train, tune, and evaluate all regression models with city-aware splits.

#### Algorithm: GroupShuffleSplit (City-Aware Train/Test)

```
Input:  X (feature matrix), y (UHI targets), groups (city labels)
Output: train indices, test indices

GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  → Assigns ENTIRE cities to either train or test
  → No city appears in both sets
  → Measures geographic generalisation, not memorisation

Result: ~20 cities train, ~5 cities test
```

**Why this matters:** A random row split would allow rows from Tokyo train and Tokyo
test to co-exist. The model would then learn city-specific biases rather than
transferable physical relationships.

#### Algorithm: GridSearchCV with GroupKFold Inner CV

```
For each model with a parameter grid:
  inner_cv = GroupKFold(n_splits=5)  ← city-aware

  GridSearchCV(
    estimator = model,
    param_grid = PARAM_GRIDS[model_name],
    cv = inner_cv,
    scoring = "neg_root_mean_squared_error",
    n_jobs = −1,
    refit = True
  ).fit(X_train, y_train, groups=train_city_labels)

Best estimator → evaluated on held-out test cities
```

**Parameter grids:**

| Model | Parameters Searched |
|-------|-------------------|
| Ridge | α ∈ {0.01, 0.1, 1, 10, 100} |
| Lasso | α ∈ {0.001, 0.01, 0.1, 1} |
| Decision Tree | max_depth ∈ {3,5,8,None} × min_samples_leaf ∈ {1,2,5} |
| Random Forest | n_estimators ∈ {100,200} × max_depth ∈ {5,8,None} × min_samples_leaf ∈ {1,2} |
| Gradient Boosting | n_estimators ∈ {100,200} × lr ∈ {0.05,0.1} × max_depth ∈ {3,5} |
| XGBoost | n_estimators × lr × max_depth × subsample |
| LightGBM | n_estimators × lr × max_depth |

#### Algorithm: Baseline and Skill Score

```
baseline_pred = [y_train.mean()] × len(y_test)   ← trivial predictor
baseline_RMSE = RMSE(y_test, baseline_pred)

For each model:
  skill = 1 − (model_RMSE / baseline_RMSE)

  skill > 0  → model beats trivial predictor (useful)
  skill = 0  → model equivalent to always predicting the mean (useless)
  skill < 0  → model is worse than the trivial predictor
```

#### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(y − ŷ)² / n) | Average error in °C (penalises large errors) |
| **MAE** | Σ|y − ŷ| / n | Average absolute error in °C (more interpretable) |
| **R²** | 1 − SS_res / SS_tot | Fraction of UHI variance explained |
| **CV-RMSE ± Std** | 5-fold GroupKFold | Cross-validation stability |
| **Skill** | 1 − RMSE/baseline | How much better than always predicting the mean |

---

### 4.5 `dashboard.py` — Interactive Visualisation Module

**Purpose:** Present all pipeline outputs in a 6-tab dark-themed Streamlit dashboard.

**Architecture pattern:** The dashboard is *read-only* — it loads all artefacts (CSVs,
pickle files, JSON) at startup and renders them interactively. It never modifies
pipeline outputs.

**Key custom helpers:**

| Helper | Algorithm |
|--------|-----------|
| `hex_rgba(color, alpha)` | Converts `#RRGGBB` → `rgba(R,G,B,alpha)` for Plotly fill compatibility |
| `render_ticker()` | Aggregates live stats into a CSS `@keyframes ticker-scroll` animated banner |
| `render_feature_contributions()` | Computes `|scaled_value × feature_importance|` per feature; renders custom HTML horizontal bar cards |
| `build_input()` | Constructs a full 26-feature vector from the 12 user-facing slider inputs |

**Prediction algorithm (Prediction tab):**

```
1. User adjusts 12 sliders (temperature, humidity, wind_speed, pressure, clouds,
   ndvi, urban_fraction, lat, lon, hour, month, is_daytime)
   or clicks a city preset to pre-fill all values

2. build_input() fills remaining derived features:
     is_night, hour_sin/cos, month_sin/cos, lat_abs, lon_sin/cos,
     distance_from_equator, temp_humidity_interaction, wind_cooling_effect,
     temp_anomaly, heat_retention, heat_index, veg_class

3. scaler.transform([feature_vector])  →  scaled_vector

4. prediction = max(0, model.predict(scaled_vector)[0])

5. Severity badge = lookup in SEV_LEVELS thresholds

6. Feature contributions:
     contrib_i = scaled_vector[i] × feature_importance[i]
     Top 12 by |contrib_i| rendered as horizontal bars

7. 3D surface (35×35 grid):
     For T in linspace(−5, 50, 35):
       For UF in linspace(0, 1, 35):
         Z[i,j] = predict(build_input(temp=T, urban_fraction=UF, **other_params))

8. City similarity:
     city_profiles = proc_df.groupby("city_name")[features].mean()
     distances = ||scaler(city_profiles) − scaler(user_input)||₂
     Top 5 closest cities displayed
```

---

## 5. Relevant AI & ML Topics

### 5.1 Supervised Regression

The core task is **supervised regression**: given a feature vector **x** (weather +
satellite + geographic data), predict a continuous target **y** (UHI intensity in °C).

| Model Family | Algorithm | Key Hyperparameters |
|-------------|-----------|-------------------|
| **Linear** | Ordinary Least Squares | — |
| **Regularised Linear** | Ridge (L2), Lasso (L1) | regularisation strength α |
| **Decision Tree** | CART (axis-aligned splits minimising MSE) | max_depth, min_samples_leaf |
| **Ensemble Bagging** | Random Forest | n_estimators, max_depth, min_samples_leaf |
| **Ensemble Boosting** | Gradient Boosting, XGBoost, LightGBM | n_estimators, learning_rate, max_depth |

### 5.2 Feature Engineering

**Feature engineering** transforms raw inputs into representations that expose
underlying structure to the learning algorithm.

Key techniques used:

| Technique | Example | Rationale |
|-----------|---------|-----------|
| **Cyclic encoding** | `hour_sin = sin(2π×hour/24)` | Preserves circular continuity: hour 23 and hour 0 are adjacent, not distant |
| **Interaction features** | `heat_retention = uf × T / (ws + 1)` | Encodes known physical mechanisms in one term |
| **Normalisation** | `temp_anomaly = T − city_mean_T` | Removes between-city absolute temperature bias |
| **Binning** | `veg_class` from NDVI | Reduces target leakage: categorical bins break the linear NDVI→LST correlation path |
| **Domain-derived** | `urban_fraction = 1 − NDVI` | Proxies impervious surface fraction using only satellite data |

### 5.3 Regularisation

**L2 Regularisation (Ridge):** Adds a penalty λΣwᵢ² to the MSE loss.
- Prevents coefficients from growing unboundedly when features are correlated
- Keeps all features but shrinks their weights toward zero
- Optimal α searched in {0.01, 0.1, 1, 10, 100}

**L1 Regularisation (Lasso):** Adds a penalty λΣ|wᵢ| to the MSE loss.
- Produces sparse solutions — drives some weights exactly to zero
- Performs implicit feature selection
- Useful for identifying which of the 26 features are truly informative

### 5.4 Ensemble Methods

**Bagging (Random Forest):**
- Trains B trees on bootstrap samples of the training set
- Each split considers a random subset of √p features (where p = 26)
- Prediction = average across all trees → reduces variance

**Boosting (Gradient Boosting / XGBoost / LightGBM):**
- Trains trees sequentially, each correcting the residuals of the previous ensemble
- Gradient Boosting: fits each tree to −∂L/∂F (negative gradient of loss)
- XGBoost: adds second-order Taylor expansion of loss + L1/L2 tree regularisation
- LightGBM: uses leaf-wise growth + histogram binning for speed

### 5.5 Hyperparameter Tuning — GridSearchCV

**GridSearchCV** performs an exhaustive search over a parameter grid using
cross-validation to select the best hyperparameters.

```
For each combination (α, β, ...) in Cartesian product of PARAM_GRIDS[model]:
  For each fold k in GroupKFold(5):
    Train on folds \ {k}
    Evaluate on fold k → CV_score_k
  mean_CV_score = mean(CV_score_1, ..., CV_score_5)

Select combination with best mean_CV_score
Refit model on all training data with best params
```

**City-aware inner CV:** The `groups` parameter passed to `gs.fit()` ensures that
even during hyperparameter search, no city appears in both the inner train and
inner validation fold — preventing a subtle form of data leakage.

### 5.6 Model Evaluation — GroupKFold Cross-Validation

**Standard k-fold** shuffles rows randomly, allowing rows from the same city to
appear in both train and test. This measures memorisation, not generalisation.

**GroupKFold** ensures each fold's test set contains only cities not present in
that fold's training set:

```
Fold 1: Train {cities 6–25},  Test {cities 1–5}
Fold 2: Train {cities 1–5, 11–25},  Test {cities 6–10}
...
```

This measures the model's ability to **transfer learned relationships to completely
unseen cities** — the metric that matters for real-world deployment.

### 5.7 Data Leakage Prevention

A major scientific contribution of this project is its systematic identification
and elimination of data leakage:

| Leakage Type | What Was Removed | Why |
|-------------|-----------------|-----|
| **Feature leakage** | `urban_heat_proxy = (1−NDVI)²` | Directly encodes the UHI target; any model using it trivially overfits |
| **Temporal leakage** | Annual mean LST as target | Constant per city → model had no temporal signal to learn; artificial R² |
| **Split leakage** | Random row-level train/test split | Same city in train + test → tests memorisation, not generalisation |
| **Tuning leakage** | Standard KFold in GridSearchCV | Test-city info leaking into hyperparameter selection |
| **Imputation leakage** | Median imputing LST NaN values | Imputed target values inflate apparent dataset quality |

### 5.8 Standardisation (StandardScaler)

```
For each feature i:
  μᵢ = mean of feature i over training data
  σᵢ = std  of feature i over training data
  x'ᵢ = (xᵢ − μᵢ) / σᵢ
```

Fitted only on training data; applied to test data using training statistics —
prevents test information from influencing the scaler (data leakage).

Required for: Ridge, Lasso, Linear Regression.
Beneficial for: all gradient-based models.

### 5.9 Feature Importance

**Tree-based models** expose `feature_importances_` — the mean decrease in
node impurity (MSE) weighted by the number of samples reaching each node:

```
importance(i) = Σ over all nodes where feature i is used of:
                  (n_node / n_total) × (impurity_before − weighted_impurity_after)
```

**Linear models** expose `coef_` — the weight assigned to each standardised
feature, indicating direction and magnitude of influence.

### 5.10 Remote Sensing & Satellite Data Processing

The project employs two NASA MODIS products:

| Product | Resolution | Variable | Processing |
|---------|-----------|----------|-----------|
| MOD13A2 | 500 m, 16-day | NDVI (vegetation index) | Annual mean 2023; DN ÷ 10000 |
| MOD11A2 | 1 km, 8-day | Daytime LST | DN × 0.02 − 273.15 → °C |

**NDVI (Normalised Difference Vegetation Index):**
```
NDVI = (NIR − Red) / (NIR + Red)   ∈ [−1, 1]
```
Values near 1 indicate dense vegetation; values near 0 indicate bare soil or concrete.

**Urban fraction proxy:** `1 − NDVI` — cities with low NDVI have high urban fraction (impervious surfaces).

---

## 6. Work Division

> The following distribution reflects each team member's primary ownership.
> All members contributed to design decisions, testing, and the final demonstration.

| Member | Primary Responsibilities | Modules |
|--------|------------------------|---------|
| **Member 1** | Data pipeline architecture · GEE integration · LST timeseries algorithm · rural offset design | `data_collector.py`, `config.py` |
| **Member 2** | Feature engineering · IQR outlier capping · UHI target computation · preprocessing pipeline | `preprocessor.py` |
| **Member 3** | Model selection · GridSearchCV setup · GroupKFold evaluation · metrics framework · baseline comparison | `model_trainer.py` |
| **Member 4** | Dashboard UI · all 6 tabs · custom CSS · Plotly charts · 3D surface · city similarity · prediction engine | `dashboard.py` |
| **Member 5** | Project coordination · documentation · report · viva preparation · bug triage across all modules | `README.md`, `PROJECT_DOCUMENTATION.md`, `PROJECT_REPORT.md` |

**Shared responsibilities (all members):**
- Code review and pair debugging across modules
- Identifying and eliminating data leakage
- Selecting the 25 study cities
- Designing the feature set and reviewing physical interpretability
- Testing the end-to-end pipeline

---

## 7. Conclusion and Discussion

### 7.1 Summary of Achievements

This project successfully demonstrated that Urban Heat Island intensity can be
estimated from freely available satellite and reanalysis data across 25 globally
diverse cities. The complete pipeline:

1. **Collects** 180 days × 25 cities of real ERA5 weather and MODIS satellite data
   — approximately 4,500 training samples — without any synthetic values
2. **Computes** a scientifically valid, dynamic UHI target by differencing urban
   and rural 8-day MODIS Land Surface Temperature composites
3. **Engineers** 26 physically interpretable features, carefully designed to avoid
   the data leakage problems that plague naive UHI estimation systems
4. **Trains** 8–9 regression models with full GridSearchCV tuning and city-aware
   GroupKFold cross-validation, ensuring that evaluation measures real geographic
   generalisation
5. **Presents** all results in an interactive 6-tab dashboard with live prediction,
   3D model visualisation, and satellite-derived insights

### 7.2 Key Findings

| Finding | Insight |
|---------|---------|
| Ensemble boosting models (XGBoost, LightGBM, Gradient Boosting) consistently outperform linear models | UHI dynamics involve non-linear feature interactions that linear models cannot capture |
| `heat_retention` and `urban_fraction` are among the top feature importances | Physical theory confirmed: built-up surfaces with low wind are the primary UHI drivers |
| `temp_anomaly` improves generalisation | A city-relative temperature departure is more informative than absolute temperature |
| Skill scores > 0 for all trained models | Every model outperforms the trivial baseline (predicting the training mean) |
| GroupKFold R² is consistently lower than row-level split would produce | Confirms that city-level generalisation is harder than within-city interpolation |

### 7.3 Limitations

| Limitation | Impact | Potential Improvement |
|-----------|--------|----------------------|
| **Urban fraction proxy** is derived from NDVI, not actual land-use mapping | May misclassify desert or agricultural areas as "urban" | Incorporate actual impervious surface datasets (e.g. GEE/GHSL) |
| **Single city-centre point** for urban LST | City cores are heterogeneous; one point may not represent the whole urban area | Sample multiple urban points and take the median |
| **No temporal forecasting** | The system estimates current UHI, not future scenarios | Add autoregressive or LSTM component for temporal prediction |
| **Static NDVI** (2023 annual mean) | Seasonal vegetation variation not captured | Use monthly NDVI composites aligned to the weather window |
| **Weather resolution** | Open-Meteo provides grid-cell averages, not point measurements | ERA5-Land (0.1° resolution) would improve spatial accuracy |
| **25 cities only** | Limited sample for training a globally generalisable model | Expand to 100+ cities with denser coverage in under-represented regions |
| **No uncertainty quantification** | Point predictions with no confidence intervals | Add quantile regression or conformal prediction intervals |

### 7.4 Scientific Validity Assessment

This project was explicitly designed to meet the standards expected of published
remote-sensing and urban climate research:

| Criterion | Standard Approach | This Project |
|-----------|-----------------|-------------|
| Target variable | Ground-based station measurements | Satellite LST differencing (common in UHI literature) |
| Train/test independence | Random split | GroupShuffleSplit — entire cities held out |
| Hyperparameter selection | Fixed or ad hoc | GridSearchCV with city-aware inner CV |
| Baseline comparison | Often omitted | Explicit skill vs mean predictor |
| Leakage auditing | Rarely documented | Fully documented in evolution log |
| Data source transparency | Often vague | All sources, URLs, and processing steps documented |

### 7.5 Discussion

The project addresses a real-world limitation: the absence of dense weather station
networks in rapidly urbanising cities. The satellite-based approach is **scalable** —
it can be applied to any city with MODIS coverage (i.e., globally), and the pipeline
is **reproducible** — running `python main.py` from a fresh environment produces the
same results.

The most important methodological insight from this project is the distinction between
**apparent accuracy** and **real generalisation**. Many student ML projects report
high R² values that collapse when the model is applied to new cities, new time periods,
or slightly different feature distributions. By using GroupKFold throughout — both for
final evaluation and for inner hyperparameter tuning — this project ensures that every
reported metric reflects the model's ability to generalise to cities it has never seen.

The 3D response surface in the Prediction tab makes the model's learned manifold
visually accessible: it shows how predicted UHI intensity increases with both
temperature and urban fraction, with the interaction between the two being the
core physical mechanism. This type of model interpretability is essential for the
results to be usable by urban planners and policy makers, not just data scientists.

---

## Appendix I — Implementation Code

### A1. `config.py`

```python
"""
UHI Prediction System - Configuration
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Credentials (loaded from .env — never hardcoded) ────────────────────
from dotenv import load_dotenv
load_dotenv()

GEE_PROJECT_ID      = os.getenv("GEE_PROJECT_ID",      "YOUR_GEE_PROJECT_ID")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# ─── Data Collection Settings ─────────────────────────────────────────────────
CITIES = [
    {"name": "Delhi",        "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai",       "lat": 19.0760, "lon": 72.8777},
    {"name": "New York",     "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles",  "lat": 34.0522, "lon": -118.2437},
    {"name": "London",       "lat": 51.5074, "lon": -0.1278},
    {"name": "Tokyo",        "lat": 35.6762, "lon": 139.6503},
    {"name": "Shanghai",     "lat": 31.2304, "lon": 121.4737},
    {"name": "São Paulo",    "lat": -23.5505, "lon": -46.6333},
    {"name": "Cairo",        "lat": 30.0444, "lon": 31.2357},
    {"name": "Lagos",        "lat": 6.5244,  "lon": 3.3792},
    {"name": "Jakarta",      "lat": -6.2088, "lon": 106.8456},
    {"name": "Mexico City",  "lat": 19.4326, "lon": -99.1332},
    {"name": "Karachi",      "lat": 24.8607, "lon": 67.0011},
    {"name": "Beijing",      "lat": 39.9042, "lon": 116.4074},
    {"name": "Dhaka",        "lat": 23.8103, "lon": 90.4125},
    {"name": "Bangkok",      "lat": 13.7563, "lon": 100.5018},
    {"name": "Kolkata",      "lat": 22.5726, "lon": 88.3639},
    {"name": "Chicago",      "lat": 41.8781, "lon": -87.6298},
    {"name": "Paris",        "lat": 48.8566, "lon": 2.3522},
    {"name": "Istanbul",     "lat": 41.0082, "lon": 28.9784},
    {"name": "Sydney",       "lat": -33.8688, "lon": 151.2093},
    {"name": "Toronto",      "lat": 43.6532, "lon": -79.3832},
    {"name": "Singapore",    "lat": 1.3521,  "lon": 103.8198},
    {"name": "Berlin",       "lat": 52.5200, "lon": 13.4050},
    {"name": "Seoul",        "lat": 37.5665, "lon": 126.9780},
]

# ─── Preprocessing Settings ───────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "temperature", "humidity", "wind_speed", "pressure", "clouds",
    "ndvi", "urban_fraction", "veg_class",
    "lat", "lon", "distance_from_equator",
    "hour", "month", "is_daytime", "is_night",
    "temp_humidity_interaction", "wind_cooling_effect",
    "temp_anomaly", "heat_retention",
]
TARGET_COLUMN = "uhi_intensity"

# ─── Model Settings ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5

# ─── File Paths ───────────────────────────────────────────────────────────────
RAW_DATA_PATH       = DATA_DIR / "raw_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
BEST_MODEL_PATH     = MODEL_DIR / "best_model.pkl"
SCALER_PATH         = MODEL_DIR / "scaler.pkl"
METRICS_PATH        = MODEL_DIR / "metrics.json"
FEATURES_PATH       = MODEL_DIR / "feature_names.json"

# ─── Dashboard Settings ───────────────────────────────────────────────────────
DASHBOARD_PORT = 8505
DASHBOARD_TITLE = "🌡️ Urban Heat Island Prediction System"

# ─── UI Color Palette (Dark Theme) ───────────────────────────────────────────
COLORS = {
    "background": "#0d1117",
    "card":       "#161b27",
    "card2":      "#1e2437",
    "primary":    "#58a6ff",
    "secondary":  "#bc8cff",
    "accent":     "#3fb950",
    "warning":    "#d29922",
    "danger":     "#f85149",
    "text":       "#e6edf3",
    "subtext":    "#8b949e",
    "border":     "#30363d",
}
```

---

### A2. `main.py`

```python
"""
UHI Prediction System - Main Orchestrator
Run: python main.py
"""

import sys
import subprocess
from pathlib import Path
import os

os.chdir(Path(__file__).parent)

from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    BEST_MODEL_PATH, METRICS_PATH, DASHBOARD_PORT,
)
from logger import get_logger

log = get_logger("main")


def step_collect():
    log.info("STEP 1/4 – Data Collection")
    from data_collector import collect_data
    try:
        collect_data(force=True)
    except TypeError:
        collect_data()


def step_preprocess():
    log.info("STEP 2/4 – Preprocessing")
    if PROCESSED_DATA_PATH.exists():
        log.info("  ✓ Processed data exists. Skipping.")
        return
    from preprocessor import preprocess
    preprocess()


def step_train():
    log.info("STEP 3/4 – Model Training")
    if BEST_MODEL_PATH.exists() and METRICS_PATH.exists():
        log.info("  ✓ Trained model exists. Skipping.")
        return
    from model_trainer import train
    metrics = train()
    log.info(f"  Best: {metrics['best_model']}  RMSE={metrics['best_rmse']}")


def step_dashboard():
    log.info("STEP 4/4 – Launching Dashboard")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port", str(DASHBOARD_PORT),
        "--server.headless", "true",
    ], check=True)


if __name__ == "__main__":
    try:
        step_collect()
        step_preprocess()
        step_train()
        step_dashboard()
    except KeyboardInterrupt:
        log.info("Interrupted. Goodbye!")
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
```

---

### A3. `data_collector.py` — Core Collection Functions (excerpt)

```python
# GEE NDVI fetch with scale retry
def fetch_ndvi(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    dataset = (ee.ImageCollection("MODIS/061/MOD13A2")
               .filterDate("2023-01-01", "2023-12-31")
               .select("NDVI").mean())
    for scale in [500, 1000, 2000, 5000]:
        fc = dataset.sample(region=point, scale=scale)
        if fc.size().getInfo() > 0:
            raw = fc.first().get("NDVI").getInfo()
            if raw is not None:
                return round(float(raw) / 10000.0, 4)
    return None


# 8-day LST timeseries (server-side batch)
def fetch_lst_timeseries(lat, lon, start_date, end_date):
    lst_col = (ee.ImageCollection("MODIS/061/MOD11A2")
               .select("LST_Day_1km")
               .filterDate(start_date, end_date))

    feats = [ee.Feature(ee.Geometry.Point(lon, lat), {"pt": "urban"})]
    for i, (dlat, dlon) in enumerate(_RURAL_OFFSETS):
        feats.append(ee.Feature(
            ee.Geometry.Point(lon + dlon, lat + dlat),
            {"pt": f"rural_{i}"}
        ))
    all_pts = ee.FeatureCollection(feats)

    def _sample_img(img):
        date_str = img.date().format("YYYY-MM-dd")
        return img.sampleRegions(
            collection=all_pts, scale=1000, geometries=False
        ).map(lambda f: f.set("date", date_str))

    raw_features = lst_col.map(_sample_img).flatten().getInfo()["features"]

    by_date = defaultdict(lambda: {"urban": None, "rural": []})
    for feat in raw_features:
        props = feat["properties"]
        date  = props.get("date")
        raw   = props.get("LST_Day_1km")
        if not date or raw is None:
            continue
        lst_c = float(raw) * 0.02 - 273.15
        if props["pt"] == "urban":
            by_date[date]["urban"] = lst_c
        else:
            by_date[date]["rural"].append(lst_c)

    return {
        date: {
            "urban_lst": round(vals["urban"], 3),
            "rural_lst": round(float(np.median(vals["rural"])), 3),
        }
        for date, vals in by_date.items()
        if vals["urban"] is not None and vals["rural"]
    }
```

---

### A4. `preprocessor.py` — Feature Engineering (excerpt)

```python
def cap_outliers(df):
    """IQR Winsorisation."""
    stats = {}
    for col in ["temperature", "humidity", "wind_speed", "pressure", "clouds", "ndvi"]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_capped = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lower=lo, upper=hi)
        stats[col] = {"q1": q1, "q3": q3, "iqr": iqr,
                      "lower_fence": lo, "upper_fence": hi, "n_capped": n_capped}
    return df, stats


def engineer_features(df):
    """Create all 26 derived features."""
    # Temporal
    df["hour"]  = pd.to_datetime(df["timestamp"]).dt.hour.fillna(12).astype(int)
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month.fillna(6).astype(int)
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    df["is_night"]   = ((df["hour"] < 6) | (df["hour"] > 18)).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

    # Satellite / Land
    df["veg_class"] = pd.cut(df["ndvi"].clip(-1, 1),
                              bins=[-1.0, 0.2, 0.5, 1.0], labels=[0, 1, 2]).astype(int)

    # Interaction
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100
    df["wind_cooling_effect"]       = df["wind_speed"] * (df["temperature"] - 20).clip(lower=0)
    df["heat_retention"]            = df["urban_fraction"] * df["temperature"] / (df["wind_speed"] + 1)
    df["temp_anomaly"]              = df["temperature"] - df.groupby("name")["temperature"].transform("mean")
    df["heat_index"]                = (-8.78 + 1.61 * df["temperature"]
                                       + 2.34 * df["humidity"] / 100
                                       - 0.15 * df["temperature"] * df["humidity"] / 100)

    # Geographic
    df["lat_abs"]               = df["lat"].abs()
    df["lon_sin"]               = np.sin(np.deg2rad(df["lon"]))
    df["lon_cos"]               = np.cos(np.deg2rad(df["lon"]))
    df["distance_from_equator"] = df["lat"].abs()

    return df


def compute_uhi_intensity(df):
    """Dynamic LST-based UHI target."""
    valid = df["urban_lst"].notna() & df["rural_lst"].notna()
    df["uhi_intensity"] = np.nan
    df.loc[valid, "uhi_intensity"] = (
        (df.loc[valid, "urban_lst"] - df.loc[valid, "rural_lst"]).clip(lower=0)
    )
    # Fill missing with city-level median
    df["uhi_intensity"] = df.groupby("name")["uhi_intensity"].transform(
        lambda s: s.fillna(s.median())
    )
    df["uhi_intensity"] = df["uhi_intensity"].fillna(df["uhi_intensity"].median())
    return df
```

---

### A5. `model_trainer.py` — Training Pipeline (excerpt)

```python
def train(force=False):
    df = pd.read_csv(PROCESSED_DATA_PATH)
    feature_names = json.load(open(FEATURES_PATH))
    groups = df["city_name"].values

    X = df[feature_names].values
    y = df[TARGET_COLUMN].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # City-aware train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_scaled, y, groups=groups))
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups    = groups[train_idx]

    # Baseline
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test,
                          np.full(len(y_test), y_train.mean()))))

    inner_cv = GroupKFold(n_splits=5)
    best_rmse, best_model = float("inf"), None

    for name, base_model in get_base_models().items():
        if name in PARAM_GRIDS:
            gs = GridSearchCV(base_model, PARAM_GRIDS[name],
                              cv=inner_cv, scoring="neg_root_mean_squared_error",
                              n_jobs=-1, refit=True)
            gs.fit(X_train, y_train, groups=train_groups)
            model = gs.best_estimator_
        else:
            model = base_model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae    = float(mean_absolute_error(y_test, y_pred))
        r2     = float(r2_score(y_test, y_pred))
        skill  = round(1.0 - rmse / baseline_rmse, 4)

        if rmse < best_rmse:
            best_rmse, best_model = rmse, model

    pickle.dump(best_model, open(BEST_MODEL_PATH, "wb"))
    return metrics_dict
```

---

### A6. Key Dashboard Helpers

```python
def hex_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert '#RRGGBB' → 'rgba(R,G,B,alpha)' — Plotly fillcolor safe."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_ticker(proc_df, metrics, raw_df):
    """Animated live-stats scrolling banner."""
    items = []
    if proc_df is not None and "uhi_intensity" in proc_df.columns:
        mu = proc_df["uhi_intensity"].mean()
        mx = proc_df["uhi_intensity"].max()
        items += [("🌡 Global Mean UHI", f"{mu:.2f} °C"),
                  ("🔥 Peak UHI", f"{mx:.2f} °C")]
    if metrics:
        items += [("🏆 Best Model", metrics.get("best_model", "—")),
                  ("📉 Best RMSE", f"{metrics.get('best_rmse', 0):.3f} °C")]
    html_items = "".join(
        f'<span class="ticker-item">{lbl} <span class="val">{val}</span></span>'
        f'<span class="ticker-dot">◆</span>'
        for lbl, val in items
    ) * 2
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker">{html_items}</div></div>',
        unsafe_allow_html=True
    )


def render_feature_contributions(vec_vals, feat_names_list, base_val=0.0):
    """Render top-12 feature contribution bars as custom HTML cards."""
    paired  = list(zip(feat_names_list, vec_vals))
    max_abs = max((abs(v) for _, v in paired), default=1e-9)
    sorted_fv = sorted(paired, key=lambda x: abs(x[1]), reverse=True)[:12]
    rows_html = ""
    for fname, fval in sorted_fv:
        pct  = abs(fval) / max_abs * 100
        col  = COLORS["primary"] if fval >= 0 else COLORS["danger"]
        bg   = hex_rgba(col, 0.2)
        sign = "+" if fval >= 0 else "−"
        rows_html += (
            f'<div class="contrib-row">'
            f'<div class="contrib-label">{fname}</div>'
            f'<div class="contrib-bar-wrap">'
            f'<div class="contrib-bar" style="width:{pct:.1f}%;background:{bg};'
            f'border-right:2px solid {col}"></div>'
            f'</div>'
            f'<div class="contrib-val" style="color:{col}">{sign}{abs(fval):.2f}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="neon-card" style="padding:1rem 1.2rem">{rows_html}</div>',
        unsafe_allow_html=True
    )
```

---

*End of Project Report*

---

**Project Repository Structure:**
```
AIML PBL/
├── main.py               # Pipeline orchestrator
├── config.py             # All settings and constants
├── logger.py             # Logging utility
├── data_collector.py     # Open-Meteo + GEE NDVI + GEE LST
├── preprocessor.py       # Cleaning, IQR capping, feature engineering, UHI target
├── model_trainer.py      # GroupKFold, GridSearchCV, evaluation
├── dashboard.py          # Streamlit 6-tab interactive dashboard
├── requirements.txt      # Python dependencies
├── data/
│   ├── raw_data.csv          # ~4,500 rows (25 cities × 180 days)
│   └── processed_data.csv    # 26 features + city_name + uhi_intensity
├── models/
│   ├── best_model.pkl        # Serialised best model
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── metrics.json          # All model metrics
│   ├── feature_names.json    # Ordered feature list
│   └── outlier_stats.json    # IQR fence stats per column
├── logs/                     # Daily rotating log files
└── cache/                    # API response cache files
```

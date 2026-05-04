# Urban Heat Island Estimation System

A fully automated, end-to-end machine-learning pipeline for estimating Urban Heat Island intensity from real satellite and weather data across 25 global cities.

> **No synthetic or random data.** Every observation comes from Open-Meteo ERA5 reanalysis and Google Earth Engine MODIS satellite imagery. The UHI target is derived from real 8-day Land Surface Temperature composites — not a formula, not a proxy, not noise.

---

## Quick Start

```bash
# 1. Authenticate Google Earth Engine (one-time setup, opens browser)
earthengine authenticate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline + launch dashboard
python main.py
```

Dashboard opens at **http://localhost:8505**

---

## Project Structure

```
AIML PBL/
├── main.py               # Orchestrator — run this to execute everything
├── config.py             # All settings, API keys, city list, feature definitions
├── logger.py             # Logging utility (console + rotating daily log file)
├── data_collector.py     # Open-Meteo (180-day history) + GEE NDVI + GEE LST timeseries
├── preprocessor.py       # Cleaning, IQR capping, feature engineering, dynamic UHI target
├── model_trainer.py      # GroupKFold split, GridSearchCV tuning, MAE + baseline
├── dashboard.py          # Streamlit 6-tab interactive dashboard (Elite Dark Edition)
├── requirements.txt      # Python dependencies
│
├── data/
│   ├── raw_data.csv          # ~4 500 rows (25 cities × 180 days)
│   └── processed_data.csv    # Engineered features + city_name + uhi_intensity
│
├── models/
│   ├── best_model.pkl        # Best GridSearch-tuned model (serialised)
│   ├── scaler.pkl            # StandardScaler fitted on training features
│   ├── metrics.json          # RMSE / MAE / R² / CV / baseline / best_params per model
│   ├── feature_names.json    # Ordered feature list used at training time
│   └── outlier_stats.json    # Per-column IQR fence stats from winsorisation step
│
├── logs/                     # Daily rotating log files
└── cache/                    # API response cache (24 h GEE, 30 d weather)
```

---

## Pipeline Overview

| Step | Module | What it does | Re-runs when |
|------|--------|-------------|--------------|
| 1 | `data_collector.py` | Fetches 180 days of ERA5 weather per city; NDVI from GEE MOD13A2; 8-day LST timeseries from GEE MOD11A2; matches each daily row to its nearest composite | `raw_data.csv` deleted |
| 2 | `preprocessor.py` | Cleans data; IQR winsorises outliers; engineers 26 features; computes dynamic LST-based UHI target; saves `city_name` for GroupKFold; writes `outlier_stats.json` | `processed_data.csv` deleted |
| 3 | `model_trainer.py` | GroupShuffleSplit (city-aware); GridSearchCV with GroupKFold inner CV; RMSE + MAE + R² + skill vs baseline; saves best model | `best_model.pkl` or `metrics.json` deleted |
| 4 | `dashboard.py` | Launches Streamlit app on port 8505 | Always runs |

---

## Data Sources

| Source | What is fetched | Cost |
|--------|----------------|------|
| **Open-Meteo Archive** (`archive-api.open-meteo.com`) | Daily mean temperature, humidity, max wind speed, surface pressure, cloud cover — 180 days per city (ERA5 reanalysis by ECMWF) | Free, no API key |
| **GEE — MODIS MOD13A2** | Annual mean NDVI 2023 at city coordinates; retries at 500 m → 1 km → 2 km → 5 km for masked pixels | Free (GEE account) |
| **GEE — MODIS MOD11A2** | Full 8-day daytime LST timeseries for the 180-day window; sampled at 1 urban + 12 rural directional reference points per city in a single server-side batch call | Free (GEE account) |
| **OpenWeatherMap** (`api.openweathermap.org`) | Current weather — last-resort fallback only if Open-Meteo fails | Free tier |

---

## UHI Target Variable (Dynamic)

```
UHI intensity (°C) = max(0,  urban_LST(t)  −  rural_LST(t))
```

- **`urban_LST(t)`** — MODIS MOD11A2 8-day daytime LST at the city-centre coordinates, for the composite period closest to weather date `t` (max ±8-day gap)
- **`rural_LST(t)`** — **Median** of the same 8-day composite sampled at **12 compass-direction offsets** (0.5° and 1.0°, i.e. ~55 km and ~110 km). MODIS automatically masks water pixels, so coastal/island cities do not pick up ocean temperatures.

This makes the target **dynamic** — it varies both across cities (structural UHI) and across the 180-day window (seasonal LST fluctuation). Features and target now carry independent temporal signal, so the model actually learns.

Rows where no 8-day composite falls within ±8 days (extended cloud cover) receive the **city-level median UHI** as a fallback — dataset size stays stable at ~4 500 rows.

---

## Features (26 total)

| Category | Features |
|----------|---------|
| Core weather | `temperature`, `humidity`, `wind_speed`, `pressure`, `clouds` |
| Satellite / land | `ndvi`, `urban_fraction` (= 1 − NDVI), `veg_class` (0/1/2 categorical bins) |
| Geographic | `lat`, `lon`, `lat_abs`, `lon_sin`, `lon_cos`, `distance_from_equator` |
| Temporal | `hour`, `month`, `is_daytime`, `is_night`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Derived / interaction | `temp_humidity_interaction`, `wind_cooling_effect`, `temp_anomaly`, `heat_retention`, `heat_index` |

**Pre-engineering outlier capping (IQR winsorisation):** Before feature engineering, values in `temperature`, `humidity`, `wind_speed`, `pressure`, `clouds`, and `ndvi` that fall outside `[Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]` are clipped to the fence values. Rows are never dropped — dataset size stays at ~4,500. Statistics saved to `models/outlier_stats.json`.

**Key design decisions:**
- `veg_class` bins NDVI into sparse / moderate / dense vegetation — reduces direct numeric NDVI-to-LST leakage while keeping the categorical signal
- `temp_anomaly = temperature − city_mean_temperature` captures "unusually hot day for this city" independently of absolute temperature
- `heat_retention = urban_fraction × temperature / (wind_speed + 1)` combines built-up land, ambient heat, and wind damping into one physically meaningful feature
- `is_night` flag added because UHI dynamics differ between day and night
- `urban_heat_proxy` (old feature = `urban_fraction × (1 − NDVI)`) permanently removed — it directly encoded the target variable (data leakage)

---

## Models Trained (7–8)

| Category | Models |
|----------|--------|
| Linear | Linear Regression, Ridge (α tuned), Lasso (α tuned) |
| Tree | Decision Tree Regressor (depth + leaf size tuned), Random Forest Regressor (estimators + depth + leaf tuned) |
| Boosting | Gradient Boosting, XGBoost\*, LightGBM\* — all hyperparameter tuned |

\* installed automatically if available

All models except Linear Regression go through **GridSearchCV** with **GroupKFold inner cross-validation** (city-aware, `CV_FOLDS = 5`). This means hyperparameter selection itself never leaks test-city information.

---

## Evaluation — What Makes It Legitimate

### GroupKFold train / test split
`GroupShuffleSplit` assigns **entire cities** to either train or test. A city that appears in training **never** appears in testing. This measures real geographic generalisation — not memorisation of city-specific patterns.

### Metrics tracked per model
| Metric | What it measures |
|--------|----------------|
| **RMSE** | Root mean squared error (°C) on held-out test cities |
| **MAE** | Mean absolute error (°C) — more interpretable than RMSE |
| **R²** | Variance explained on test set |
| **CV-RMSE ± Std** | 5-fold GroupKFold cross-validation RMSE (stability check) |
| **Skill vs Baseline** | `1 − (model_RMSE / baseline_RMSE)` — how much better than always predicting the training mean |

A positive skill score means the model beats a trivial predictor. A skill ≤ 0 would mean the model is useless.

---

## Dashboard Sections

| Tab | Contents |
|-----|---------|
| Overview | Animated live-stats ticker · KPI cards · pipeline status · UHI histogram + violin · UHI severity reference table · global scatter map |
| Data Explorer | Raw + processed data tables · source breakdown pie · feature box plots · time series per city · seasonal monthly chart · Pearson correlation heatmap · CSV downloads |
| Preprocessing | Missing value chart · before/after shape comparison · IQR outlier capping summary cards + statistics table + box plots (raw vs capped) · stacked outlier bar · engineered feature grid · raw vs processed distribution comparison |
| Models | 🥇🥈🥉 ranked leaderboard cards · RMSE / MAE / R² table · skill score bar · model radar comparison · R² vs RMSE bubble · predicted vs actual scatter · **residual analysis** (histogram + normal fit, residuals vs predicted, per-city MAE) · feature importance · GroupKFold CV error bars |
| Heatmap | Interactive scatter / density map · UHI range filter · top-10 hottest / coolest cities · seasonal city × month heatmap |
| Prediction | 12 input sliders · city quick-load preset · live predicted UHI with severity badge · gauge chart · radar driving factors · **scaled feature contribution bars** · sensitivity analysis · **3D response surface** (Temperature × Urban Fraction → UHI) · **city similarity finder** (feature-space nearest training city) · **mitigation insights** (evidence-based strategies) |

---

## Force Re-run Any Step

Delete the output file for that step, then re-run `python main.py`:

```bash
# Windows
del data\raw_data.csv              # re-collect (clears cache too for fresh GEE calls)
del data\processed_data.csv        # re-preprocess
del models\best_model.pkl          # re-train
del models\metrics.json            # re-train

# macOS / Linux
rm data/raw_data.csv
rm data/processed_data.csv
rm models/best_model.pkl models/metrics.json
```

To also force fresh API calls (not use cached responses):
```bash
# Windows — delete all cache files
del cache\cache_*.json
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit >= 1.32` | Dashboard web framework |
| `plotly >= 5.18` | Interactive charts (gauge, radar, choropleth, scatter) |
| `scikit-learn >= 1.3` | ML models, GridSearchCV, GroupKFold, StandardScaler |
| `xgboost >= 2.0` | XGBoost gradient boosting regressor |
| `lightgbm >= 4.0` | LightGBM gradient boosting regressor |
| `pandas >= 2.0` | DataFrames and CSV I/O |
| `numpy >= 1.24` | Numerical operations |
| `requests >= 2.31` | HTTP calls to Open-Meteo and OpenWeatherMap |
| `earthengine-api >= 0.1.390` | Google Earth Engine Python client (NDVI + LST) |
| `google-auth >= 2.0` | GEE OAuth authentication |

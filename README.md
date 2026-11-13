# Urban Heat Island Estimation System

A fully automated, end-to-end machine-learning pipeline for estimating Urban Heat Island intensity from real satellite and weather data across 25 global cities.

> **No synthetic or random data.** Every observation comes from Open-Meteo ERA5 reanalysis and MODIS satellite products (via Google Earth Engine, or the Microsoft Planetary Computer mirror). The UHI target is derived from real 8-day Land Surface Temperature composites. Rows without an observed composite are **excluded, never imputed**, and negative UHI (urban cool islands) is kept.

> **Evaluation (v2.0):** all model-quality numbers come from **nested leave-one-city-out cross-validation** with fold-local preprocessing, two baselines and city-level bootstrap confidence intervals. See [docs/EVALUATION_METHODOLOGY.md](docs/EVALUATION_METHODOLOGY.md) and [evaluation/metrics.json](evaluation/metrics.json). Headline: on unseen cities the tuned ML pipeline does **not** beat the simple mean baseline (see the results table in the methodology doc).

---

## Quick Start

```bash
# 1. Install dependencies (requirements-dev adds pytest + the Planetary Computer collector)
pip install -r requirements-dev.txt

# 2. Run the tests
python -m pytest -q

# 3. Reproduce the published evaluation from the committed data snapshot
python scripts/evaluate.py --seed 42

# 4. Launch the dashboard
streamlit run streamlit_app.py --server.port 8505
```

Full pipeline with fresh Google Earth Engine data (needs an Earth Engine project):

```bash
earthengine authenticate      # one-time, opens a browser
python main.py                # collect → preprocess → nested evaluation → dashboard
```

No Earth Engine account? Collect the same MODIS products from Microsoft Planetary Computer:

```bash
python data_collector_pc.py --out data/raw_data.csv
python scripts/evaluate.py --data data/raw_data.csv --seed 42
```

Dashboard opens at **http://localhost:8505**

---

## Streamlit Cloud Deployment

Use `streamlit_app.py` as the Streamlit Cloud entrypoint. It loads the dashboard
directly instead of running the local pipeline orchestrator.

For live Google Earth Engine data on Streamlit Cloud, create a Google service
account that has access to your Earth Engine project, then paste the service
account JSON into **Streamlit Cloud > App > Settings > Secrets**. Use
`.streamlit/secrets.example.toml` as the template:

```toml
GEE_PROJECT_ID = "your-google-cloud-project-id"

[gcp_service_account]
type = "service_account"
project_id = "your-google-cloud-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-google-cloud-project-id.iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```

Local development can still use `earthengine authenticate` / `gcloud`. The app
uses service-account credentials only when they are available; otherwise local
runs fall back to the normal Earth Engine authentication flow.

After deployment, press **Refresh live data** in the dashboard sidebar if the
cloud app has no generated `data/` or `models/` artifacts yet.

---

## Project Structure

```
HeatFluxAI/
├── streamlit_app.py        # Streamlit Cloud entrypoint
├── main.py                 # Local orchestrator — collect → preprocess → evaluate → dashboard
├── config.py               # Settings, city list, feature list, fixed collection window, paths
├── logger.py               # Logging utility
├── data_collector.py       # Open-Meteo weather + GEE NDVI + GEE 8-day LST (covering-composite match)
├── data_collector_pc.py    # Same MODIS products via Planetary Computer (no GEE account)
├── preprocessor.py         # STATELESS only: validity rules, feature formulas, signed observed target + provenance
├── evaluation.py           # Fold-local pipeline, nested leave-one-city-out CV, baselines, metrics, bootstrap CIs
├── eval_artifacts.py       # Validated loading of evaluation artifacts for the dashboard
├── model_trainer.py        # Wrapper: nested evaluation + separately selected deployment model
├── dashboard.py            # Streamlit 6-tab dashboard
├── scripts/evaluate.py     # Reproduce evaluation/metrics.json + heldout_predictions.csv
├── tests/                  # pytest suite (CV, preprocessing, target, metric and dashboard integrity)
├── docs/EVALUATION_METHODOLOGY.md
│
├── data/
│   ├── snapshot_modis_pc_2025H1.csv   # committed raw snapshot used for the published evaluation
│   ├── raw_data.csv                   # (generated, gitignored)
│   └── processed_data.csv             # (generated, gitignored) features + uhi_intensity + target_provenance
│
├── evaluation/                        # committed, reproducible
│   ├── metrics.json                   # per-city + aggregate MAE/RMSE, CIs, config, fingerprints
│   └── heldout_predictions.csv        # out-of-fold predictions & residuals for all 3 models
│
└── models/                            # (generated, gitignored)
    ├── best_model.pkl                 # deployment sklearn Pipeline (raw features in)
    ├── feature_names.json             # candidate input features
    └── outlier_stats.json             # descriptive IQR stats (capping itself is fold-local)
```

---

## Pipeline Overview

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `data_collector.py` / `data_collector_pc.py` | 180 days (fixed window 2025-01-01 → 2025-06-29) of ERA5 weather per city; annual NDVI; 8-day LST for 1 urban + 12 rural points; each day is matched to the composite whose 8-day period **covers** it |
| 2 | `preprocessor.py` | Stateless only: physical-range checks on present values, feature formulas, signed target `urban − rural LST`, `target_provenance`; no imputation, scaling, capping or clipping |
| 3 | `evaluation.py` via `scripts/evaluate.py` / `model_trainer.py` | Nested leave-one-city-out CV of the mean baseline, the latitude+NDVI baseline and the inner-CV-selected ML pipeline; city-bootstrap CIs; then a separately selected deployment model |
| 4 | `dashboard.py` | Streamlit app; the Models tab reads only validated evaluation artifacts |

---

## Data Sources

| Source | What is fetched | Cost |
|--------|----------------|------|
| **Open-Meteo Archive** (`archive-api.open-meteo.com`) | Daily mean temperature, humidity, max wind speed, surface pressure, cloud cover — 180 days per city (ERA5 reanalysis by ECMWF) | Free, no API key |
| **Planetary Computer — MODIS MOD11A2 / MOD13A1** (`data_collector_pc.py`) | Same 8-day LST product and 500 m NDVI, read anonymously from the STAC mirror; used for the committed snapshot | Free, no account |
| **GEE — MODIS MOD13A2** | Annual mean NDVI 2023 at city coordinates; retries at 500 m → 1 km → 2 km → 5 km for masked pixels | Free (GEE account) |
| **GEE — MODIS MOD11A2** | Full 8-day daytime LST timeseries for the 180-day window; sampled at 1 urban + 12 rural directional reference points per city in a single server-side batch call | Free (GEE account) |
| **OpenWeatherMap** (`api.openweathermap.org`) | Current weather — last-resort fallback only if Open-Meteo fails | Free tier |

---

## UHI Target Variable

```
UHI intensity (°C) = urban_LST(t) − rural_LST(t)        # signed, never clipped
```

- **`urban_LST(t)`**: MODIS MOD11A2 8-day daytime LST at the city-centre coordinate, from the composite whose period contains day `t`
- **`rural_LST(t)`**: median of the same composite at 12 compass offsets (0.5° and 1.0°, ~55 km and ~110 km)
- **Provenance**: rows with both values are `observed_modis_lst_difference` and are the only rows used for training and scoring. Rows without a covering composite keep `NaN` (`missing_lst_no_covering_composite`). They are **not** filled with a city or global median.
- **Negative values are valid.** They mean the city surface was cooler than its surroundings, which is typical for desert-ringed cities such as Cairo and Karachi.

---

## Features (26 total)

| Category | Features |
|----------|---------|
| Core weather | `temperature`, `humidity`, `wind_speed`, `pressure`, `clouds` |
| Satellite / land | `ndvi`, `urban_fraction` (= 1 − NDVI), `veg_class` (0/1/2 categorical bins) |
| Geographic | `lat`, `lon`, `lat_abs`, `lon_sin`, `lon_cos`, `distance_from_equator` |
| Temporal | `hour`, `month`, `is_daytime`, `is_night`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Derived / interaction | `temp_humidity_interaction`, `wind_cooling_effect`, `temp_anomaly`, `heat_retention`, `heat_index` |

**Fold-local preprocessing:** IQR winsorisation (`temperature`, `humidity`, `wind_speed`, `pressure`, `clouds`, `ndvi`), median imputation, constant-feature removal (`VarianceThreshold`), exact-duplicate removal and standardisation are steps of one scikit-learn `Pipeline`. They are re-fitted on the training rows of every CV fold. `models/outlier_stats.json` is descriptive only.

**Removed / redundant inputs:** `urban_fraction` is dropped by definition because it equals `clip(1 − NDVI)`. The fold-local selectors drop `hour`, `is_daytime`, `is_night`, `hour_sin` and `hour_cos`, which are constant for daily data at 12:00, and `distance_from_equator`, an exact copy of `lat_abs`. The retained features per fold are listed in `metrics.json`.

**Key design decisions:**
- `veg_class` bins NDVI into sparse / moderate / dense vegetation — reduces direct numeric NDVI-to-LST leakage while keeping the categorical signal
- `temp_anomaly = temperature − city_mean_temperature` captures "unusually hot day for this city" independently of absolute temperature
- `heat_retention = urban_fraction × temperature / (wind_speed + 1)` combines built-up land, ambient heat, and wind damping into one physically meaningful feature
- `is_night` flag added because UHI dynamics differ between day and night
- `urban_heat_proxy` (old feature = `urban_fraction × (1 − NDVI)`) permanently removed — it directly encoded the target variable (data leakage)

---

## Models and Evaluation

Candidate grid (searched jointly by inner `GridSearchCV`): Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost and LightGBM, each with a small hyper-parameter grid (see `evaluation.candidate_param_grid` and `metrics.json → models.nested_selected_model.search_space`).

**Nested leave-one-city-out evaluation:**

1. Outer `LeaveOneGroupOut` over cities. Each city is held out once and scored once.
2. Inner `GroupKFold` (≤ 5 folds) over the remaining cities selects algorithm and hyper-parameters by mean inner RMSE. The held-out city is never used for any choice.
3. The selected pipeline (including its preprocessing) is refit on all remaining cities and predicts the held-out city.
4. The mean baseline (`DummyRegressor`) and the latitude + NDVI linear baseline run on the same folds.
5. Per-city MAE/RMSE, macro (city-averaged) and pooled (observation-level) aggregates, and 95 % city-bootstrap CIs (2 000 resamples) are written to `evaluation/metrics.json`.

**Results:** see the table in [docs/EVALUATION_METHODOLOGY.md](docs/EVALUATION_METHODOLOGY.md#10-results).

The dashboard's Prediction tab uses a **deployment model**, selected separately by grouped CV on all cities. Its selection-CV score is not a test estimate.

---

## Dashboard Sections

| Tab | Contents |
|-----|---------|
| Overview | Animated live-stats ticker · KPI cards · pipeline status · UHI histogram + violin · UHI severity reference table · global scatter map |
| Data Explorer | Raw + processed data tables · source breakdown pie · feature box plots · time series per city · seasonal monthly chart · Pearson correlation heatmap · CSV downloads |
| Preprocessing | Missing values · target provenance (observed / excluded / negative counts) · descriptive IQR statistics (capping is fold-local) · engineered feature views |
| Models | Nested leave-one-city-out comparison of mean baseline, latitude+NDVI baseline and selected ML pipeline (macro & pooled MAE/RMSE, city-bootstrap CIs, paired difference) · per-city table and chart · algorithm chosen per fold · **Held-Out City Residuals (Nested Cross-Validation)** by model and city · deployment model card + importances |
| Heatmap | Interactive scatter / density map · UHI range filter · top-10 hottest / coolest cities · seasonal city × month heatmap |
| Prediction | 12 input sliders · city quick-load preset · live predicted UHI with severity badge · gauge chart · radar driving factors · **standardised feature input bars (display only)** · sensitivity analysis · **3D response surface** (Temperature × Urban Fraction → UHI) · **city similarity finder** (feature-space nearest training city) · **mitigation insights** (evidence-based strategies) |

---

## Re-running

```bash
python scripts/evaluate.py --seed 42                 # evaluation from the committed snapshot
python scripts/evaluate.py --fast --bootstrap 500    # quick check with a reduced grid
python -m pytest -q                                  # test suite
```

To re-collect data with `python main.py`, delete `data/raw_data.csv` (and `cache/cache_*.json` for fresh API calls).

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit >= 1.32` | Dashboard web framework |
| `plotly >= 5.18` | Interactive charts (gauge, radar, choropleth, scatter) |
| `scikit-learn >= 1.3` | Pipelines, GridSearchCV, LeaveOneGroupOut, GroupKFold, DummyRegressor |
| `xgboost >= 2.0` | XGBoost gradient boosting regressor |
| `lightgbm >= 4.0` | LightGBM gradient boosting regressor |
| `pandas >= 2.0` | DataFrames and CSV I/O |
| `numpy >= 1.24` | Numerical operations |
| `requests >= 2.31` | HTTP calls to Open-Meteo and OpenWeatherMap |
| `earthengine-api >= 0.1.390` | Google Earth Engine Python client (NDVI + LST) |
| `google-auth >= 2.0` | GEE OAuth authentication |
| `pytest` (dev) | Test suite |
| `pystac-client`, `planetary-computer`, `rasterio` (dev) | Planetary Computer MODIS collector |

# HeatFluxAI — Evaluation Methodology (v2.0)

This document explains how HeatFluxAI's model-quality numbers are produced, why
the previous procedure overstated performance, and how to reproduce every
published number. The machine-readable result is
[`evaluation/metrics.json`](../evaluation/metrics.json). Per-observation held-out
predictions are in
[`evaluation/heldout_predictions.csv`](../evaluation/heldout_predictions.csv).

---

## 1. What was wrong before (audit summary)

| # | Problem in the previous pipeline | Where | Effect |
|---|---|---|---|
| 1 | The "best model" was chosen by RMSE on the test cities, and that same test RMSE was reported as its performance | `model_trainer.py` (`if rmse < best_rmse`) | Optimistic, selection-biased estimate |
| 2 | Median imputation of every numeric column on the full dataset before splitting | `preprocessor.py` step 3 | Test-city values influenced training inputs |
| 3 | IQR winsorisation fences computed on the full dataset | `preprocessor.cap_outliers` | Same leakage, through the fences |
| 4 | `StandardScaler` fitted on the full dataset and reused inside CV | `preprocessor.py` step 8, `model_trainer.py` | Same leakage, through the scaler |
| 5 | "CV RMSE" ran `cross_val_score` on models whose hyper-parameters were already tuned on most of the same cities | `model_trainer.py` | CV score not independent of tuning |
| 6 | One `GroupShuffleSplit` (≈5 test cities) | `model_trainer.py` | High-variance estimate from a single arbitrary split |
| 7 | Missing UHI targets (no LST composite) were filled with the city median, or else the global median, and then scored like real observations | `preprocessor.compute_uhi_intensity` | Evaluation partly against invented targets |
| 8 | UHI was clipped at 0 (`.clip(lower=0)`, twice) | `preprocessor.compute_uhi_intensity` | Real negative UHI (urban cool islands, e.g. Cairo, Karachi) was erased |
| 9 | Dashboard "Test residuals" were computed from the final model on **all** rows (≈80 % training rows), with imputed targets and predictions clipped at 0 | `dashboard.py`, Models tab | In-sample residuals shown as test residuals |
| 10 | Each daily weather row took the LST composite with the *nearest start date* (±8 days), which could be a period that does not cover that day | `data_collector.match_to_nearest_lst` | Target attached to the wrong period |
| 11 | Constant features (`hour`=12, `is_daytime`, `is_night`, `hour_sin/cos` — daily data at 12:00) and an exact duplicate (`distance_from_equator` = `lat_abs`) were fed to the models; `urban_fraction` is defined as `1 − NDVI` | `preprocessor.py`, `data_collector.py` | Redundant / non-informative inputs |
| 12 | The collection window rolled with the current date; missing pressure/clouds were hard-filled with 1013 hPa / 50 % at collection | `data_collector.py` | Non-reproducible data; silent feature imputation |

All of these are fixed. Items 1–9 are covered by automated tests (`tests/`).

---

## 2. Why test RMSE must not select the model

If you train several candidates, look at their test errors and keep the smallest,
the kept error is the *minimum* of several noisy estimates. It is biased
downward: the test set has been used for fitting, at the level of model choice.
With only a handful of test cities this bias is large. HeatFluxAI now chooses the
algorithm and its hyper-parameters using **only validation folds built from
training cities**. Held-out cities are scored once, after selection and refit,
and those scores never feed back into any choice.

## 3. Why all preprocessing is fitted inside the folds

Anything that *learns from data*, such as medians for imputation, IQR fences,
scaler means and standard deviations, or which columns are constant or
duplicated, carries information about the rows it was learned from. Fitting it
on all rows lets held-out cities shape the training inputs. Every such step is
now part of one scikit-learn `Pipeline`:

```
Winsorizer(IQR k=1.5 on temperature, humidity, wind_speed, pressure, clouds, ndvi)
 → SimpleImputer(median)
 → VarianceThreshold(1e-12)            # constant-feature removal
 → DuplicateFeatureDropper(exact)      # identical-column removal
 → StandardScaler
 → model
```

`GridSearchCV` clones and refits this whole pipeline on the training part of
every inner fold, and again on all outer-training cities. Held-out rows are only
`transform`ed. `preprocessor.py` now does only stateless, row-wise work: physical
range checks on present values, feature formulas and the target definition.
Tests check that imputer statistics, scaler means and winsorisation fences equal
those from a training-only fit, and differ from a fit on all rows.

`temp_anomaly` (temperature minus the city's own mean over the window) is the one
within-city statistic. It uses only that city's own weather, never targets or
other cities. It is still transductive for the held-out city, and this is listed
as a limitation.

## 4. Grouped, nested, leave-one-city-out evaluation

The question the dashboard answers is *"how wrong is the model for a city it has
never seen?"*. Random row splits put rows of the same city, often from the same
8-day LST composite, on both sides, which measures memorisation. HeatFluxAI uses:

```
eligible rows (observed targets only)
└─ OUTER: LeaveOneGroupOut over city_name  (one fold per city)
   ├─ held-out city ─────────────────────────────► scored once
   └─ remaining cities
      └─ INNER: GroupKFold(n_splits = min(5, #training cities))
         └─ GridSearchCV over {algorithm × hyper-parameters}
            selection metric: mean inner-fold RMSE (lowest wins, ties → grid order)
         → refit best pipeline on all remaining cities → predict held-out city
```

* **Groups are real cities.** `city_name` comes from the configured city list.
  `validate_groups` refuses to run if any city has more than one coordinate pair,
  if coordinates are shared between names, or if there are more groups than half
  the rows (a row-level ID).
* **No silent fallback.** With fewer than 4 cities that have observed targets,
  evaluation raises `InsufficientGroupsError` instead of switching to row splits.
* **Inner vs outer.** Inner folds choose. The outer fold judges. A city is never
  split between inner train and validation, and the outer city never enters the
  inner loop. One test feeds absurd target values to a held-out city and checks
  that the selected parameters and predictions for that fold do not change.
* **Deployment model.** The Prediction tab needs one fitted model. It is chosen
  with the same grid by `GroupKFold` CV on all cities and refit on all of them.
  Its selection-CV RMSE is reported separately and labelled as *not* a test
  estimate. The unbiased estimate of this procedure is the nested result.

## 5. Baselines

Both baselines run on exactly the same outer folds:

* **Mean baseline**: `DummyRegressor(strategy="mean")`. It predicts the mean UHI
  of the outer-training cities. It never sees held-out targets, and a test
  checks this.
* **Latitude + NDVI baseline**: `LinearRegression` on `[lat, ndvi]` (signed
  latitude, annual-mean NDVI), with fold-local median imputation. Both inputs are
  static and available at prediction time. If either column were missing, the
  baseline would fail rather than silently change its definition.

## 6. Only real targets are scored, and negative UHI is valid

`uhi_intensity = urban_LST − median(rural ring LST)` in °C, from the MODIS
MOD11A2 8-day daytime LST composite whose period **covers** the observation date.
Each row gets a `target_provenance`:

* `observed_modis_lst_difference`: both LST values exist. This is the only
  eligible provenance for training, tuning and scoring.
* `missing_lst_no_covering_composite`: no composite or masked pixels. The target
  stays `NaN`. The row is kept for maps and inference and is never imputed.

If provenance is absent, `eligible_rows` raises an error rather than assume all
non-null targets are observed. No `clip` or `abs` is applied to the target.
**Negative UHI is kept**: it means the urban surface was cooler than its rural
ring, which is common where the surroundings are desert or bare soil (Cairo,
Karachi and Delhi in this dataset). Predictions may be negative too. Maps use a
diverging colour scale centred on 0 °C, with marker *size* set by |UHI|.

## 7. Metrics and confidence intervals

For every model and held-out city: n, MAE, RMSE, and mean residual
(`residual = observed − predicted`). Aggregates:

* **Macro** = unweighted mean of per-city MAE or RMSE. Each city counts once.
* **Pooled** = MAE or RMSE over all held-out observations, so large cities weigh
  more. The mean of city RMSEs is *never* called pooled RMSE.

**95 % CIs come from a city-level percentile bootstrap.** It draws 2 000
resamples with seed 42. Each resample draws cities with replacement, and each
drawn city keeps all of its held-out observations. Macro and pooled metrics are
recomputed per resample. The same draws are used for every model, which yields
paired-difference intervals (selected − baseline). Thousands of daily rows from
24 cities are *not* treated as thousands of independent test units. With fewer
than 8 cities, intervals are not reported. Below 20 cities they are flagged as
unstable. These intervals are descriptive, not significance tests.

## 8. Where the dashboard's residuals come from

The Models tab reads `evaluation/metrics.json` and
`evaluation/heldout_predictions.csv` through `eval_artifacts.load_evaluation`. That
function refuses to display anything if:

* the files are missing;
* `evaluation.method` is not `nested_leave_one_city_out`;
* the prediction file's SHA-256 does not match `metrics.json`;
* any scored target is not observed; or
* any aggregate in `metrics.json` cannot be recomputed from the predictions.

The residual panel is titled **"Held-Out City Residuals (Nested Cross-Validation)"**.
It shows the evaluated model, the held-out city (or all of them), the number of
valid observations, and the MAE/RMSE of exactly those observations. A note says
that observations within a city are not independent. The dashboard never computes
residuals from the deployment model on its own training data.

## 9. Reproducing everything

```bash
pip install -r requirements-dev.txt
python -m pytest -q                         # 41 tests, ~1 min
python scripts/evaluate.py --seed 42        # ~3 min on 8 cores; rewrites evaluation/*
streamlit run streamlit_app.py              # or: python main.py
```

`scripts/evaluate.py` reads the committed raw snapshot
`data/snapshot_modis_pc_2025H1.csv`. Its SHA-256 is recorded in `metrics.json`
next to a fingerprint of the eligible modelling table and a fingerprint of
`evaluation.py`, `preprocessor.py` and `config.py`. The script checks that every
aggregate can be recomputed from the saved predictions to within 1e-5. With the
same data, code and package versions, a re-run reproduces `metrics.json`. Expect
differences only in the last decimals on other platforms or library versions
(tree and boosting implementations change between releases).

To rebuild the snapshot itself (needs network access, no account):

```bash
python data_collector_pc.py --out data/snapshot_modis_pc_2025H1.csv
```

The Google Earth Engine collector (`data_collector.py`, used by `python main.py`
and the dashboard's live refresh) produces a different dataset (MOD13A2 1 km
NDVI). Re-run the evaluation before quoting numbers from it.

## 10. Results

Generated by `python scripts/evaluate.py --seed 42` on the committed snapshot (raw SHA-256 `36fec997cb3d…`). There are **24 cities** and **3,624 observed targets** (461 distinct city × 8-day composites). **876 rows were excluded** because they had no observed LST, and **856 negative targets** (23.6 %) were kept. All errors are in °C on held-out cities.

| Model | Macro MAE | Macro RMSE (95 % CI) | Pooled MAE | Pooled RMSE (95 % CI) | Evaluation |
|---|---|---|---|---|---|
| Mean baseline | 2.367 | 2.843 [2.44, 3.34] | 2.324 | 3.039 [2.50, 3.61] | Held-out cities, nested LOCO |
| Latitude + NDVI | 2.513 | 2.991 [2.58, 3.49] | 2.451 | 3.164 [2.63, 3.72] | Held-out cities, nested LOCO |
| Selected ML pipeline (nested) | 2.651 | 3.130 [2.73, 3.59] | 2.620 | 3.308 [2.81, 3.81] | Held-out cities, nested LOCO |

* Paired city-bootstrap 95 % CI of the macro-RMSE difference: selected − mean baseline **[-0.02, +0.65]**; selected − latitude+NDVI **[-0.16, +0.49]**. **The tuned ML pipeline shows no improvement over predicting the training-city mean UHI for an unseen city.** Its point estimates are worse, and the intervals are compatible with no difference or with a small degradation.
* Inner CV chose: Lasso ×22, XGBRegressor ×2 (out of 24 outer folds).
* Deployment model (Prediction tab): Lasso {'alpha': 0.1}, selection-CV RMSE 3.020 (a selection score, not a test estimate).
* The old pipeline's single-split "best RMSE" is not comparable. It was selected on the test cities, scored partly against imputed targets, and computed on a target clipped at 0.

Per-city results (selected pipeline vs mean baseline, RMSE):

| City | n | Selected | ML RMSE | Mean-baseline RMSE | Lat+NDVI RMSE |
|---|---|---|---|---|---|
| Bangkok | 120 | Lasso | 3.050 | 2.030 | 2.134 |
| Beijing | 176 | Lasso | 2.182 | 2.328 | 2.360 |
| Berlin | 172 | Lasso | 2.325 | 2.478 | 2.455 |
| Cairo | 180 | Lasso | 3.942 | 4.689 | 4.668 |
| Chicago | 172 | Lasso | 2.868 | 2.908 | 2.892 |
| Delhi | 160 | Lasso | 2.660 | 3.266 | 3.611 |
| Dhaka | 136 | Lasso | 2.075 | 1.341 | 1.201 |
| Istanbul | 132 | Lasso | 1.919 | 1.847 | 1.822 |
| Jakarta | 20 | XGBRegressor | 4.141 | 4.960 | 5.127 |
| Karachi | 180 | Lasso | 5.216 | 5.605 | 5.816 |
| Kolkata | 152 | Lasso | 3.268 | 2.370 | 2.354 |
| Lagos | 64 | Lasso | 3.670 | 2.906 | 2.998 |
| London | 180 | Lasso | 5.054 | 2.932 | 3.073 |
| Los Angeles | 180 | Lasso | 3.236 | 2.525 | 2.509 |
| Mexico City | 172 | Lasso | 1.775 | 1.988 | 2.241 |
| Mumbai | 144 | Lasso | 2.605 | 2.224 | 2.217 |
| Paris | 164 | XGBRegressor | 5.597 | 3.199 | 3.516 |
| Seoul | 176 | Lasso | 2.736 | 2.272 | 2.268 |
| Shanghai | 172 | Lasso | 2.332 | 1.864 | 1.887 |
| Singapore | 108 | Lasso | 2.719 | 1.552 | 3.159 |
| Sydney | 172 | Lasso | 1.900 | 2.142 | 2.335 |
| São Paulo | 164 | Lasso | 2.091 | 2.085 | 2.131 |
| Tokyo | 172 | Lasso | 4.392 | 5.193 | 5.176 |
| Toronto | 156 | Lasso | 3.377 | 3.525 | 3.834 |

## 11. Limitations

* **Few, hand-picked cities.** There are 24 evaluable cities, all megacities,
  chosen by hand. Bootstrap intervals assume they are exchangeable draws from a
  population of cities, which they are not. Do not generalise to small towns or
  unrepresented climates.
* **One season, one year.** The window is 2025-01-01 to 2025-06-29: northern
  winter to early summer, southern summer to early winter. Other seasons or years
  may behave differently.
* **Surface, daytime, ~1 km.** The target is a MODIS daytime land-surface
  temperature difference, not an air-temperature UHI and not night-time UHI. The
  rural ring (55–110 km) can include water, mountains or other urban areas.
  Clear-sky sampling biases toward cloud-free days.
* **Autocorrelation.** Each 8-day composite is shared by up to 8 daily rows, so
  the effective sample size is the number of (city, composite) pairs (461 here), not
  3 624.
* **City-constant inputs.** lat, lon, NDVI and veg class are constant within a
  city, so with 23 training cities the models can at best interpolate between
  cities.
* **Residual dependence.** Leave-one-city-out does not remove spatial or
  climatic similarity between cities (Delhi/Karachi, Kolkata/Dhaka), or the
  shared time window.
* **Sources.** The published snapshot uses the Planetary Computer MODIS mirror
  (MOD11A2 LST, the same product GEE serves, and MOD13A1 500 m NDVI instead of
  MOD13A2 1 km) plus Open-Meteo ERA5 weather. New York has no valid urban LST
  pixel at its configured coordinate (water-edge fill), so it has no observed
  targets and is excluded from scoring.
* **Software versions.** The committed artifacts were generated with the package
  versions listed under `software` in `metrics.json`. Streamlit Cloud pins older
  versions (`requirements.txt`), and a cloud re-run may differ in late decimals.

"""
UHI Prediction System - Data Collection
Sources:
  - Open-Meteo archive (ERA5, free)  → 180 days of daily weather per city
  - GEE MODIS MOD13A2                → annual mean NDVI 2023
  - GEE MODIS MOD11A2                → 8-day LST time series (dynamic target)
No synthetic or random data.  Cities without NDVI are skipped.
"""

import json
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta

import ee

from config import (
    CITIES, OPENWEATHER_API_KEY, OPENWEATHER_BASE_URL,
    RAW_DATA_PATH, CACHE_DIR, GEE_PROJECT_ID
)
from logger import get_logger

log = get_logger("data_collector")

# ─── INIT GEE ─────────────────────────────────────────────────────────────────
try:
    ee.Initialize(project=GEE_PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_ID)


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(label: str) -> Path:
    h = hashlib.md5(label.encode()).hexdigest()[:10]
    return CACHE_DIR / f"cache_{h}.json"


def _load_cache(label: str, ttl: int = 3600):
    path = _cache_key(label)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if time.time() - data.get("_ts", 0) < ttl:
                return data
        except Exception:
            pass
    return None


def _save_cache(label: str, data: dict):
    path = _cache_key(label)
    data["_ts"] = time.time()
    try:
        path.write_text(json.dumps(data))
    except Exception:
        pass


# ─── Directional offsets for rural reference (0.5° ≈ 55 km, 1.0° ≈ 110 km) ──
_RURAL_OFFSETS = [
    ( 0.5,  0.0), (-0.5,  0.0), ( 0.0,  0.5), ( 0.0, -0.5),
    ( 0.5,  0.5), (-0.5,  0.5), (-0.5, -0.5), ( 0.5, -0.5),
    ( 1.0,  0.0), (-1.0,  0.0), ( 0.0,  1.0), ( 0.0, -1.0),
]


# ─── GEE: NDVI (MODIS MOD13A2) ────────────────────────────────────────────────

def fetch_ndvi(lat: float, lon: float) -> float | None:
    """
    Annual mean NDVI for 2023. Retries at coarser scales so dense urban
    or water-edge pixels (e.g. New York) do not get skipped.
    """
    label = f"ndvi_{lat:.4f}_{lon:.4f}"
    cached = _load_cache(label, ttl=86400)
    if cached and "ndvi" in cached:
        return round(float(cached["ndvi"]), 4)

    try:
        point = ee.Geometry.Point(lon, lat)
        dataset = (
            ee.ImageCollection("MODIS/061/MOD13A2")
            .filterDate("2023-01-01", "2023-12-31")
            .select("NDVI")
            .mean()
        )
        ndvi = None
        for scale in [500, 1000, 2000, 5000]:
            fc = dataset.sample(region=point, scale=scale)
            if fc.size().getInfo() > 0:
                raw = fc.first().get("NDVI").getInfo()
                if raw is not None:
                    ndvi = round(float(raw) / 10000.0, 4)
                    break
        if ndvi is None:
            raise ValueError("All scales returned null NDVI")
        _save_cache(label, {"ndvi": ndvi})
        return ndvi
    except Exception as e:
        log.warning(f"  NDVI fetch failed ({lat},{lon}): {e}")
        return None


# ─── GEE: 8-day LST time series (MODIS MOD11A2) ───────────────────────────────

def fetch_lst_timeseries(
    lat: float, lon: float, start_date: str, end_date: str
) -> dict:
    """
    Fetch all 8-day daytime LST composites in [start_date, end_date] for
    one urban point and 12 rural reference directions.  Uses a single
    server-side GEE batch call (sampleRegions over the whole collection)
    to minimise round-trips.

    Returns:
        {date_str: {"urban_lst": float, "rural_lst": float}, ...}
        rural_lst = MEDIAN of valid surrounding land pixels  (Fix #5)
    """
    label = f"lst_ts_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}"
    cached = _load_cache(label, ttl=86400 * 30)
    if cached and "ts" in cached:
        return cached["ts"]

    try:
        lst_col = (
            ee.ImageCollection("MODIS/061/MOD11A2")
            .select("LST_Day_1km")
            .filterDate(start_date, end_date)
        )

        # Build FeatureCollection: 1 urban + 12 rural points
        feats = [ee.Feature(ee.Geometry.Point(lon, lat), {"pt": "urban"})]
        for i, (dlat, dlon) in enumerate(_RURAL_OFFSETS):
            feats.append(ee.Feature(
                ee.Geometry.Point(lon + dlon, lat + dlat),
                {"pt": f"rural_{i}"},
            ))
        all_pts = ee.FeatureCollection(feats)

        # Server-side: for every image, sample all points at once
        def _sample_img(img):
            date_str = img.date().format("YYYY-MM-dd")
            return img.sampleRegions(
                collection=all_pts,
                scale=1000,
                geometries=False,
            ).map(lambda f: f.set("date", date_str))

        raw_features = (
            lst_col.map(_sample_img).flatten().getInfo().get("features", [])
        )

        # Parse results into {date: {urban, rural_list}}
        by_date = defaultdict(lambda: {"urban": None, "rural": []})
        for feat in raw_features:
            props = feat.get("properties", {})
            date  = props.get("date")
            raw   = props.get("LST_Day_1km")
            if not date or raw is None:
                continue
            lst_c = float(raw) * 0.02 - 273.15
            if props["pt"] == "urban":
                by_date[date]["urban"] = lst_c
            else:
                by_date[date]["rural"].append(lst_c)

        # Build final timeseries dict
        ts = {}
        for date, vals in by_date.items():
            if vals["urban"] is not None and vals["rural"]:
                ts[date] = {
                    "urban_lst": round(vals["urban"], 3),
                    # median of valid surrounding pixels — more stable than min (Fix #5)
                    "rural_lst": round(float(np.median(vals["rural"])), 3),
                }

        _save_cache(label, {"ts": ts})
        log.info(f"    LST timeseries: {len(ts)} composites")
        return ts

    except Exception as e:
        log.warning(f"  LST timeseries failed ({lat},{lon}): {e}")
        return {}


# ─── Match weather date → nearest LST composite ───────────────────────────────

def match_to_nearest_lst(
    date_str: str, lst_ts: dict
) -> tuple[float | None, float | None]:
    """
    Find the nearest available 8-day LST composite for a daily weather date.
    Returns (None, None) if the gap exceeds one composite period (8 days).
    """
    if not lst_ts:
        return None, None
    try:
        weather_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        best_key = min(
            lst_ts.keys(),
            key=lambda d: abs((weather_dt - datetime.strptime(d, "%Y-%m-%d")).days),
        )
        gap = abs((weather_dt - datetime.strptime(best_key, "%Y-%m-%d")).days)
        if gap > 8:
            return None, None
        return lst_ts[best_key]["urban_lst"], lst_ts[best_key]["rural_lst"]
    except Exception:
        return None, None


# ─── Open-Meteo: free historical weather (ERA5 reanalysis) ────────────────────

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_openmeteo_history(city: dict, start_date: str, end_date: str) -> list[dict]:
    """Daily mean weather from Open-Meteo archive (free, no API key)."""
    label = f"om_{city['name']}_{start_date}_{end_date}"
    cached = _load_cache(label, ttl=86400 * 30)
    if cached and "rows" in cached:
        return cached["rows"]

    try:
        r = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude":  city["lat"],
                "longitude": city["lon"],
                "start_date": start_date,
                "end_date":   end_date,
                "daily": (
                    "temperature_2m_mean,relative_humidity_2m_mean,"
                    "wind_speed_10m_max,surface_pressure_mean,cloud_cover_mean"
                ),
                "timezone": "UTC",
            },
            timeout=20,
        )
        r.raise_for_status()
        daily = r.json().get("daily", {})

        rows = []
        for i, date_str in enumerate(daily.get("time", [])):
            t = daily["temperature_2m_mean"][i]
            h = daily["relative_humidity_2m_mean"][i]
            w = daily["wind_speed_10m_max"][i]
            p = daily.get("surface_pressure_mean", [None] * (i + 1))[i]
            c = daily.get("cloud_cover_mean",       [None] * (i + 1))[i]
            if any(v is None for v in [t, h, w]):
                continue
            rows.append({
                "name":        city["name"],
                "lat":         city["lat"],
                "lon":         city["lon"],
                "temperature": round(float(t), 2),
                "humidity":    round(float(h), 1),
                "wind_speed":  round(float(w) / 3.6, 2),  # km/h → m/s
                "pressure":    round(float(p), 1) if p else 1013.0,
                "clouds":      int(round(float(c))) if c else 50,
                "timestamp":   f"{date_str}T12:00:00",
            })

        _save_cache(label, {"rows": rows})
        log.info(f"  Open-Meteo {city['name']:15s} {len(rows)} days")
        return rows

    except Exception as e:
        log.warning(f"  Open-Meteo failed for {city['name']}: {e}")
        return []


# ─── OpenWeatherMap: current weather (fallback) ───────────────────────────────

def fetch_openweather_current(city: dict) -> dict | None:
    cached = _load_cache(f"ow_{city['name']}", ttl=3600)
    if cached and "temperature" in cached:
        return cached
    try:
        r = requests.get(
            f"{OPENWEATHER_BASE_URL}/weather",
            params={
                "lat": city["lat"], "lon": city["lon"],
                "appid": OPENWEATHER_API_KEY, "units": "metric",
            },
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        result = {
            "name":        city["name"],
            "lat":         city["lat"],
            "lon":         city["lon"],
            "temperature": d["main"]["temp"],
            "humidity":    d["main"]["humidity"],
            "wind_speed":  d["wind"]["speed"],
            "pressure":    d["main"].get("pressure", 1013),
            "clouds":      d.get("clouds", {}).get("all", 50),
            "timestamp":   datetime.utcfromtimestamp(d["dt"]).isoformat(),
        }
        _save_cache(f"ow_{city['name']}", result)
        return result
    except Exception as e:
        log.warning(f"  OWM fallback failed for {city['name']}: {e}")
        return None


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def collect_data(force: bool = False) -> pd.DataFrame:
    if RAW_DATA_PATH.exists() and not force:
        log.info(f"Raw data exists. Loading ...")
        return pd.read_csv(RAW_DATA_PATH)

    log.info("=" * 60)
    log.info("STEP 1 – DATA COLLECTION")
    log.info("=" * 60)

    # 180-day window, ending 30 days ago so MODIS is fully available (Fix #7)
    end_dt   = datetime.utcnow() - timedelta(days=30)
    start_dt = end_dt - timedelta(days=179)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")
    log.info(f"  Window: {start_str} to {end_str}  ({180} days)")

    rows = []

    for city in CITIES:

        # ── 1. NDVI — skip city if all scales fail ────────────────────────
        ndvi = fetch_ndvi(city["lat"], city["lon"])
        if ndvi is None:
            log.info(f"  Skipping {city['name']} (NDVI unavailable)")
            continue

        # ── 2. Dynamic 8-day LST timeseries (Fix #1) ──────────────────────
        lst_ts = fetch_lst_timeseries(city["lat"], city["lon"], start_str, end_str)

        # ── 3. urban_fraction proxy = 1 − NDVI (no random) ───────────────
        urban_fraction = round(float(np.clip(1.0 - ndvi, 0.0, 1.0)), 3)

        # ── 4. Historical weather via Open-Meteo ──────────────────────────
        weather_rows = fetch_openmeteo_history(city, start_str, end_str)
        if not weather_rows:
            rec = fetch_openweather_current(city)
            if rec:
                weather_rows = [rec]
            time.sleep(0.3)

        if not weather_rows:
            log.info(f"  No weather data for {city['name']} — skipping")
            continue

        # ── 5. Match each daily row to nearest 8-day LST composite ───────
        city_rows = 0
        for rec in weather_rows:
            urban_lst, rural_lst = match_to_nearest_lst(rec["timestamp"][:10], lst_ts)
            rec["ndvi"]           = ndvi
            rec["urban_fraction"] = urban_fraction
            rec["urban_lst"]      = urban_lst     # None if no composite nearby
            rec["rural_lst"]      = rural_lst
            rec["source"]         = "openmeteo+gee"
            rows.append(rec)
            city_rows += 1

        lst_cover = sum(1 for r in rows[-city_rows:] if r["urban_lst"] is not None)
        log.info(
            f"  {city['name']:15s} {city_rows} rows  "
            f"LST coverage {lst_cover}/{city_rows}"
        )

    if not rows:
        raise RuntimeError("No data collected — check GEE access and network.")

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DATA_PATH, index=False)
    log.info(f"  Saved {len(df)} rows to {RAW_DATA_PATH}")
    return df


if __name__ == "__main__":
    collect_data(force=True)

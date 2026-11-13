"""
UHI Prediction System - Data Collection via Microsoft Planetary Computer

Alternative to ``data_collector.py`` for machines without Google Earth Engine
credentials.  It reads the *same MODIS Collection 6.1 products* from the
Planetary Computer STAC mirror (anonymous access, no account) and writes the
same raw schema, so the rest of the pipeline is unchanged.

  - Open-Meteo archive (ERA5)    → daily weather (shared with data_collector)
  - MODIS MOD13A1 v061 (500 m)   → annual mean NDVI for config.NDVI_YEAR
        (the GEE collector uses MOD13A2 1 km sampled at 500 m scale; MOD13A2 is
        not on Planetary Computer, MOD13A1 is the same product at 500 m)
  - MODIS MOD11A2 v061 (Terra)   → 8-day daytime LST, urban point + 12 rural
        reference points (same offsets as the GEE collector), rural = median

Only Terra (MOD*) granules are used, matching GEE's MODIS/061/MOD11A2.  Fill
values are dropped, never replaced.  No synthetic data.

Run:  python data_collector_pc.py [--out data/raw_data.csv]
"""

from __future__ import annotations

import argparse
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import CITIES, COLLECTION_START, COLLECTION_END, NDVI_YEAR, RAW_DATA_PATH
from data_collector import (
    _RURAL_OFFSETS,
    fetch_openmeteo_history,
    match_to_covering_lst,
)
from logger import get_logger

log = get_logger("data_collector_pc")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LST_COLLECTION, LST_ASSET, LST_PREFIX = "modis-11A2-061", "LST_Day_1km", "MOD11A2."
NDVI_COLLECTION, NDVI_ASSET, NDVI_PREFIX = "modis-13A1-061", "500m_16_days_NDVI", "MOD13A1."
LST_SCALE, LST_FILL = 0.02, 0
NDVI_SCALE, NDVI_FILL = 0.0001, -3000

SOURCE_TAG = "openmeteo+planetary_computer_modis"

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

_catalog_lock = threading.Lock()
_catalog = None


def _catalog_client():
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            import planetary_computer
            import pystac_client

            _catalog = pystac_client.Client.open(
                STAC_URL, modifier=planetary_computer.sign_inplace
            )
    return _catalog


def _search(collection: str, prefix: str, bbox, start: str, end: str) -> list:
    """Items of one collection whose *start* date lies in [start, end]
    (same semantics as ee.ImageCollection.filterDate on image start time)."""
    items = _catalog_client().search(
        collections=[collection], bbox=bbox, datetime=f"{start}/{end}"
    ).item_collection()
    out = []
    lo, hi = datetime.fromisoformat(start), datetime.fromisoformat(end)
    for it in items:
        if not it.id.startswith(prefix):
            continue
        sd = datetime.fromisoformat(it.properties["start_datetime"][:10])
        if lo <= sd <= hi:
            out.append(it)
    return out


def _read_points(href: str, points: list[tuple[float, float]], fill: int) -> list:
    """Raw pixel values at (lon, lat) points; None for fill / outside the tile."""
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(href) as ds:
        xs, ys = transform("EPSG:4326", ds.crs, [p[0] for p in points], [p[1] for p in points])
        vals = []
        for x, y in zip(xs, ys):
            r, c = ds.index(x, y)
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                vals.append(None)
                continue
            v = ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0]
            vals.append(None if int(v) == fill else int(v))
        return vals


def _points_for(city: dict) -> list[tuple[float, float]]:
    pts = [(city["lon"], city["lat"])]
    pts += [(city["lon"] + dlon, city["lat"] + dlat) for dlat, dlon in _RURAL_OFFSETS]
    return pts


def _bbox(points, pad=0.01):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def fetch_lst_timeseries_pc(city: dict, start: str, end: str) -> dict:
    """{composite_start_date: {urban_lst, rural_lst, n_rural}} in °C."""
    pts = _points_for(city)
    items = _search(LST_COLLECTION, LST_PREFIX, _bbox(pts), start, end)
    by_date: dict = defaultdict(lambda: {"urban": None, "rural": {}})
    for it in items:
        date = it.properties["start_datetime"][:10]
        vals = _read_points(it.assets[LST_ASSET].href, pts, LST_FILL)
        for i, v in enumerate(vals):
            if v is None:
                continue
            c = v * LST_SCALE - 273.15
            if i == 0:
                by_date[date]["urban"] = c
            else:
                # a rural point may fall in two adjacent tiles' edge; keep first
                by_date[date]["rural"].setdefault(i, c)
    ts = {}
    for date, v in by_date.items():
        if v["urban"] is not None and v["rural"]:
            ts[date] = {
                "urban_lst": round(v["urban"], 3),
                "rural_lst": round(float(np.median(list(v["rural"].values()))), 3),
                "n_rural": len(v["rural"]),
            }
    return ts


def fetch_ndvi_pc(city: dict, year: int = NDVI_YEAR) -> float | None:
    """Annual mean NDVI at the city point (mean of valid 16-day composites).
    If the pixel is fill in every composite, the mean of the valid pixels in a
    3×3 then 5×5 neighbourhood is used (mirrors the GEE coarser-scale retry)."""
    import rasterio
    from rasterio.warp import transform

    pt = (city["lon"], city["lat"])
    items = _search(NDVI_COLLECTION, NDVI_PREFIX, _bbox([pt]), f"{year}-01-01", f"{year}-12-31")
    for half in (0, 1, 2):
        vals = []
        for it in items:
            with rasterio.open(it.assets[NDVI_ASSET].href) as ds:
                (x,), (y,) = transform("EPSG:4326", ds.crs, [pt[0]], [pt[1]])
                r, c = ds.index(x, y)
                win = ((max(r - half, 0), r + half + 1), (max(c - half, 0), c + half + 1))
                arr = ds.read(1, window=win).astype(float)
                arr = arr[arr != NDVI_FILL]
                if arr.size:
                    vals.append(arr.mean() * NDVI_SCALE)
        if vals:
            return round(float(np.mean(vals)), 4)
    return None


def _collect_city(city: dict, start: str, end: str) -> list[dict]:
    ndvi = fetch_ndvi_pc(city)
    if ndvi is None:
        log.info(f"  Skipping {city['name']} (NDVI unavailable)")
        return []
    lst_ts = fetch_lst_timeseries_pc(city, start, end)
    weather = fetch_openmeteo_history(city, start, end)
    if not weather:
        log.info(f"  No weather data for {city['name']} — skipping")
        return []
    urban_fraction = round(float(np.clip(1.0 - ndvi, 0.0, 1.0)), 3)
    rows = []
    for rec in weather:
        rec = dict(rec)
        m = match_to_covering_lst(rec["timestamp"][:10], lst_ts)
        rec.update({
            "ndvi": ndvi,
            "urban_fraction": urban_fraction,
            "urban_lst": m["urban_lst"] if m else None,
            "rural_lst": m["rural_lst"] if m else None,
            "n_rural_valid": m["n_rural"] if m else None,
            "lst_date": m["lst_date"] if m else None,
            "source": SOURCE_TAG,
        })
        rows.append(rec)
    cover = sum(r["urban_lst"] is not None for r in rows)
    log.info(f"  {city['name']:15s} {len(rows)} rows  LST coverage {cover}/{len(rows)}  "
             f"composites {len(lst_ts)}  NDVI {ndvi}")
    return rows


def collect_data_pc(
    out_path: Path = RAW_DATA_PATH,
    cities: list[dict] | None = None,
    start: str = COLLECTION_START,
    end: str = COLLECTION_END,
    workers: int = 6,
) -> pd.DataFrame:
    cities = cities or CITIES
    log.info(f"Planetary Computer collection: {len(cities)} cities, {start} → {end}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda c: _collect_city(c, start, end), cities))
    rows = [r for city_rows in results for r in city_rows]
    if not rows:
        raise RuntimeError("No data collected from Planetary Computer.")
    df = pd.DataFrame(rows).sort_values(["name", "timestamp"]).reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(RAW_DATA_PATH))
    ap.add_argument("--start", default=COLLECTION_START)
    ap.add_argument("--end", default=COLLECTION_END)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    collect_data_pc(Path(a.out), start=a.start, end=a.end, workers=a.workers)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import OBSERVED_TARGET_PROVENANCE  # noqa: E402

FEATURES = ["temperature", "humidity", "wind_speed", "ndvi", "lat", "lon",
            "lat_abs", "distance_from_equator", "hour", "city_code"]


def make_processed(n_cities: int = 6, n_per: int = 30, seed: int = 0,
                   negative_city: bool = True) -> pd.DataFrame:
    """Compact synthetic modelling table with a known structure.

    * ``hour`` is constant (12) everywhere          → constant feature
    * ``distance_from_equator`` == ``lat_abs``      → exact duplicate
    * ``city_code`` identifies the city              → used by leakage spies
    * one city has strongly negative UHI            → sign preservation
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cities):
        lat = -30 + 12 * c
        ndvi = 0.1 + 0.08 * c
        base = -3.0 if (negative_city and c == 0) else 1.0 + 0.4 * c
        for d in range(n_per):
            t = 20 + 5 * np.sin(d / 5) + rng.normal(0, 1)
            y = base + 0.15 * (t - 20) + rng.normal(0, 0.5)
            rows.append({
                "observation_id": f"city{c}_{d:03d}",
                "city_name": f"City{c}",
                "date": f"2025-01-{d + 1:02d}" if d < 31 else f"2025-02-{d - 30:02d}",
                "lst_date": f"2025-01-{(d // 8) * 8 + 1:02d}",
                "temperature": t,
                "humidity": 50 + rng.normal(0, 10),
                "wind_speed": abs(rng.normal(3, 1)),
                "ndvi": ndvi, "lat": lat, "lon": 10.0 * c,
                "lat_abs": abs(lat), "distance_from_equator": abs(lat),
                "hour": 12.0, "city_code": float(c),
                "uhi_intensity": y,
                "target_provenance": OBSERVED_TARGET_PROVENANCE,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def processed():
    return make_processed()


@pytest.fixture
def features():
    return list(FEATURES)

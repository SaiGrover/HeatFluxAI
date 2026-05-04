"""
UHI Prediction System - Configuration

API keys are loaded from the .env file in this folder.
Never commit .env or any real keys to version control.
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (same folder as this file)
load_dotenv(Path(__file__).parent / ".env")

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API Credentials (loaded from .env — never hardcoded) ────────────────────
GEE_PROJECT_ID      = os.getenv("GEE_PROJECT_ID",      "YOUR_GEE_PROJECT_ID")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

CDS_API_URL = "https://cds.climate.copernicus.eu/api"
CDS_API_KEY = os.getenv("CDS_API_KEY", "YOUR_CDS_API_KEY")

# Warn (don't crash) if keys look like placeholders — lets the dashboard still open
_missing = [
    name for name, val in [
        ("GEE_PROJECT_ID",      GEE_PROJECT_ID),
        ("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY),
        ("CDS_API_KEY",         CDS_API_KEY),
    ]
    if val.startswith("YOUR_")
]
if _missing:
    warnings.warn(
        f"Missing env vars: {_missing}. "
        "Add them to the .env file before running data collection.",
        stacklevel=2,
    )

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
    # Core weather
    "temperature", "humidity", "wind_speed", "pressure", "clouds",
    # Satellite / land
    "ndvi", "urban_fraction", "veg_class",
    # Geographic
    "lat", "lon", "distance_from_equator",
    # Temporal
    "hour", "month", "is_daytime", "is_night",
    # Interaction / derived
    "temp_humidity_interaction", "wind_cooling_effect",
    "temp_anomaly", "heat_retention",
]
TARGET_COLUMN = "uhi_intensity"

# ─── Model Settings ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5          # used with GroupKFold — 5 folds across city groups

# ─── File Paths ───────────────────────────────────────────────────────────────
RAW_DATA_PATH       = DATA_DIR / "raw_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
BEST_MODEL_PATH     = MODEL_DIR / "best_model.pkl"
SCALER_PATH         = MODEL_DIR / "scaler.pkl"
METRICS_PATH        = MODEL_DIR / "metrics.json"
FEATURES_PATH       = MODEL_DIR / "feature_names.json"

# ─── Dashboard Settings ───────────────────────────────────────────────────────
DASHBOARD_PORT  = 8505
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
    "glow_blue":  "rgba(88,166,255,0.15)",
    "glow_purple":"rgba(188,140,255,0.15)",
    "glow_green": "rgba(63,185,80,0.15)",
}

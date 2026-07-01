"""Cloud-safe Streamlit entrypoint for HeatFluxAI.

The full local dashboard remains in `dashboard.py`. This file is intentionally
lighter for Streamlit Cloud so widget reruns do not rebuild the heavyweight
analysis dashboard or crash the session.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BEST_MODEL_PATH,
    CITIES,
    FEATURES_PATH,
    METRICS_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
)


st.set_page_config(
    page_title="HeatFluxAI",
    page_icon="!",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background:#0d1117; color:#e6edf3; }
    [data-testid="stHeader"] { background:#0d1117; }
    section[data-testid="stSidebar"] { background:#101722; }
    .hero {
        padding: 2.2rem 0 1.4rem;
        border-bottom: 1px solid #263244;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 3rem;
        line-height: 1.05;
        margin: 0 0 .7rem;
        color: #79a7ff;
    }
    .hero p { color:#a7b2c3; max-width: 860px; font-size: 1rem; }
    .metric-card {
        background:#161f2d;
        border:1px solid #2d3a4d;
        border-radius:14px;
        padding:1rem 1.1rem;
        min-height:120px;
    }
    .metric-card .label {
        color:#91a0b5;
        font-size:.72rem;
        text-transform:uppercase;
        letter-spacing:.08em;
    }
    .metric-card .value {
        font-size:2rem;
        font-weight:800;
        margin:.4rem 0;
        color:#e6edf3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None


def safe_read_json(path: Path) -> dict | list | None:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception:
        return None
    return None


def safe_read_pickle(path: Path):
    try:
        if path.exists():
            with open(path, "rb") as handle:
                return pickle.load(handle)
    except Exception:
        return None
    return None


@st.cache_data(ttl=120)
def load_artifacts():
    return {
        "raw": safe_read_csv(RAW_DATA_PATH),
        "processed": safe_read_csv(PROCESSED_DATA_PATH),
        "metrics": safe_read_json(METRICS_PATH),
        "features": safe_read_json(FEATURES_PATH),
    }


@st.cache_resource
def load_prediction_artifacts():
    return {
        "model": safe_read_pickle(BEST_MODEL_PATH),
        "scaler": safe_read_pickle(SCALER_PATH),
    }


def metric_card(label: str, value: str, detail: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div style="color:#7f8da3;font-size:.85rem">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_refresh_form():
    with st.sidebar:
        st.subheader("Live data")
        st.caption("Run a small cloud-safe refresh first. Increase only after it works.")
        with st.form("live_refresh_form"):
            city_count = st.number_input(
                "Cities to refresh",
                min_value=1,
                max_value=len(CITIES),
                value=3,
                step=1,
            )
            train_model = st.checkbox(
                "Train lightweight model",
                value=False,
            )
            submitted = st.form_submit_button("Refresh live data", use_container_width=True)

        if submitted:
            status = st.empty()
            try:
                selected = CITIES[: int(city_count)]
                status.info("Collecting live GEE/Open-Meteo data...")
                from data_collector import collect_data

                collect_data(force=True, cities=selected)

                status.info("Preprocessing refreshed data...")
                from preprocessor import preprocess

                preprocess(force=True)

                if train_model:
                    status.info("Training lightweight cloud model...")
                    from model_trainer import train

                    train(force=True, fast=True)
                else:
                    status.info("Training skipped.")

                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Refresh completed. Press browser refresh once to reload artifacts.")
            except Exception as exc:
                st.error("Live refresh failed.")
                st.exception(exc)


def render_sidebar_status(artifacts: dict, prediction_artifacts: dict):
    raw = artifacts["raw"]
    processed = artifacts["processed"]
    model = prediction_artifacts["model"]

    with st.sidebar:
        st.divider()
        st.subheader("Pipeline status")
        st.write("Data collected:", "yes" if raw is not None else "no")
        st.write("Data processed:", "yes" if processed is not None else "no")
        st.write("Model trained:", "yes" if model is not None else "no")


def render_overview(artifacts: dict):
    raw = artifacts["raw"]
    processed = artifacts["processed"]
    metrics = artifacts["metrics"] if isinstance(artifacts["metrics"], dict) else None
    features = artifacts["features"] if isinstance(artifacts["features"], list) else None

    st.markdown(
        """
        <div class="hero">
            <h1>Urban Heat Island Estimation System</h1>
            <p>
                HeatFluxAI estimates urban heat island intensity from live
                weather data and Google Earth Engine satellite signals.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    with cols[0]:
        metric_card("Raw samples", f"{len(raw):,}" if raw is not None else "-", "GEE + weather rows")
    with cols[1]:
        metric_card("Processed rows", f"{len(processed):,}" if processed is not None else "-", "ML-ready records")
    with cols[2]:
        metric_card("Features", f"{len(features):,}" if features else "-", "model inputs")
    with cols[3]:
        metric_card("Best RMSE", str(metrics.get("best_rmse", "-")) if metrics else "-", "degrees C")
    with cols[4]:
        metric_card("Best model", str(metrics.get("best_model", "-")) if metrics else "-", "current artifact")

    st.divider()

    if processed is None:
        st.info("No processed artifact is available yet. Use the sidebar refresh with 1-3 cities first.")
        return

    if "uhi_intensity" in processed.columns:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("UHI distribution")
            st.bar_chart(processed["uhi_intensity"].value_counts(bins=20).sort_index())
        with right:
            st.subheader("City summary")
            city_col = "city_name" if "city_name" in processed.columns else "name"
            if city_col in processed.columns:
                summary = (
                    processed.groupby(city_col)["uhi_intensity"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(12)
                    .reset_index()
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("Raw data preview"):
        if raw is not None:
            st.dataframe(raw.head(200), use_container_width=True)
        else:
            st.caption("No raw data artifact found.")

    with st.expander("Processed data preview"):
        st.dataframe(processed.head(200), use_container_width=True)


def render_prediction(artifacts: dict, prediction_artifacts: dict):
    model = prediction_artifacts["model"]
    scaler = prediction_artifacts["scaler"]
    features = artifacts["features"] if isinstance(artifacts["features"], list) else None

    st.subheader("Prediction")
    if model is None or scaler is None or not features:
        st.info("Train a lightweight model from the sidebar before using prediction.")
        return

    defaults = {
        "temperature": 32.0,
        "humidity": 55.0,
        "wind_speed": 3.0,
        "pressure": 1013.0,
        "clouds": 30.0,
        "ndvi": 0.35,
        "urban_fraction": 0.65,
        "veg_class": 1.0,
        "lat": 28.6,
        "lon": 77.2,
        "distance_from_equator": 28.6,
        "hour": 14.0,
        "month": 6.0,
        "is_daytime": 1.0,
        "is_night": 0.0,
        "temp_humidity_interaction": 17.6,
        "wind_cooling_effect": 9.0,
        "temp_anomaly": 0.0,
        "heat_retention": 20.8,
    }

    with st.form("predict_form"):
        values = {}
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                values[feature] = st.number_input(
                    feature,
                    value=float(defaults.get(feature, 0.0)),
                    step=0.1,
                )
        submitted = st.form_submit_button("Predict UHI")

    if submitted:
        row = pd.DataFrame([[values[f] for f in features]], columns=features)
        try:
            pred = float(model.predict(scaler.transform(row))[0])
            st.metric("Predicted UHI intensity", f"{pred:.2f} C")
        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)


artifacts = load_artifacts()
prediction_artifacts = load_prediction_artifacts()

render_refresh_form()
render_sidebar_status(artifacts, prediction_artifacts)

tab_overview, tab_prediction = st.tabs(["Overview", "Prediction"])
with tab_overview:
    render_overview(artifacts)
with tab_prediction:
    render_prediction(artifacts, prediction_artifacts)

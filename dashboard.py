"""
UHI Estimation System — Streamlit Dashboard  ✦ Elite Dark Edition ✦
ERA5 · MODIS 8-day LST · 25 global cities · nested leave-one-city-out evaluation
"""

import json, math, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
# Must be the first Streamlit command: importing config reads st.secrets.
st.set_page_config(
    page_title="UHI Estimation System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, BEST_MODEL_PATH,
    METRICS_PATH, FEATURES_PATH, COLORS, CITIES, MODEL_DIR,
    HELDOUT_PREDICTIONS_PATH, SNAPSHOT_DATA_PATH,
)
from eval_artifacts import (
    MODEL_DISPLAY_NAMES, load_evaluation, per_city_frame, residual_label,
    summary_rows,
)

OUTLIER_STATS_PATH = MODEL_DIR / "outlier_stats.json"

# ══════════════════════════════════════════════════════════════════════════════
# ✦  ELITE CSS  ✦
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Keyframe animations ─────────────────────────────────────── */
@keyframes gradientFlow {{
    0%   {{ background-position: 0%   50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0%   50%; }}
}}
@keyframes pulseGlow {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(88,166,255,0); }}
    50%       {{ box-shadow: 0 0 20px 4px rgba(88,166,255,0.18); }}
}}
@keyframes fadeInUp {{
    from {{ opacity:0; transform:translateY(16px); }}
    to   {{ opacity:1; transform:translateY(0);    }}
}}
@keyframes rotateBorder {{
    0%   {{ --angle: 0deg;   }}
    100% {{ --angle: 360deg; }}
}}
@keyframes dotPulse {{
    0%, 100% {{ transform:scale(1);   opacity:1;   }}
    50%       {{ transform:scale(1.4); opacity:0.6; }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position:  200% center; }}
}}
@keyframes ticker {{
    0%   {{ opacity:0; transform:translateY(8px);  }}
    15%  {{ opacity:1; transform:translateY(0);    }}
    85%  {{ opacity:1; transform:translateY(0);    }}
    100% {{ opacity:0; transform:translateY(-8px); }}
}}

/* ── Base ────────────────────────────────────────────────────── */
html, body, [class*="css"],
.stApp, .main, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {COLORS['background']} !important;
    color: {COLORS['text']} !important;
}}
.main .block-container {{
    padding: 0 2.5rem 3rem !important;
    max-width: 1500px;
    background-color: {COLORS['background']} !important;
}}
[data-testid="stHeader"] {{ background: {COLORS['background']} !important; border-bottom:1px solid #21262d; }}
[data-testid="stToolbar"] {{ display:none; }}
footer {{ display:none !important; }}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a0e18 0%, #0d1117 60%, #0a0e18 100%) !important;
    border-right: 1px solid #1c2230 !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding: 1.2rem 1rem !important;
}}
[data-testid="stSidebar"] [data-baseweb="radio"] label {{
    color: {COLORS['subtext']} !important;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 0.9rem;
    border-radius: 9px;
    transition: all 0.15s;
    display: flex; align-items: center;
}}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {{
    background: rgba(88,166,255,0.08) !important;
    color: {COLORS['primary']} !important;
    padding-left: 1.1rem;
}}
[data-testid="stSidebar"] hr {{ border-color: #1c2230 !important; margin:0.7rem 0; }}

/* ── Hero banner ─────────────────────────────────────────────── */
.hero {{
    background: linear-gradient(135deg, #0d1117 0%, #0e1520 40%, #12101c 70%, #0d1117 100%);
    border-bottom: 1px solid #21262d;
    padding: 2.4rem 2.5rem 2rem;
    margin: 0 -2.5rem 2rem;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(88,166,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 30%, rgba(188,140,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse at 50% 90%, rgba(63,185,80,0.04)  0%, transparent 50%);
    pointer-events: none;
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #e6edf3, {COLORS['primary']}, {COLORS['secondary']}, #e6edf3);
    background-size: 300% 300%;
    animation: gradientFlow 6s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}}
.hero-sub {{
    font-size: 0.88rem;
    color: {COLORS['subtext']};
    font-weight: 400;
    letter-spacing: 0.01em;
    max-width: 600px;
    line-height: 1.6;
}}
.hero-tags {{
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    margin-top: 1rem;
}}
.hero-tag {{
    background: rgba(255,255,255,0.04);
    border: 1px solid #2d333b;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: {COLORS['subtext']};
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}}

/* ── KPI / metric cards ──────────────────────────────────────── */
.kpi-card {{
    background: linear-gradient(145deg, {COLORS['card']} 0%, {COLORS['card2']} 100%);
    border: 1px solid {COLORS['border']};
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    cursor: default;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1),
                border-color 0.2s, box-shadow 0.2s;
    animation: fadeInUp 0.5s ease both;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--kpi-color, {COLORS['primary']}), transparent);
    opacity: 0.8;
}}
.kpi-card::after {{
    content: '';
    position: absolute;
    inset: 0; border-radius: 18px;
    background: radial-gradient(circle at 50% -20%, var(--kpi-color, {COLORS['primary']})0f 0%, transparent 70%);
    pointer-events: none;
}}
.kpi-card:hover {{
    transform: translateY(-5px) scale(1.02);
    border-color: var(--kpi-color, {COLORS['primary']});
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 0 1px var(--kpi-color, {COLORS['primary']})22;
}}
.kpi-icon  {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1;
    color: var(--kpi-color, {COLORS['primary']});
}}
.kpi-label {{
    font-size: 0.68rem; color: {COLORS['subtext']};
    font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-top: 0.4rem;
}}
.kpi-sub {{
    font-size: 0.7rem; color: #484f58;
    margin-top: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
}}

/* ── Section header ──────────────────────────────────────────── */
.sec-head {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem; font-weight: 700;
    color: {COLORS['text']};
    letter-spacing: -0.02em;
    margin: 1.8rem 0 0.25rem;
    display: flex; align-items: center; gap: 0.6rem;
}}
.sec-sub {{
    font-size: 0.82rem; color: {COLORS['subtext']};
    margin-bottom: 1.2rem; line-height: 1.55;
}}

/* ── Glass card ──────────────────────────────────────────────── */
.glass {{
    background: rgba(22,27,39,0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
}}

/* ── Status pipeline card ────────────────────────────────────── */
.pipe-card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    animation: fadeInUp 0.5s ease both;
}}
.pipe-card:hover {{ transform:translateY(-3px); }}
.pipe-step {{
    font-size: 0.6rem; font-weight: 800;
    letter-spacing: 0.12em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    padding: 0.15rem 0.55rem;
    border-radius: 5px;
    display: inline-block; margin-bottom: 0.6rem;
}}

/* ── Feat pill ───────────────────────────────────────────────── */
.feat-pill {{
    background: {COLORS['card2']};
    border: 1px solid {COLORS['border']};
    border-left: 3px solid {COLORS['primary']};
    border-radius: 7px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.35rem;
    font-size: 0.76rem;
    font-family: 'JetBrains Mono', monospace;
    color: {COLORS['text']};
    display: block;
    transition: border-left-color 0.15s, background 0.15s;
}}
.feat-pill:hover {{
    border-left-color: {COLORS['secondary']};
    background: rgba(188,140,255,0.06);
}}

/* ── Best badge ──────────────────────────────────────────────── */
.best-badge {{
    display: inline-block;
    background: linear-gradient(135deg, {COLORS['accent']}, #1cb845);
    color: #071a0f; font-size: 0.63rem; font-weight: 800;
    padding: 0.15rem 0.6rem; border-radius: 999px;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-left: 0.4rem; vertical-align: middle;
}}

/* ── Prediction box ──────────────────────────────────────────── */
.pred-box {{
    background: linear-gradient(135deg,
        rgba(88,166,255,0.07) 0%,
        rgba(188,140,255,0.07) 50%,
        rgba(63,185,80,0.05)  100%);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 22px;
    padding: 2rem 1.8rem;
    text-align: center;
    position: relative; overflow: hidden;
}}
.pred-box::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%,
        rgba(88,166,255,0.08) 0%, transparent 65%);
    pointer-events: none;
}}
.pred-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4rem; font-weight: 800;
    background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
    background-size: 200% 200%;
    animation: gradientFlow 4s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.04em; line-height: 1;
}}
.sev-badge {{
    display: inline-block;
    padding: 0.4rem 1.8rem;
    border-radius: 999px;
    font-weight: 800; font-size: 0.88rem;
    letter-spacing: 0.05em;
    margin-top: 0.8rem;
    border: 1px solid;
    transition: box-shadow 0.3s;
}}

/* ── Inline code / mono labels ───────────────────────────────── */
.mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['primary']};
    background: rgba(88,166,255,0.08);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
}}

/* ── Streamlit widget overrides ──────────────────────────────── */
[data-testid="stAlert"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
}}
[data-testid="stDataFrame"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important;
    overflow: hidden;
}}
[data-baseweb="select"] > div {{
    background: {COLORS['card2']} !important;
    border-color: {COLORS['border']} !important;
    color: {COLORS['text']} !important;
    border-radius: 9px !important;
}}
[data-baseweb="menu"] {{
    background: {COLORS['card2']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}}
[data-baseweb="option"]:hover {{ background: rgba(88,166,255,0.1) !important; }}

/* Slider */
[data-testid="stSlider"] [role="slider"] {{
    background: {COLORS['primary']} !important;
    border: 2px solid {COLORS['background']} !important;
    box-shadow: 0 0 0 3px {COLORS['primary']}44 !important;
}}

/* Tabs */
[data-baseweb="tab-list"] {{
    background: {COLORS['card']} !important;
    border-radius: 12px !important;
    border: 1px solid {COLORS['border']} !important;
    padding: 5px !important;
    gap: 3px;
}}
[data-baseweb="tab"] {{
    font-weight: 600 !important; font-size: 0.82rem !important;
    color: {COLORS['subtext']} !important; border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important;
    background: transparent !important; border: none !important;
    transition: all 0.15s !important;
}}
[data-baseweb="tab"]:hover {{
    color: {COLORS['text']} !important;
    background: rgba(255,255,255,0.05) !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: linear-gradient(135deg, {COLORS['primary']}, #3d8ef8) !important;
    color: #0d1117 !important;
    box-shadow: 0 2px 12px rgba(88,166,255,0.35) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display:none !important; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {COLORS['primary']}, #3d8ef8) !important;
    color: #0d1117 !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.88rem !important; padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 16px rgba(88,166,255,0.3) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.02em;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(88,166,255,0.45) !important;
}}

/* Download button */
[data-testid="stDownloadButton"] > button {{
    background: {COLORS['card2']} !important;
    color: {COLORS['primary']} !important;
    border: 1px solid {COLORS['primary']}33 !important;
    border-radius: 9px !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    transition: all 0.15s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: rgba(88,166,255,0.1) !important;
    border-color: {COLORS['primary']} !important;
    transform: translateY(-1px) !important;
}}

/* st.metric */
[data-testid="metric-container"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
}}
div[data-testid="stMetricValue"] {{
    font-size:1.7rem !important; font-weight:800 !important;
    color:{COLORS['text']} !important;
    font-family:'Space Grotesk',sans-serif !important;
}}
div[data-testid="stMetricLabel"] {{
    color:{COLORS['subtext']} !important;
    font-size:0.7rem !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:0.08em !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    background: {COLORS['card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 12px !important;
}}
[data-testid="stExpander"] summary {{
    color: {COLORS['text']} !important; font-weight: 600 !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{COLORS['background']}; }}
::-webkit-scrollbar-thumb {{
    background: #30363d; border-radius:3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: {COLORS['primary']}66; }}

/* hr */
hr {{ border-color: {COLORS['border']} !important; margin:1.2rem 0 !important; }}

/* ── Live ticker strip ───────────────────────────────────────── */
.ticker-wrap {{
    width:100%; overflow:hidden;
    background:linear-gradient(90deg,#080c14 0%,#0a0e18 50%,#080c14 100%);
    border-top:1px solid #141a26; border-bottom:1px solid #141a26;
    padding:0.42rem 0; margin-bottom:1.2rem; position:relative;
}}
.ticker-wrap::before,.ticker-wrap::after {{
    content:''; position:absolute; top:0; bottom:0; width:80px; z-index:2; pointer-events:none;
}}
.ticker-wrap::before {{ left:0;  background:linear-gradient(90deg,#0d1117,transparent); }}
.ticker-wrap::after  {{ right:0; background:linear-gradient(270deg,#0d1117,transparent); }}
.ticker {{ display:flex; animation:ticker-scroll 36s linear infinite; white-space:nowrap; }}
@keyframes ticker-scroll {{ from{{transform:translateX(0)}} to{{transform:translateX(-50%)}} }}
.ticker-item {{
    font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:600;
    color:#484f58; padding:0 2.5rem; display:inline-flex; align-items:center; gap:0.45rem;
}}
.ticker-item .val {{ color:{COLORS['primary']}; font-weight:800; }}
.ticker-dot {{ color:{COLORS['accent']}; font-size:0.5rem; margin:0 1rem; }}

/* ── Rank badge ──────────────────────────────────────────────── */
.rank-badge {{
    display:inline-flex;align-items:center;justify-content:center;
    width:22px;height:22px;border-radius:50%;
    font-size:0.6rem;font-weight:900;margin-right:0.35rem;vertical-align:middle;
}}
.rank-1 {{ background:linear-gradient(135deg,#ffd700,#d4a017);color:#1a0e00;box-shadow:0 0 8px #ffd70066; }}
.rank-2 {{ background:linear-gradient(135deg,#c0c0c0,#9e9e9e);color:#1a1a1a; }}
.rank-3 {{ background:linear-gradient(135deg,#cd7f32,#a0522d);color:#1a0a00; }}

/* ── Severity table ──────────────────────────────────────────── */
.sev-table {{ width:100%;border-collapse:collapse;font-size:0.78rem; }}
.sev-table th {{
    font-size:0.6rem;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;
    color:{COLORS['subtext']};padding:0.5rem 0.9rem;border-bottom:1px solid #1c2230;
    text-align:left;
}}
.sev-table td {{ padding:0.5rem 0.9rem;border-bottom:1px solid #111520; }}
.sev-row:hover {{ background:rgba(255,255,255,0.025); }}
.sev-dot {{
    display:inline-block;width:9px;height:9px;border-radius:50%;
    margin-right:0.4rem;vertical-align:middle;
}}

/* ── Neon gradient border card ───────────────────────────────── */
.neon-card {{
    background:linear-gradient(145deg,#0f1520,{COLORS['card']});
    border:1px solid rgba(88,166,255,0.18);
    border-radius:18px;padding:1.4rem 1.5rem;position:relative;overflow:hidden;
    box-shadow:0 0 40px rgba(88,166,255,0.04),inset 0 1px 0 rgba(255,255,255,0.04);
    transition:box-shadow 0.3s,border-color 0.3s;
}}
.neon-card:hover {{
    border-color:rgba(88,166,255,0.42);
    box-shadow:0 0 50px rgba(88,166,255,0.12),inset 0 1px 0 rgba(255,255,255,0.07);
}}

/* ── 3D chart glow wrapper ───────────────────────────────────── */
.chart-3d-wrap {{
    border:1px solid rgba(88,166,255,0.14);border-radius:16px;overflow:hidden;
    box-shadow:0 8px 40px rgba(0,0,0,0.35),0 0 60px rgba(88,166,255,0.04);
}}

/* ── Contribution bars ───────────────────────────────────────── */
.contrib-row {{
    display:flex;align-items:center;gap:0.5rem;margin:0.22rem 0;
    padding:0.18rem 0;font-size:0.77rem;
}}
.contrib-label {{
    width:155px;color:{COLORS['subtext']};font-family:'JetBrains Mono',monospace;
    font-size:0.7rem;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}}
.contrib-bar-wrap {{
    flex:1;background:{COLORS['card2']};border-radius:4px;overflow:hidden;
    height:13px;position:relative;
}}
.contrib-bar {{ height:100%;border-radius:4px;min-width:2px; }}
.contrib-val {{
    width:60px;text-align:right;font-family:'JetBrains Mono',monospace;
    font-size:0.72rem;font-weight:700;flex-shrink:0;
}}

/* ── Insight cards ───────────────────────────────────────────── */
.insight-card {{
    background:{COLORS['card']};border:1px solid {COLORS['border']};
    border-radius:12px;padding:0.9rem 1rem;margin-bottom:0.5rem;
    font-size:0.78rem;line-height:1.6;
    border-left:3px solid var(--ic,{COLORS['primary']});
}}
.insight-card strong {{ color:{COLORS['text']}; }}

/* ── Animated stat ticker inside overview ────────────────────── */
@keyframes statPop {{
    0%  {{ transform:scale(0.92);opacity:0; }}
    60% {{ transform:scale(1.04);opacity:1; }}
    100%{{ transform:scale(1);   opacity:1; }}
}}
.stat-pop {{ animation:statPop 0.35s cubic-bezier(.34,1.56,.64,1) both; }}

/* ── City flag pill ──────────────────────────────────────────── */
.city-pill {{
    display:inline-flex;align-items:center;gap:0.3rem;
    background:{COLORS['card2']};border:1px solid {COLORS['border']};
    border-radius:999px;padding:0.2rem 0.65rem;
    font-size:0.7rem;font-weight:600;color:{COLORS['text']};
    margin:0.15rem;white-space:nowrap;
    transition:background 0.15s,border-color 0.15s;
}}
.city-pill:hover {{
    background:rgba(88,166,255,0.1);border-color:{COLORS['primary']}44;
}}

/* ── Residual zero-line ──────────────────────────────────────── */
.res-note {{
    font-size:0.72rem;color:{COLORS['subtext']};margin-top:0.4rem;
    font-family:'JetBrains Mono',monospace;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ✦  CACHED LOADERS  ✦
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_raw_data():
    """Collected raw data; falls back to the committed evaluation snapshot."""
    try:
        for path in (RAW_DATA_PATH, SNAPSHOT_DATA_PATH):
            if path.exists():
                return pd.read_csv(path)
    except Exception:
        return None
    return None

@st.cache_data(ttl=300)
def load_processed_data():
    """Modelling table (stateless preprocessing only).  Old-format files that
    lack target provenance (they contained median-imputed, clipped targets) are
    rebuilt from the raw data instead of being displayed."""
    try:
        if PROCESSED_DATA_PATH.exists():
            df = pd.read_csv(PROCESSED_DATA_PATH)
            if "target_provenance" in df.columns:
                return df
        raw = load_raw_data()
        if raw is None:
            return None
        from preprocessor import build_processed_table
        return build_processed_table(raw)
    except Exception:
        return None

@st.cache_resource
def load_model():
    """Deployment sklearn Pipeline (raw features in).  Legacy pickles that
    expected externally scaled input are ignored rather than mis-used."""
    try:
        if BEST_MODEL_PATH.exists():
            with open(BEST_MODEL_PATH, "rb") as f:
                m = pickle.load(f)
            if hasattr(m, "named_steps") and "preprocess" in m.named_steps:
                return m
    except Exception:
        return None
    return None

@st.cache_resource
def load_display_scaler(_proc_df, feat_names_key):
    """Imputer+StandardScaler fitted on the full table for DISPLAY ONLY
    (scaled-input bars, city similarity).  Never used by the model, whose
    preprocessing lives inside its own fitted pipeline."""
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        cols = list(feat_names_key)
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler()).fit(_proc_df[cols].to_numpy(float))
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_eval_view():
    return load_evaluation(METRICS_PATH, HELDOUT_PREDICTIONS_PATH)

def load_metrics():
    """metrics.json only when it is a validated nested-LOCO evaluation."""
    view = load_eval_view()
    return view.metrics if view.ok else None

@st.cache_data
def load_feature_names():
    """Candidate model inputs (the deployment pipeline drops constant and
    duplicate columns internally)."""
    try:
        if FEATURES_PATH.exists():
            with open(FEATURES_PATH) as f:
                return json.load(f)
        from preprocessor import candidate_features
        return candidate_features()
    except Exception:
        return None

@st.cache_data(ttl=300)
def load_outlier_stats():
    try:
        if OUTLIER_STATS_PATH.exists():
            with open(OUTLIER_STATS_PATH) as f:
                return json.load(f)
    except Exception:
        return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ✦  PLOTLY DARK THEME  ✦
# ══════════════════════════════════════════════════════════════════════════════
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Space Grotesk, sans-serif",
              color=COLORS["text"], size=12),
    margin=dict(l=24, r=24, t=44, b=24),
    xaxis=dict(gridcolor="#1c2230", zerolinecolor="#1c2230",
               linecolor="#30363d", tickcolor=COLORS["subtext"]),
    yaxis=dict(gridcolor="#1c2230", zerolinecolor="#1c2230",
               linecolor="#30363d", tickcolor=COLORS["subtext"]),
    legend=dict(bgcolor="rgba(13,17,23,0.85)", bordercolor="#30363d",
                borderwidth=1, font_size=11),
    hoverlabel=dict(bgcolor=COLORS["card2"], bordercolor=COLORS["border"],
                    font_color=COLORS["text"], font_size=12),
)

PAL = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
       COLORS["warning"], COLORS["danger"],
       "#38bdf8","#f472b6","#34d399","#fb923c","#a78bfa","#fbbf24","#60a5fa"]

def sfig(fig): fig.update_layout(**_PL); return fig

def hex_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert '#RRGGBB' → 'rgba(R,G,B,alpha)' — Plotly fillcolor safe."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════════════════════════════════════
# ✦  HELPERS  ✦
# ══════════════════════════════════════════════════════════════════════════════
def kpi(col, icon, value, label, sub="", color=None):
    c = color or COLORS["primary"]
    col.markdown(f"""
    <div class="kpi-card" style="--kpi-color:{c};animation-delay:{0}s">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def sec(title, sub=""):
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


def build_input(temp, hum, wind, ndvi_val, uf, lat, lon,
                hour, month, pressure, clouds, feat_names):
    is_daytime = int(6 <= hour <= 18)
    is_night   = int(hour < 6 or hour > 18)
    t_h        = temp * hum / 100
    w_cool     = wind * max(0.0, temp - 20)
    ndvi_c     = max(-1.0, min(1.0, ndvi_val))
    veg_class  = 0 if ndvi_c < 0.2 else (2 if ndvi_c >= 0.5 else 1)
    heat_ret   = uf * temp / (wind + 1)
    h_sin = math.sin(2*math.pi*hour/24);  h_cos = math.cos(2*math.pi*hour/24)
    m_sin = math.sin(2*math.pi*month/12); m_cos = math.cos(2*math.pi*month/12)
    heat_idx = (-8.78 + 1.61*temp + 2.34*hum/100 - 0.15*temp*hum/100)
    m = {
        "temperature": temp, "humidity": hum, "wind_speed": wind,
        "pressure": pressure, "clouds": float(clouds),
        "ndvi": ndvi_val, "urban_fraction": uf, "veg_class": float(veg_class),
        "lat": lat, "lon": lon, "distance_from_equator": abs(lat),
        "hour": float(hour), "month": float(month),
        "is_daytime": float(is_daytime), "is_night": float(is_night),
        "temp_humidity_interaction": t_h,
        "wind_cooling_effect": w_cool,
        "temp_anomaly": 0.0,
        "heat_retention": heat_ret,
        "hour_sin": h_sin, "hour_cos": h_cos,
        "month_sin": m_sin, "month_cos": m_cos,
        "lat_abs": abs(lat),
        "lon_sin": math.sin(math.radians(lon)),
        "lon_cos": math.cos(math.radians(lon)),
        "heat_index": heat_idx,
    }
    return [m.get(f, 0.0) for f in feat_names]


def predict_uhi(rows):
    """Deployment-pipeline prediction from raw feature vectors (signed °C)."""
    X = pd.DataFrame(np.atleast_2d(np.asarray(rows, dtype=float)), columns=feat_names)
    return model.predict(X)


def sel_metric(m, key, model_key="nested_selected_model"):
    try:
        return m["models"][model_key][key]
    except Exception:
        return None


MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# UHI severity reference table data
SEV_LEVELS = [
    ("< 0 °C",  "Cool island","#58a6ff", "Urban surface cooler than its rural reference (common where surroundings are desert or bare soil). A valid observation, not an error."),
    ("0 – 1 °C","Negligible", "#3fb950", "Urban area barely warmer than surroundings. Dense vegetation or coastal breeze likely."),
    ("1 – 2 °C","Low",        "#3fb950", "Mild warming. Common in cities with parks & low building density."),
    ("2 – 3 °C","Moderate",   "#d29922", "Noticeable heat island. Higher energy demand; vulnerable populations at risk during heatwaves."),
    ("3 – 5 °C","High",       "#f85149", "Strong UHI. Significant health risk. Typical of dense megacities in summer."),
    ("> 5 °C",  "Extreme",    "#ff6b6b", "Extreme heat island. Life-threatening during heatwaves. Urgent green-infrastructure needed."),
]

# City emoji mapping for flair
CITY_EMOJI = {
    "Delhi":"🇮🇳","Mumbai":"🇮🇳","New York":"🇺🇸","Los Angeles":"🇺🇸",
    "London":"🇬🇧","Tokyo":"🇯🇵","Shanghai":"🇨🇳","São Paulo":"🇧🇷",
    "Cairo":"🇪🇬","Lagos":"🇳🇬","Jakarta":"🇮🇩","Mexico City":"🇲🇽",
    "Karachi":"🇵🇰","Beijing":"🇨🇳","Dhaka":"🇧🇩","Bangkok":"🇹🇭",
    "Kolkata":"🇮🇳","Chicago":"🇺🇸","Paris":"🇫🇷","Istanbul":"🇹🇷",
    "Sydney":"🇦🇺","Toronto":"🇨🇦","Singapore":"🇸🇬","Berlin":"🇩🇪","Seoul":"🇰🇷",
}


def render_ticker(proc_df_t, metrics_t, raw_df_t):
    """Render the animated live-stats ticker strip."""
    items = []
    if proc_df_t is not None and "uhi_intensity" in proc_df_t.columns:
        mu  = proc_df_t["uhi_intensity"].mean()
        mx  = proc_df_t["uhi_intensity"].max()
        items += [
            ("🌡 Global Mean UHI", f"{mu:.2f} °C"),
            ("🔥 Peak UHI", f"{mx:.2f} °C"),
        ]
        if "city_name" in proc_df_t.columns:
            top_city = proc_df_t.groupby("city_name")["uhi_intensity"].mean().idxmax()
            top_val  = proc_df_t.groupby("city_name")["uhi_intensity"].mean().max()
            items.append(("🏙 Hottest City", f"{top_city}  {top_val:.2f} °C"))
        items.append(("📊 Processed Rows", f"{len(proc_df_t):,}"))
    if raw_df_t is not None:
        items.append(("🛰 Raw Samples", f"{len(raw_df_t):,}"))
    if metrics_t:
        items += [
            ("🧪 Held-out cities", str(metrics_t["dataset"]["n_cities"])),
            ("📉 ML held-out RMSE", f"{sel_metric(metrics_t,'macro_rmse'):.2f} °C"),
            ("📏 Mean-baseline RMSE",
             f"{sel_metric(metrics_t,'macro_rmse','mean_baseline'):.2f} °C"),
        ]
    if not items:
        return
    # Duplicate for seamless scroll
    html_items = "".join(
        f'<span class="ticker-item">{lbl} <span class="val">{val}</span></span>'
        f'<span class="ticker-dot">◆</span>'
        for lbl,val in items
    ) * 2
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker">{html_items}</div></div>',
        unsafe_allow_html=True
    )


def render_severity_table():
    """Render the UHI severity reference table."""
    rows = "".join(
        f'<tr class="sev-row">'
        f'<td><span class="sev-dot" style="background:{c}"></span>'
        f'<b style="color:{c}">{rng}</b></td>'
        f'<td><b style="color:{c}">{sev}</b></td>'
        f'<td style="color:#8b949e;font-size:0.74rem">{desc}</td>'
        f'</tr>'
        for rng,sev,c,desc in SEV_LEVELS
    )
    st.markdown(f"""
    <div class="neon-card" style="padding:1rem 1.2rem">
        <div style="font-size:0.62rem;font-weight:800;text-transform:uppercase;
                    letter-spacing:0.12em;color:{COLORS['subtext']};margin-bottom:0.8rem">
            🌡️ UHI Severity Reference Guide
        </div>
        <table class="sev-table">
            <thead><tr>
                <th>Range</th><th>Severity</th><th>Implications</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)


def render_feature_contributions(vec_vals, feat_names_list, base_val=0.0):
    """
    Render a horizontal contribution bar for each feature.
    Uses the difference from a neutral (zero-scaled) prediction as proxy.
    """
    if not feat_names_list:
        return
    # Build feature → value map
    fv = dict(zip(feat_names_list, vec_vals))
    # Sort by absolute value (scaled values, so centres around 0)
    sorted_fv = sorted(fv.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    max_abs = max(abs(v) for _,v in sorted_fv) or 1.0

    rows_html = ""
    for fname, fval in sorted_fv:
        pct   = abs(fval) / max_abs * 100
        col   = COLORS["primary"] if fval >= 0 else COLORS["danger"]
        bg    = hex_rgba(col, 0.2)
        sign  = "+" if fval >= 0 else "−"
        # Keep each div on one line — Streamlit markdown parser is sensitive to newlines in HTML
        rows_html += (
            f'<div class="contrib-row">'
            f'<div class="contrib-label" title="{fname}">{fname}</div>'
            f'<div class="contrib-bar-wrap">'
            f'<div class="contrib-bar" style="width:{pct:.1f}%;background:{bg};border-right:2px solid {col}"></div>'
            f'</div>'
            f'<div class="contrib-val" style="color:{col}">{sign}{abs(fval):.2f}</div>'
            f'</div>'
        )

    header = (
        f'<div style="font-size:0.62rem;font-weight:800;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{COLORS["subtext"]};margin-bottom:0.7rem">'
        f'&#128208; Scaled Feature Values (top 12 by magnitude)</div>'
    )
    note = (
        f'<div style="font-size:0.7rem;color:{COLORS["subtext"]};'
        f'font-family:\'JetBrains Mono\',monospace;margin-top:0.6rem">'
        f'Standardised against the dataset for display; the model applies its own fold-fitted preprocessing.</div>'
    )
    card = (
        f'<div class="neon-card" style="padding:1rem 1.2rem">'
        f'{header}{rows_html}{note}'
        f'</div>'
    )
    st.markdown(card, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ✦  SIDEBAR  ✦
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
        <div style="font-size:3rem;line-height:1;filter:drop-shadow(0 0 16px #58a6ff88)">🌡️</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                    font-size:1.1rem;color:#e6edf3;margin-top:0.6rem;
                    letter-spacing:-0.01em">UHI Estimation</div>
        <div style="font-size:0.65rem;color:#484f58;margin-top:0.2rem;font-weight:600;
                    text-transform:uppercase;letter-spacing:0.12em">
            Urban Heat Island System
        </div>
    </div>""", unsafe_allow_html=True)

    nav = st.radio("nav", [
        "📊  Overview", "📁  Data Explorer", "🔬  Preprocessing",
        "🤖  Models",   "🗺️  Heatmap",       "🔮  Prediction",
    ], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)

    metrics    = load_metrics()
    feat_names = load_feature_names()

    if metrics:
        _sel_rmse = sel_metric(metrics, "macro_rmse")
        _sel_mae  = sel_metric(metrics, "macro_mae")
        _bl_rmse  = sel_metric(metrics, "macro_rmse", "mean_baseline")
        _n_cit    = metrics["dataset"]["n_cities"]
        _cell = lambda lbl, val, c: (
            f"<div style='background:rgba(255,255,255,0.03);border-radius:7px;padding:0.4rem;text-align:center'>"
            f"<div style='color:{COLORS['subtext']};font-size:0.6rem;text-transform:uppercase'>{lbl}</div>"
            f"<div style='color:{c};font-weight:700;font-family:JetBrains Mono,monospace;font-size:0.82rem'>{val}</div></div>")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{COLORS['card']},{COLORS['card2']});
                    border:1px solid {COLORS['border']};border-radius:14px;
                    padding:1rem;font-size:0.8rem;
                    border-left:3px solid {COLORS['accent']};margin-bottom:0.6rem">
            <div style="color:{COLORS['subtext']};font-size:0.62rem;font-weight:800;
                        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem">
                🧪 Held-out city evaluation
            </div>
            <div style="color:{COLORS['primary']};font-weight:800;font-size:0.9rem;
                        font-family:'Space Grotesk',sans-serif">Nested leave-one-city-out</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.6rem">
                {_cell("ML RMSE", f"{_sel_rmse:.2f}°C", COLORS['text'])}
                {_cell("ML MAE", f"{_sel_mae:.2f}°C", COLORS['text'])}
                {_cell("Mean-BL RMSE", f"{_bl_rmse:.2f}°C", COLORS['warning'])}
                {_cell("Cities", _n_cit, COLORS['accent'])}
            </div>
            <div style='margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid {COLORS['border']};font-size:0.66rem;color:{COLORS['subtext']}'>Macro averages over held-out cities. See Models tab.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.caption("No validated held-out evaluation found.")

    # Status dots
    checks = [
        ("Data collected",  RAW_DATA_PATH.exists(),    COLORS["primary"]),
        ("Data processed",  PROCESSED_DATA_PATH.exists(), COLORS["secondary"]),
        ("Model trained",   BEST_MODEL_PATH.exists(),   COLORS["accent"]),
    ]
    for label, ok, dot_c in checks:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.55rem;
                    padding:0.22rem 0.3rem;font-size:0.76rem;color:{COLORS['subtext']}">
            <span style="color:{dot_c if ok else COLORS['border']};font-size:0.75rem;
                         {'animation:dotPulse 2s ease infinite' if ok else ''}">●</span>
            {label}
        </div>""", unsafe_allow_html=True)


# ── Load everything once ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<hr>", unsafe_allow_html=True)
    refresh_city_count = st.slider(
        "Live refresh cities",
        min_value=4,
        max_value=len(CITIES),
        value=min(5, len(CITIES)),
        help="Use fewer cities on Streamlit Cloud to avoid killing the session. "
             "Nested leave-one-city-out evaluation needs at least 4 cities with "
             "observed LST; confidence intervals need at least 8.",
    )
    run_training = st.checkbox(
        "Train model after refresh",
        value=False,
        help="Uses a lightweight cloud-safe training pass.",
    )
    if st.button("Refresh live data", use_container_width=True):
        try:
            status_box = st.empty()
            selected_cities = CITIES[:refresh_city_count]

            with st.spinner(f"Collecting live data for {refresh_city_count} cities..."):
                from data_collector import collect_data
                from preprocessor import preprocess

                status_box.info("Step 1/3: collecting live GEE/Open-Meteo data...")
                collect_data(force=True, cities=selected_cities)
                status_box.info("Step 2/3: preprocessing refreshed data...")
                preprocess(force=True)

                if run_training:
                    from model_trainer import train

                    status_box.info("Step 3/3: training lightweight cloud model...")
                    train(force=True, fast=True)
                else:
                    status_box.info("Step 3/3: skipped model training.")

                st.cache_data.clear()
                st.cache_resource.clear()
            st.success("Live refresh completed. Refresh the browser once to reload the latest artifacts.")
        except Exception as exc:
            st.error(f"Live pipeline failed: {exc}")

raw_df        = load_raw_data()
proc_df       = load_processed_data()
model         = load_model()
eval_view     = load_eval_view()
metrics       = eval_view.metrics if eval_view.ok else None
feat_names    = load_feature_names()
if feat_names is None and model is not None:
    feat_names = list(getattr(model, "feature_names_in_", [])) or None
scaler        = (load_display_scaler(proc_df, tuple(feat_names))
                 if proc_df is not None and feat_names else None)
outlier_stats = load_outlier_stats()


# ══════════════════════════════════════════════════════════════════════════════
# ✦ TAB 1 — OVERVIEW  ✦
# ══════════════════════════════════════════════════════════════════════════════
if nav == "📊  Overview":

    # Hero
    tags = ["ERA5 reanalysis", "MODIS 8-day LST", "25 global cities",
            "Nested leave-one-city-out", "Observed targets only", "Zero synthetic data"]
    tag_html = "".join(f'<span class="hero-tag">{t}</span>' for t in tags)
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">Urban Heat Island Estimation System</div>
        <div class="hero-sub">
            End-to-end machine learning pipeline estimating UHI intensity from
            satellite-derived Land Surface Temperature and ERA5 reanalysis weather —
            fully automated, no synthetic data.
        </div>
        <div class="hero-tags">{tag_html}</div>
    </div>""", unsafe_allow_html=True)

    # ── Live stats ticker ─────────────────────────────────────────────────────
    render_ticker(proc_df, metrics, raw_df)

    # KPI cards
    n_raw   = f"{len(raw_df):,}"  if raw_df  is not None else "—"
    n_proc  = f"{len(proc_df):,}" if proc_df is not None else "—"
    n_feat  = len(feat_names) if feat_names else "—"
    n_model = metrics["dataset"]["n_cities"] if metrics else "—"
    b_rmse  = f"{sel_metric(metrics,'macro_rmse'):.2f} °C" if metrics else "—"
    skill_s = (f"{sel_metric(metrics,'macro_rmse') - sel_metric(metrics,'macro_rmse','mean_baseline'):+.2f} °C"
               if metrics else "—")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1,"🛰️", n_raw,  "Raw Samples",    "ERA5 + GEE rows",   COLORS["primary"])
    kpi(c2,"⚙️", n_proc, "Processed Rows", "after pipeline",    COLORS["secondary"])
    kpi(c3,"🔬", n_feat, "Features",        "engineered",        COLORS["accent"])
    kpi(c4,"🧪", n_model,"Held-out Cities", "nested LOCO folds", COLORS["warning"])
    kpi(c5,"📉", b_rmse, "Held-out RMSE",   "selected ML · macro", COLORS["danger"])
    kpi(c6,"📏", skill_s,"vs Mean Baseline","RMSE difference (− = better)", COLORS["accent"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline steps
    sec("⚙️ Pipeline Status")
    steps = [
        ("01","Data Collection",  raw_df  is not None,"🛰️","ERA5 + MODIS via GEE",COLORS["primary"]),
        ("02","Preprocessing",    proc_df is not None,"⚙️","stateless; fitted steps per fold",COLORS["secondary"]),
        ("03","Model Training",   metrics is not None,"🤖","nested leave-one-city-out",COLORS["accent"]),
        ("04","Dashboard",        True,               "📊","Streamlit on :8505",COLORS["warning"]),
    ]
    pipe_cols = st.columns(4)
    for col,(step,name,done,icon,desc,c) in zip(pipe_cols,steps):
        col.markdown(f"""
        <div class="pipe-card" style="border-left:3px solid {'c' if not done else c}">
            <div class="pipe-step" style="background:{c}22;color:{c}">STEP {step} {'✓' if done else '…'}</div>
            <div style="font-size:1.6rem;margin:0.3rem 0">{icon}</div>
            <div style="font-weight:700;font-size:0.88rem;
                        color:{'var(--text)' if done else COLORS['subtext']}">{name}</div>
            <div style="font-size:0.7rem;color:{COLORS['subtext']};margin-top:0.25rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if proc_df is not None and "uhi_intensity" in proc_df.columns:
        col_dist, col_rank = st.columns([3, 2])

        with col_dist:
            sec("📈 UHI Intensity Distribution")
            mean_u = proc_df["uhi_intensity"].mean()
            median_u = proc_df["uhi_intensity"].median()
            fig = px.histogram(proc_df, x="uhi_intensity", nbins=55,
                               color_discrete_sequence=[COLORS["primary"]],
                               labels={"uhi_intensity":"UHI Intensity (°C)"})
            fig.update_traces(marker_line_color=COLORS["card"],
                              marker_line_width=0.5, opacity=0.85)
            fig.add_vline(x=mean_u, line_dash="dash", line_color=COLORS["warning"],
                          annotation_text=f"Mean {mean_u:.2f}°C",
                          annotation_font_color=COLORS["warning"])
            fig.add_vline(x=median_u, line_dash="dot", line_color=COLORS["secondary"],
                          annotation_text=f"Median {median_u:.2f}°C",
                          annotation_font_color=COLORS["secondary"],
                          annotation_position="top left")
            fig.update_layout(title="All cities · 180-day window", **_PL)
            st.plotly_chart(fig, use_container_width=True)

        with col_rank:
            sec("🏙️ City Rankings")
            if "city_name" in proc_df.columns:
                city_s = (proc_df.groupby("city_name")["uhi_intensity"]
                          .agg(["mean","std","min","max"])
                          .sort_values("mean", ascending=True)
                          .reset_index())
                fig2 = go.Figure()
                # Error bar shows std
                fig2.add_trace(go.Bar(
                    x=city_s["mean"], y=city_s["city_name"],
                    orientation="h",
                    marker=dict(
                        color=city_s["mean"],
                        colorscale=[[0,COLORS["accent"]],[0.45,COLORS["warning"]],[1,COLORS["danger"]]],
                        showscale=False,
                    ),
                    error_x=dict(array=city_s["std"],color=COLORS["subtext"],thickness=1.2),
                    text=city_s["mean"].round(2),
                    textposition="outside", textfont_color=COLORS["subtext"],
                    hovertemplate="<b>%{y}</b><br>Mean UHI: %{x:.2f}°C<br>Std: %{error_x.array:.2f}°C<extra></extra>",
                ))
                fig2.update_layout(title="Mean UHI ± std", xaxis_title="°C",
                                   height=480, **_PL)
                st.plotly_chart(fig2, use_container_width=True)

    # ── Severity guide + violin ───────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    sv_col, vio_col = st.columns([1.15, 1])
    with sv_col:
        render_severity_table()
    with vio_col:
        if proc_df is not None and "uhi_intensity" in proc_df.columns:
            sec("🎻 UHI Distribution (Violin)")
            if "city_name" in proc_df.columns:
                top8 = (proc_df.groupby("city_name")["uhi_intensity"]
                        .mean().sort_values(ascending=False).head(8).index.tolist())
                vio_df = proc_df[proc_df["city_name"].isin(top8)]
                fig_vio = go.Figure()
                for i, city in enumerate(top8):
                    vals = vio_df[vio_df["city_name"]==city]["uhi_intensity"]
                    fig_vio.add_trace(go.Violin(
                        y=vals, name=CITY_EMOJI.get(city,"🏙") + " " + city,
                        line_color=PAL[i%len(PAL)],
                        fillcolor=hex_rgba(PAL[i%len(PAL)], 0.13),
                        meanline_visible=True, box_visible=True,
                        points=False,
                    ))
                fig_vio.update_layout(
                    title="Top-8 hottest cities — distribution",
                    yaxis_title="UHI (°C)", showlegend=False,
                    violinmode="overlay", height=310, **_PL)
                st.plotly_chart(fig_vio, use_container_width=True)

    # Mini world scatter map
    if proc_df is not None:
        sec("🌍 Global UHI Map")
        if "city_name" in proc_df.columns and "uhi_intensity" in proc_df.columns:
            # Build city_agg using lat/lon from proc_df (already feature columns)
            # or fall back to raw_df lookup if proc_df lacks them
            _has_latlon = "lat" in proc_df.columns and "lon" in proc_df.columns
            if _has_latlon:
                city_agg = (proc_df.groupby("city_name")
                            .agg(lat=("lat","first"), lon=("lon","first"),
                                 uhi_intensity=("uhi_intensity","mean"))
                            .reset_index())
            elif raw_df is not None and "name" in raw_df.columns:
                _coord = raw_df[["name","lat","lon"]].drop_duplicates("name").rename(columns={"name":"city_name"})
                city_agg = (proc_df.groupby("city_name")["uhi_intensity"]
                            .mean().reset_index()
                            .merge(_coord, on="city_name", how="left"))
            else:
                city_agg = None

            if city_agg is not None and not city_agg.empty:
                city_agg = city_agg.dropna(subset=["lat","lon","uhi_intensity"])
                city_agg["|UHI|"] = city_agg["uhi_intensity"].abs() + 0.2
                fig3 = px.scatter_mapbox(
                    city_agg,
                    lat="lat", lon="lon",
                    color="uhi_intensity", size="|UHI|",
                    hover_name="city_name",
                    hover_data={"uhi_intensity":":.2f","lat":False,"lon":False,"|UHI|":False},
                    color_continuous_scale=[[0,COLORS["primary"]],[0.5,"#e6edf3"],[1,COLORS["danger"]]],
                    color_continuous_midpoint=0,
                    size_max=28, zoom=1.1, mapbox_style="carto-darkmatter",
                )
                fig3.update_layout(
                    height=420, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor=COLORS["card"],
                    coloraxis_colorbar=dict(
                        title=dict(text="°C",font_color=COLORS["subtext"]),
                        tickcolor=COLORS["subtext"], tickfont_color=COLORS["subtext"],
                        bgcolor=COLORS["card"], outlinecolor=COLORS["border"],
                    ),
                    font_color=COLORS["text"],
                )
                st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ✦ TAB 2 — DATA EXPLORER  ✦
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📁  Data Explorer":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    sec("📁 Data Explorer", "Inspect raw ERA5 + GEE data and the processed feature matrix")

    tab_raw, tab_proc, tab_ts, tab_corr = st.tabs([
        "📄  Raw Data", "✨  Processed", "📈  Time Series", "🔗  Correlations"])

    with tab_raw:
        if raw_df is None:
            st.warning("Run `python main.py` first.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Rows",    f"{len(raw_df):,}")
            c2.metric("Columns", len(raw_df.columns))
            c3.metric("Cities",  raw_df["name"].nunique() if "name" in raw_df.columns else "—")
            if "timestamp" in raw_df.columns:
                sp = pd.to_datetime(raw_df["timestamp"],errors="coerce")
                c4.metric("Span", f"{int((sp.max()-sp.min()).days)}d" if not sp.isna().all() else "—")
            st.markdown("<br>", unsafe_allow_html=True)

            ca, cb = st.columns([1,2])
            with ca:
                if "source" in raw_df.columns:
                    fig = px.pie(
                        raw_df.groupby("source").size().reset_index(name="count"),
                        values="count", names="source",
                        hole=0.5, color_discrete_sequence=PAL, title="Data Sources",
                    )
                    fig.update_traces(textfont_color=COLORS["text"])
                    sfig(fig); st.plotly_chart(fig, use_container_width=True)
            with cb:
                num_cols = [c for c in ["temperature","humidity","wind_speed","ndvi","urban_fraction"]
                            if c in raw_df.columns]
                if num_cols:
                    fig2 = go.Figure()
                    for i,col in enumerate(num_cols):
                        fig2.add_trace(go.Box(
                            y=raw_df[col], name=col.replace("_"," ").title(),
                            marker_color=PAL[i%len(PAL)], boxmean="sd",
                            boxpoints="outliers",
                            marker=dict(outliercolor=COLORS["warning"],size=3),
                        ))
                    fig2.update_layout(title="Key feature distributions",
                                       showlegend=False, **_PL)
                    st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(raw_df.head(300), use_container_width=True, height=280)
            st.download_button("⬇ Download Raw CSV",raw_df.to_csv(index=False),"raw_data.csv","text/csv")

    with tab_proc:
        if proc_df is None:
            st.warning("Run `python main.py` first.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Rows",     f"{len(proc_df):,}")
            c2.metric("Candidate features", len(feat_names) if feat_names else "—")
            if "uhi_intensity" in proc_df.columns:
                c3.metric("UHI Range",
                          f"{proc_df['uhi_intensity'].min():.1f}–{proc_df['uhi_intensity'].max():.1f}°C")
            st.markdown("<br>", unsafe_allow_html=True)

            key_cols = [c for c in proc_df.select_dtypes(include=np.number).columns
                        if c in (feat_names or []) + ["uhi_intensity"]][:22]
            if len(key_cols) > 2:
                corr = proc_df[key_cols].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index,
                    colorscale=[[0,COLORS["danger"]],[0.5,"#21262d"],[1,COLORS["primary"]]],
                    zmid=0, text=corr.values.round(2),
                    texttemplate="%{text}", textfont_size=8,
                    colorbar=dict(tickcolor=COLORS["subtext"],
                                  tickfont_color=COLORS["subtext"],
                                  bgcolor=COLORS["card"],
                                  outlinecolor=COLORS["border"]),
                ))
                fig.update_layout(title="Pearson correlation matrix", height=520, **_PL)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(proc_df.head(300), use_container_width=True, height=280)
            st.download_button("⬇ Download Processed CSV",
                               proc_df.to_csv(index=False),"processed_data.csv","text/csv")

    with tab_ts:
        if proc_df is None or raw_df is None:
            st.warning("Run `python main.py` first.")
        else:
            sec("📈 UHI Intensity — Time Series per City")
            # Merge timestamps
            if "date" in proc_df.columns and "city_name" in proc_df.columns:
                ts_df = proc_df.copy()
                ts_df["date"] = pd.to_datetime(ts_df["date"], errors="coerce").dt.date

                city_list = sorted(ts_df["city_name"].dropna().unique().tolist())
                sel_cities = st.multiselect("Select cities", city_list,
                                            default=city_list[:6])
                if sel_cities:
                    fig = go.Figure()
                    for i,city in enumerate(sel_cities):
                        sub = (ts_df[ts_df["city_name"]==city]
                               .groupby("date")["uhi_intensity"].mean()
                               .reset_index())
                        fig.add_trace(go.Scatter(
                            x=sub["date"], y=sub["uhi_intensity"],
                            mode="lines", name=city,
                            line=dict(color=PAL[i%len(PAL)],width=2),
                            hovertemplate=f"<b>{city}</b><br>%{{x}}<br>UHI: %{{y:.2f}}°C<extra></extra>",
                        ))
                    fig.update_layout(
                        title="Observed UHI (8-day MODIS composites; gaps = no observed LST, not imputed)",
                        xaxis_title="Date", yaxis_title="UHI (°C)",
                        hovermode="x unified", **_PL)
                    st.plotly_chart(fig, use_container_width=True)

                    # Monthly average
                    sec("📅 Monthly Average UHI")
                    if "month" in ts_df.columns:
                        sub2 = (ts_df[ts_df["city_name"].isin(sel_cities)]
                                .groupby(["city_name","month"])["uhi_intensity"]
                                .mean().reset_index())
                        sub2["month_name"] = sub2["month"].map(MONTH_NAMES)
                        fig2 = px.line(sub2, x="month_name", y="uhi_intensity",
                                       color="city_name",
                                       color_discrete_sequence=PAL,
                                       markers=True,
                                       labels={"uhi_intensity":"UHI (°C)",
                                               "month_name":"Month",
                                               "city_name":"City"})
                        fig2.update_layout(title="Seasonal UHI pattern", **_PL)
                        st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Timestamp data not available in this dataset snapshot.")

    with tab_corr:
        if proc_df is None:
            st.warning("Run `python main.py` first.")
        elif "uhi_intensity" in proc_df.columns and feat_names:
            sec("🔗 Feature Correlation with UHI Intensity",
                "Pearson r — how strongly each feature co-varies with the target")
            num_f = [f for f in feat_names if f in proc_df.columns]
            corr_uhi = (proc_df[num_f + ["uhi_intensity"]]
                        .corr()["uhi_intensity"]
                        .drop("uhi_intensity")
                        .sort_values())
            bar_c = [COLORS["danger"] if v < 0 else COLORS["accent"]
                     for v in corr_uhi.values]
            fig = go.Figure(go.Bar(
                x=corr_uhi.values, y=corr_uhi.index,
                orientation="h", marker_color=bar_c,
                text=corr_uhi.values.round(3),
                textposition="outside", textfont_color=COLORS["subtext"],
                hovertemplate="<b>%{y}</b><br>r = %{x:.4f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color=COLORS["border"], line_width=1.5)
            fig.update_layout(
                title="Pearson r with uhi_intensity  (red=negative, green=positive)",
                xaxis_title="Pearson r", height=520, **_PL)
            st.plotly_chart(fig, use_container_width=True)

            # Scatter: top positive correlator
            top_feat = corr_uhi.abs().idxmax()
            st.markdown(f'<div class="sec-sub">Top correlator: <span class="mono">{top_feat}</span></div>',
                        unsafe_allow_html=True)
            if top_feat in proc_df.columns:
                color_by = "city_name" if "city_name" in proc_df.columns else None
                fig2 = px.scatter(
                    proc_df.sample(min(2000,len(proc_df)), random_state=42),
                    x=top_feat, y="uhi_intensity",
                    color=color_by,
                    color_discrete_sequence=PAL,
                    opacity=0.5, trendline="ols",
                    trendline_scope="overall",
                    trendline_color_override=COLORS["warning"],
                    labels={"uhi_intensity":"UHI (°C)"},
                )
                fig2.update_traces(marker_size=4)
                fig2.update_layout(title=f"{top_feat} vs UHI Intensity",
                                   showlegend=color_by is not None, **_PL)
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ✦ TAB 3 — PREPROCESSING  ✦
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔬  Preprocessing":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    sec("🔬 Preprocessing & Feature Engineering",
        "Stateless cleaning & feature engineering here · imputation, IQR capping, scaling and "
        "constant/duplicate removal are fitted inside each CV fold · signed, observed-only UHI target")

    if raw_df is None or proc_df is None:
        st.warning("Run the pipeline first: `python main.py`")
    else:
        c1, c2 = st.columns(2)
        with c1:
            sec("🚨 Missing Values — Raw Data")
            missing = raw_df.isnull().sum(); missing = missing[missing>0]
            if missing.empty:
                st.markdown(f"""<div class="glass" style="text-align:center;color:{COLORS['accent']};
                    font-weight:700;font-size:0.9rem;padding:1.2rem">
                    ✓ No missing values in raw data</div>""", unsafe_allow_html=True)
            else:
                fig = px.bar(x=missing.values, y=missing.index, orientation="h",
                             color=missing.values,
                             color_continuous_scale=[[0,COLORS["warning"]],[1,COLORS["danger"]]],
                             labels={"x":"Missing Count","y":"Column"})
                fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                  title="Columns with nulls", **_PL)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            sec("📐 Shape: Before → After")
            cmp = pd.DataFrame({"Stage":["Raw","Processed"],
                                "Rows":[len(raw_df),len(proc_df)],
                                "Columns":[len(raw_df.columns),len(proc_df.columns)]})
            fig2 = go.Figure()
            fig2.add_bar(x=cmp["Stage"],y=cmp["Rows"],name="Rows",
                         marker_color=COLORS["primary"],
                         text=cmp["Rows"],textposition="outside")
            fig2.add_bar(x=cmp["Stage"],y=cmp["Columns"],name="Columns",
                         marker_color=COLORS["secondary"],
                         text=cmp["Columns"],textposition="outside")
            fig2.update_layout(barmode="group",title="Rows & Columns",**_PL)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Outlier section ─────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        if "target_provenance" in proc_df.columns:
            sec("🎯 Target provenance",
                "Only observed MODIS LST differences are used for training, tuning and scoring")
            prov = proc_df["target_provenance"].value_counts()
            obs_y = proc_df.loc[proc_df["target_provenance"] == "observed_modis_lst_difference",
                                "uhi_intensity"]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Observed targets", f"{int(prov.get('observed_modis_lst_difference', 0)):,}")
            t2.metric("Excluded (no LST, not imputed)", f"{int(len(proc_df) - len(obs_y)):,}")
            t3.metric("Negative UHI kept", f"{int((obs_y < 0).sum()):,}")
            t4.metric("Observed range", f"{obs_y.min():.1f} … {obs_y.max():.1f} °C" if len(obs_y) else "—")
            st.markdown("<hr>", unsafe_allow_html=True)
        sec("📦 Outlier Detection (descriptive)",
            "Tukey fences [Q1−1.5·IQR, Q3+1.5·IQR] on the full table, for inspection only — "
            "the model's capping fences are re-learned on each training fold")

        if outlier_stats:
            cap_cols = [c for c,s in outlier_stats.items() if s["n_capped"]>0]
            if cap_cols:
                cards = st.columns(min(len(cap_cols),5))
                for col_w,feat in zip(cards,cap_cols):
                    s = outlier_stats[feat]
                    col_w.markdown(f"""
                    <div class="kpi-card" style="--kpi-color:{COLORS['warning']}">
                        <div class="kpi-icon">📌</div>
                        <div class="kpi-value" style="color:{COLORS['warning']}">{s['n_capped']}</div>
                        <div class="kpi-label">{feat.replace('_',' ')}</div>
                        <div class="kpi-sub">{s['pct']}% · [{s['lower_fence']}, {s['upper_fence']}]</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="glass" style="text-align:center;color:{COLORS['accent']};
                    font-weight:700;padding:1rem">✓ No outliers found — all within 1.5 × IQR</div>""",
                    unsafe_allow_html=True)

            with st.expander("📋 Full IQR Statistics Table"):
                iqr_rows=[{"Feature":k,"Q1":v["q1"],"Q3":v["q3"],"IQR":v["iqr"],
                           "Lower Fence":v["lower_fence"],"Upper Fence":v["upper_fence"],
                           "Capped Low":v["n_low"],"Capped High":v["n_high"],
                           "Total":v["n_capped"],"% Rows":v["pct"]}
                          for k,v in outlier_stats.items()]
                iqr_df=pd.DataFrame(iqr_rows)
                st.dataframe(iqr_df.style
                    .applymap(lambda v:"color:#d29922;font-weight:700"
                              if isinstance(v,(int,float)) and v>0 else "",
                              subset=["Total","Capped Low","Capped High"])
                    .format({"Q1":"{:.3f}","Q3":"{:.3f}","IQR":"{:.3f}",
                             "Lower Fence":"{:.3f}","Upper Fence":"{:.3f}","% Rows":"{:.2f}%"}),
                    use_container_width=True, hide_index=True)

            # Box plots raw vs capped
            avail = [c for c in outlier_stats if c in raw_df.columns and c in proc_df.columns]
            if avail:
                sec("📊 Box Plots with full-table fences")
                n_c = min(len(avail),3)
                box_cols = st.columns(n_c)
                for i,feat in enumerate(avail):
                    s = outlier_stats[feat]
                    with box_cols[i%n_c]:
                        fb = go.Figure()
                        fb.add_trace(go.Box(y=raw_df[feat].dropna(),name="Raw",
                                            marker_color=COLORS["primary"],boxmean="sd",
                                            marker=dict(outliercolor=COLORS["warning"],
                                                        symbol="circle-open",size=4),
                                            boxpoints="outliers",line_width=1.5))
                        for fence,txt,anchor in [(s["upper_fence"],"↑","bottom right"),
                                                  (s["lower_fence"],"↓","top right")]:
                            fb.add_hline(y=fence,line_dash="dot",
                                         line_color=COLORS["danger"],line_width=1,
                                         annotation_text=f"{txt}{fence}",
                                         annotation_font_color=COLORS["danger"],
                                         annotation_font_size=9,
                                         annotation_position=anchor)
                        fb.update_layout(**{**_PL,
                                          "title": feat.replace("_"," ").title(),
                                          "height": 310, "showlegend": True,
                                          "legend": dict(orientation="h", y=1.12,
                                                         font_size=9,
                                                         bgcolor="rgba(0,0,0,0)")})
                        st.plotly_chart(fb, use_container_width=True)

            # Stacked bar
            if any(s["n_capped"]>0 for s in outlier_stats.values()):
                oc=pd.DataFrame([{"Feature":k,"Below":v["n_low"],"Above":v["n_high"]}
                                  for k,v in outlier_stats.items()])
                fig_oc=go.Figure()
                fig_oc.add_trace(go.Bar(name="Below lower fence",x=oc["Feature"],
                                        y=oc["Below"],marker_color=COLORS["primary"],
                                        text=oc["Below"],textposition="auto"))
                fig_oc.add_trace(go.Bar(name="Above upper fence",x=oc["Feature"],
                                        y=oc["Above"],marker_color=COLORS["warning"],
                                        text=oc["Above"],textposition="auto"))
                fig_oc.update_layout(barmode="stack",
                                     title="Outlier counts by column",
                                     xaxis_title="Feature",yaxis_title="Count",**_PL)
                st.plotly_chart(fig_oc, use_container_width=True)

        # ── Feature engineering ─────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        sec("🏗️ Engineered Features")
        if feat_names:
            GROUPS = {
                "🌡️ Weather":    ["temperature","humidity","wind_speed","pressure","clouds"],
                "🛰️ Satellite":  ["ndvi","urban_fraction","veg_class"],
                "📍 Geographic": ["lat","lon","distance_from_equator","lat_abs","lon_sin","lon_cos"],
                "⏰ Temporal":   ["hour","month","is_daytime","is_night","hour_sin","hour_cos","month_sin","month_cos"],
                "🔬 Derived":    ["temp_humidity_interaction","wind_cooling_effect","temp_anomaly","heat_retention","heat_index"],
            }
            gcols = st.columns(len(GROUPS))
            for gc,(gname,gfeats) in zip(gcols,GROUPS.items()):
                gc.markdown(f"""<div style="font-size:0.7rem;font-weight:800;
                    color:{COLORS['subtext']};text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:0.5rem">{gname}</div>""",
                    unsafe_allow_html=True)
                for f in gfeats:
                    if f in feat_names:
                        gc.markdown(f'<span class="feat-pill">{f}</span>',
                                    unsafe_allow_html=True)

        # Distribution comparison
        common=[c for c in["temperature","humidity","wind_speed","ndvi","uhi_intensity"]
                if c in raw_df.columns or c in proc_df.columns]
        if common:
            st.markdown("<br>", unsafe_allow_html=True)
            sec("📉 Feature Distribution: Raw vs Processed")
            dcol,_ = st.columns([1,2])
            with dcol: sel=st.selectbox("Feature",common)
            fig3=make_subplots(rows=1,cols=2,subplot_titles=["Raw","Processed"])
            if sel in raw_df.columns:
                fig3.add_trace(go.Histogram(x=raw_df[sel],marker_color=COLORS["primary"],
                                            opacity=0.8,name="Raw",nbinsx=40,
                                            marker_line_color=COLORS["card"],marker_line_width=0.5),
                               row=1,col=1)
            if sel in proc_df.columns:
                fig3.add_trace(go.Histogram(x=proc_df[sel],marker_color=COLORS["secondary"],
                                            opacity=0.8,name="Processed",nbinsx=40,
                                            marker_line_color=COLORS["card"],marker_line_width=0.5),
                               row=1,col=2)
            fig3.update_layout(showlegend=False,**_PL)
            st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ✦ TAB 4 — MODELS  ✦
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🤖  Models":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    sec("🤖 Model Evaluation — Nested Leave-One-City-Out",
        "Each city is held out once · algorithm + hyper-parameters chosen by inner GroupKFold "
        "on the other cities · fold-local preprocessing · observed (non-imputed) targets only")

    if not eval_view.ok:
        st.warning(eval_view.message)
    else:
        ev_m   = eval_view.metrics
        ev_p   = eval_view.predictions
        ds     = ev_m["dataset"]
        ev_cfg = ev_m["evaluation"]
        ci_blk = ev_m["aggregate"]["confidence_intervals"]

        # ── Methodology card ─────────────────────────────────────────────────
        st.markdown(f"""<div class="glass" style="font-size:0.8rem;line-height:1.75">
            <b style="color:{COLORS['primary']}">Outer loop</b> {ev_cfg['outer_cv']} ·
            <b style="color:{COLORS['primary']}">Inner loop</b> {ev_cfg['inner_cv']} ·
            <b style="color:{COLORS['primary']}">Selection metric</b> inner-CV RMSE
            (test cities never used) · seed {ev_cfg['random_seed']}<br>
            <b>{ds['n_cities']}</b> cities · <b>{ds['n_evaluated_observations']:,}</b> observed targets
            scored · <b>{ds['n_excluded_targets']:,}</b> rows excluded (no observed LST pair, never imputed) ·
            <b>{ds['target_summary']['n_negative']:,}</b> negative UHI targets kept ·
            {ds.get('n_unique_city_lst_composites') or '—'} distinct (city, 8-day composite) targets
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Comparison table ─────────────────────────────────────────────────
        sec("📋 Held-out city comparison",
            "Macro = mean over cities (each city counts once) · Pooled = over all held-out "
            "observations · 95% CI = city-level bootstrap")
        rows = summary_rows(ev_m)
        fmt_ci = lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]" if isinstance(v, list) else "n/a"
        df_cmp = pd.DataFrame([{
            "Model": r["Model"], "Macro MAE": r["Macro MAE"], "Macro RMSE": r["Macro RMSE"],
            "Macro RMSE 95% CI": fmt_ci(r["Macro RMSE 95% CI"]),
            "Pooled MAE": r["Pooled MAE"], "Pooled RMSE": r["Pooled RMSE"],
            "Pooled RMSE 95% CI": fmt_ci(r["Pooled RMSE 95% CI"]),
            "Cities": r["Cities"], "Obs": r["Obs"], "Evaluation": r["Evaluation"],
        } for r in rows])
        st.dataframe(df_cmp.style.format({c: "{:.3f}" for c in
                     ["Macro MAE", "Macro RMSE", "Pooled MAE", "Pooled RMSE"]}),
                     use_container_width=True, hide_index=True)
        if ci_blk.get("status") == "unavailable":
            st.info(f"Confidence intervals not reported: {ci_blk.get('reason')}")
        else:
            diffs = ci_blk.get("paired_differences", {})
            d_mean = diffs.get("nested_selected_model - mean_baseline", {}).get("macro_rmse")
            if d_mean:
                st.markdown(f"""<div style="font-size:0.77rem;color:{COLORS['subtext']};margin:0.3rem 0 1rem">
                    Paired city-bootstrap 95% CI of (selected ML − mean baseline) macro RMSE:
                    <b style="color:{COLORS['warning']}">[{d_mean[0]:+.2f}, {d_mean[1]:+.2f}] °C</b>
                    — an interval that includes 0 means no demonstrated improvement over predicting
                    the training mean. Descriptive only; {ds['n_cities']} cities are few and not a
                    random sample.</div>""", unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        intervals = ci_blk.get("intervals", {})
        for col_box, metric_key, title in [(cc1, "macro_rmse", "Macro RMSE (per-city mean)"),
                                           (cc2, "pooled_rmse", "Pooled RMSE (all observations)")]:
            with col_box:
                names = [r["Model"] for r in rows]
                vals  = [ev_m["models"][r["key"]][metric_key] for r in rows]
                errs  = [intervals.get(r["key"], {}).get(metric_key) for r in rows]
                fig = go.Figure(go.Bar(
                    x=vals, y=names, orientation="h",
                    marker_color=[COLORS["subtext"], COLORS["secondary"], COLORS["primary"]][:len(rows)],
                    error_x=dict(type="data", symmetric=False,
                                 array=[(e[1] - v) if e else 0 for e, v in zip(errs, vals)],
                                 arrayminus=[(v - e[0]) if e else 0 for e, v in zip(errs, vals)],
                                 color=COLORS["text"], thickness=1.2),
                    text=[f"{v:.3f}" for v in vals], textposition="outside",
                    textfont_color=COLORS["subtext"]))
                fig.update_layout(title=f"{title} · lower = better", xaxis_title="°C", **_PL)
                st.plotly_chart(fig, use_container_width=True)

        # ── Per-city results ─────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        sec("🏙️ Per-city held-out results",
            "Each row is one outer fold: the city was unseen during preprocessing, tuning and fitting")
        pc_df = per_city_frame(ev_m)
        st.dataframe(pc_df.style.format({c: "{:.3f}" for c in pc_df.columns
                                         if c.endswith(("MAE", "RMSE"))}),
                     use_container_width=True, hide_index=True, height=320)
        long = pc_df.melt(id_vars=["City"], value_vars=["Mean RMSE", "Lat+NDVI RMSE", "ML RMSE"],
                          var_name="Model", value_name="RMSE")
        fig_pc = px.bar(long, x="City", y="RMSE", color="Model", barmode="group",
                        color_discrete_sequence=[COLORS["subtext"], COLORS["secondary"], COLORS["primary"]])
        fig_pc.update_layout(title="Per-city RMSE on the held-out city", yaxis_title="°C", **_PL)
        st.plotly_chart(fig_pc, use_container_width=True)
        fam = ev_m["models"]["nested_selected_model"].get("family_selection_counts", {})
        if fam:
            st.markdown(f"""<div style="font-size:0.77rem;color:{COLORS['subtext']}">
                Algorithm chosen by inner CV across the {ds['n_cities']} outer folds:
                {' · '.join(f'<b>{k}</b> ×{v}' for k, v in fam.items())}</div>""",
                unsafe_allow_html=True)

        # ── Held-out residuals ───────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        sec("📐 " + residual_label("heldout_nested_cv"),
            "residual = observed UHI − predicted UHI, from out-of-fold predictions for the "
            "held-out city only")
        rc1, rc2 = st.columns(2)
        with rc1:
            res_model = st.selectbox("Evaluated model", list(MODEL_DISPLAY_NAMES),
                                     index=2, format_func=MODEL_DISPLAY_NAMES.get)
        with rc2:
            city_opts = ["All held-out cities"] + sorted(ev_p["city"].unique())
            res_city = st.selectbox("Held-out city", city_opts)
        sub = ev_p[ev_p["model_name"] == res_model]
        if res_city != "All held-out cities":
            sub = sub[sub["city"] == res_city]
        res = sub["residual"].to_numpy()
        r_mae  = float(np.mean(np.abs(res)))
        r_rmse = float(np.sqrt(np.mean(res ** 2)))
        st.markdown(f"""<div class="res-note" style="margin-bottom:0.6rem">
            {residual_label("heldout_nested_cv")} — model <b style="color:{COLORS['primary']}">
            {MODEL_DISPLAY_NAMES[res_model]}</b> · city <b>{res_city}</b> ·
            <b>{len(res):,}</b> valid observed targets · MAE <b style="color:{COLORS['primary']}">{r_mae:.3f} °C</b>
            · RMSE <b style="color:{COLORS['primary']}">{r_rmse:.3f} °C</b> · mean residual {res.mean():+.3f} °C.
            Observations within a city share 8-day LST composites and are not independent.</div>""",
            unsafe_allow_html=True)
        ra, rb, rc = st.columns(3)
        with ra:
            fig_res = go.Figure(go.Histogram(x=res, nbinsx=40, marker_color=COLORS["primary"],
                                             opacity=0.8, marker_line_color=COLORS["card"],
                                             marker_line_width=0.5))
            fig_res.add_vline(x=0, line_dash="dash", line_color=COLORS["warning"], line_width=1.5)
            fig_res.update_layout(title="Residual distribution", xaxis_title="Observed − Predicted (°C)",
                                  height=300, **_PL)
            st.plotly_chart(fig_res, use_container_width=True)
        with rb:
            fig_rvp = go.Figure(go.Scatter(
                x=sub["predicted_uhi"], y=res, mode="markers",
                marker=dict(color=res, colorscale=[[0, COLORS["danger"]], [0.5, COLORS["subtext"]],
                                                   [1, COLORS["accent"]]], cmid=0,
                            size=4, opacity=0.55),
                customdata=sub[["city"]].to_numpy(),
                hovertemplate="%{customdata[0]}<br>Pred: %{x:.2f}°C<br>Residual: %{y:.2f}°C<extra></extra>"))
            fig_rvp.add_hline(y=0, line_dash="dash", line_color=COLORS["warning"], line_width=1.5)
            fig_rvp.update_layout(title="Residual vs predicted", xaxis_title="Predicted UHI (°C)",
                                  yaxis_title="Residual (°C)", height=300, **_PL)
            st.plotly_chart(fig_rvp, use_container_width=True)
        with rc:
            lo_ax = float(min(sub["observed_uhi"].min(), sub["predicted_uhi"].min())) - 0.5
            hi_ax = float(max(sub["observed_uhi"].max(), sub["predicted_uhi"].max())) + 0.5
            fig_pv = go.Figure(go.Scatter(
                x=sub["observed_uhi"], y=sub["predicted_uhi"], mode="markers",
                marker=dict(color=COLORS["secondary"], size=4, opacity=0.5),
                customdata=sub[["city"]].to_numpy(),
                hovertemplate="%{customdata[0]}<br>Observed: %{x:.2f}°C<br>Predicted: %{y:.2f}°C<extra></extra>"))
            fig_pv.add_trace(go.Scatter(x=[lo_ax, hi_ax], y=[lo_ax, hi_ax], mode="lines",
                                        line=dict(dash="dash", color=COLORS["warning"]),
                                        showlegend=False))
            fig_pv.update_layout(title="Held-out predicted vs observed",
                                 xaxis_title="Observed UHI (°C)", yaxis_title="Predicted UHI (°C)",
                                 height=300, showlegend=False, **_PL)
            st.plotly_chart(fig_pv, use_container_width=True)

        # ── Deployment model (separate from evaluation) ──────────────────────
        dep = ev_m.get("deployment_model")
        if dep:
            st.markdown("<hr>", unsafe_allow_html=True)
            sec("🚀 Deployment model (used by the Prediction tab)",
                "Selected separately by grouped CV on all cities — its selection score is not a test score")
            st.markdown(f"""<div class="glass" style="font-size:0.8rem;line-height:1.7">
                <b style="color:{COLORS['primary']}">{dep['selected_family']}</b> {dep['selected_params']} ·
                trained on {dep['n_train_observations']:,} observations from {dep['n_train_cities']} cities<br>
                Selection-CV RMSE {dep['selection_cv_rmse']:.3f} °C —
                <span style="color:{COLORS['warning']}">{dep['selection_cv_rmse_note']}</span><br>
                Retained features after fold-local constant/duplicate removal:
                <span class="mono">{', '.join(dep['retained_features'])}</span></div>""",
                unsafe_allow_html=True)
            fi = dep.get("feature_importance", {})
            if fi:
                fi_df = (pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                         .sort_values("Importance", ascending=True).tail(15))
                figFI = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                                         marker_color=COLORS["secondary"],
                                         text=fi_df["Importance"].map(lambda v: f"{v:.3f}"),
                                         textposition="outside", textfont_color=COLORS["subtext"]))
                figFI.update_layout(title=f"Normalised |coefficient| / importance — {dep['selected_family']} "
                                          "(standardised inputs)", **_PL)
                st.plotly_chart(figFI, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ✦ TAB 5 — HEATMAP  ✦
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🗺️  Heatmap":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    sec("🗺️ Global UHI Heatmap", "Mean observed (signed) UHI per city across the 180-day window · blue = urban cool island")

    if proc_df is None:
        st.warning("Run `python main.py` first.")
    else:
        df_map = proc_df.copy()
        if raw_df is not None and "name" in raw_df.columns:
            ci = raw_df[["lat","lon","name"]].drop_duplicates(subset=["lat","lon"])
            df_map = df_map.merge(ci,on=["lat","lon"],how="left")
        elif "city_name" in df_map.columns:
            df_map = df_map.rename(columns={"city_name":"name"})

        if "uhi_intensity" in df_map.columns:
            gcols = (["name","lat","lon"] if "name" in df_map.columns else ["lat","lon"])
            df_agg = (df_map[gcols+["uhi_intensity"]]
                      .groupby(gcols).mean().reset_index()
                      .dropna(subset=["uhi_intensity"]))   # cities with no observed LST
            df_agg["|UHI|"] = df_agg["uhi_intensity"].abs() + 0.2  # marker size; colour keeps sign

            ctrl, map_area = st.columns([1,4])
            with ctrl:
                mn,mx = float(df_agg["uhi_intensity"].min()), float(df_agg["uhi_intensity"].max())
                uhi_rng = st.slider("UHI range (°C)",mn,mx,(mn,mx),step=0.1)
                mtype   = st.selectbox("Style",["Scatter","Density"])
                mtheme  = st.selectbox("Base map",
                                       ["carto-darkmatter","carto-positron","open-street-map"])
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""<div class="glass" style="font-size:0.78rem">
                    <div style="color:{COLORS['subtext']};font-size:0.65rem;font-weight:700;
                                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">
                        Summary
                    </div>
                    <div style="line-height:2">
                        <span style="color:{COLORS['subtext']}">Cities: </span>
                        <b style="color:{COLORS['text']}">{len(df_agg)}</b><br>
                        <span style="color:{COLORS['subtext']}">Min: </span>
                        <b style="color:{COLORS['accent']}">{mn:.2f}°C</b><br>
                        <span style="color:{COLORS['subtext']}">Max: </span>
                        <b style="color:{COLORS['danger']}">{mx:.2f}°C</b><br>
                        <span style="color:{COLORS['subtext']}">Mean: </span>
                        <b style="color:{COLORS['warning']}">{df_agg['uhi_intensity'].mean():.2f}°C</b>
                    </div></div>""", unsafe_allow_html=True)

            df_filt = df_agg[df_agg["uhi_intensity"].between(uhi_rng[0],uhi_rng[1])]
            with map_area:
                cs = [[0,COLORS["primary"]],[0.5,"#e6edf3"],[1,COLORS["danger"]]]  # diverging at 0 °C
                if mtype == "Scatter":
                    figM = px.scatter_mapbox(
                        df_filt, lat="lat", lon="lon",
                        color="uhi_intensity", size="|UHI|",
                        color_continuous_midpoint=0,
                        hover_name="name" if "name" in df_filt.columns else None,
                        hover_data={"uhi_intensity":":.2f","lat":False,"lon":False,"|UHI|":False},
                        color_continuous_scale=cs, size_max=32,
                        zoom=1.1, mapbox_style=mtheme)
                else:
                    figM = px.density_mapbox(
                        df_filt, lat="lat", lon="lon", z="uhi_intensity",
                        radius=55, zoom=1.1, mapbox_style=mtheme,
                        color_continuous_scale=[[0,"rgba(63,185,80,0)"],[0.5,COLORS["warning"]],[1,COLORS["danger"]]])
                figM.update_layout(
                    height=500, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor=COLORS["card"],
                    coloraxis_colorbar=dict(
                        title=dict(text="°C",font_color=COLORS["subtext"]),
                        tickcolor=COLORS["subtext"],tickfont_color=COLORS["subtext"],
                        bgcolor=COLORS["card"],outlinecolor=COLORS["border"]),
                    font_color=COLORS["text"])
                st.plotly_chart(figM, use_container_width=True)

            # Top/bottom
            if "name" in df_agg.columns:
                ct, cb2 = st.columns(2)
                for col_w, ascending, title, c in [
                    (ct,  False, "🔴 Hottest 10",  [[0,COLORS["warning"]],[1,COLORS["danger"]]]),
                    (cb2, True,  "🟢 Coolest 10", [[0,COLORS["accent"]],[1,COLORS["primary"]]]),
                ]:
                    sub=df_agg.sort_values("uhi_intensity",ascending=ascending).head(10)
                    fig=go.Figure(go.Bar(
                        x=sub["uhi_intensity"],y=sub["name"],orientation="h",
                        marker=dict(color=sub["uhi_intensity"],colorscale=c,showscale=False),
                        text=sub["uhi_intensity"].round(2),textposition="outside",
                        textfont_color=COLORS["subtext"]))
                    fig.update_layout(title=title,**_PL)
                    col_w.plotly_chart(fig, use_container_width=True)

            # ── Seasonal city × month heatmap ───────────────────────────────
            if "city_name" in proc_df.columns and "month" in proc_df.columns:
                st.markdown("<hr>", unsafe_allow_html=True)
                sec("📅 Seasonal UHI Heatmap  (City × Month)",
                    "Mean UHI per city for each month of the year")
                pivot = (proc_df.groupby(["city_name","month"])["uhi_intensity"]
                         .mean().unstack(fill_value=0))
                pivot.columns = [MONTH_NAMES.get(int(c),c) for c in pivot.columns]
                fig_sh = go.Figure(go.Heatmap(
                    z=pivot.values, x=pivot.columns, y=pivot.index,
                    colorscale=[[0,"#0d2b1a"],[0.3,COLORS["accent"]],
                                [0.6,COLORS["warning"]],[1,COLORS["danger"]]],
                    text=np.round(pivot.values,1),
                    texttemplate="%{text}",textfont_size=8,
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}°C<extra></extra>",
                    colorbar=dict(title=dict(text="°C",font_color=COLORS["subtext"]),
                                  tickcolor=COLORS["subtext"],
                                  tickfont_color=COLORS["subtext"],
                                  bgcolor=COLORS["card"],
                                  outlinecolor=COLORS["border"]),
                ))
                fig_sh.update_layout(
                    title="Mean UHI (°C) — city × month",
                    height=max(320, len(pivot)*20+80),
                    **_PL)
                st.plotly_chart(fig_sh, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ✦ TAB 6 — PREDICTION  ✦
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔮  Prediction":
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    sec("🔮 Live UHI Prediction",
        "Deployment pipeline (raw features in, its own fitted preprocessing) · predictions "
        "can be negative (urban cool island)")

    if model is None or scaler is None or feat_names is None:
        st.error("No trained model found. Run `python main.py` first.")
    else:
        inp, out = st.columns([1.6, 1], gap="large")

        with inp:
            def group_label(icon, title):
                st.markdown(f"""<div style="font-size:0.68rem;font-weight:800;
                    color:{COLORS['subtext']};text-transform:uppercase;
                    letter-spacing:0.12em;margin:0.8rem 0 0.5rem">
                    {icon} {title}</div>""", unsafe_allow_html=True)

            group_label("🌡️","Weather Conditions")
            wa,wb = st.columns(2)
            with wa:
                temperature = st.slider("Temperature (°C)",  -10.0, 50.0, 32.0, 0.5)
                wind_speed  = st.slider("Wind Speed (m/s)",    0.0, 20.0,  2.5, 0.1)
                pressure    = st.slider("Pressure (hPa)",    950.0,1050.0,1013.0,0.5)
            with wb:
                humidity = st.slider("Humidity (%)",   0, 100, 65, 1)
                clouds   = st.slider("Cloud Cover (%)",0, 100, 30, 1)

            group_label("🌿","Land Cover")
            lc_a,lc_b = st.columns(2)
            with lc_a: ndvi       = st.slider("NDVI",          0.0,1.0,0.25,0.01)
            with lc_b: urban_frac = st.slider("Urban Fraction",0.0,1.0,0.70,0.01)
            vl = "Sparse (0)" if ndvi<0.2 else ("Dense (2)" if ndvi>=0.5 else "Moderate (1)")
            st.markdown(f"""<div style="font-size:0.72rem;color:{COLORS['subtext']};
                margin-top:-0.3rem">veg_class → <span class="mono">{vl}</span>
                &nbsp;·&nbsp; heat_retention →
                <span class="mono">{urban_frac*temperature/(wind_speed+1):.2f}</span></div>""",
                unsafe_allow_html=True)

            group_label("📍","Location & Time")
            la, lb = st.columns(2)
            with la:
                lat   = st.slider("Latitude",  -90.0, 90.0, 28.6, 0.1)
                hour  = st.slider("Hour",       0, 23, 14, 1)
            with lb:
                lon   = st.slider("Longitude",-180.0,180.0, 77.2, 0.1)
                month = st.slider("Month",      1, 12,  6,  1)

            # City preset
            st.markdown("<br>", unsafe_allow_html=True)
            preset = st.selectbox("⚡ Quick-load city",
                                   ["Custom"] + [c["name"] for c in CITIES])
            if preset != "Custom":
                cd   = next(c for c in CITIES if c["name"]==preset)
                lat  = cd["lat"]; lon = cd["lon"]
                st.markdown(f"""<div style="font-size:0.72rem;color:{COLORS['accent']}">
                    ✓ Loaded <b>{preset}</b>: lat={lat}, lon={lon}</div>""",
                    unsafe_allow_html=True)

        with out:
            try:
                vec  = build_input(temperature, humidity, wind_speed,
                                   ndvi, urban_frac, lat, lon,
                                   hour, month, pressure, clouds, feat_names)
                pred = float(predict_uhi([vec])[0])

                if pred < 0:   sev,sc,sbg = "Cool-island",COLORS["primary"],"rgba(88,166,255,0.1)"
                elif pred < 1: sev,sc,sbg = "Low",     COLORS["accent"],  "rgba(63,185,80,0.1)"
                elif pred < 2: sev,sc,sbg = "Moderate",COLORS["warning"], "rgba(210,153,34,0.1)"
                elif pred < 4: sev,sc,sbg = "High",    COLORS["danger"],  "rgba(248,81,73,0.1)"
                else:          sev,sc,sbg = "Extreme", "#ff6b6b",         "rgba(255,107,107,0.12)"

                st.markdown(f"""
                <div class="pred-box">
                    <div style="font-size:0.68rem;color:{COLORS['subtext']};font-weight:700;
                                text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.5rem">
                        Predicted UHI Intensity
                    </div>
                    <div class="pred-val">{pred:.2f} °C</div>
                    <div>
                        <span class="sev-badge"
                              style="background:{sbg};border-color:{sc}44;color:{sc}">
                            {sev} UHI
                        </span>
                    </div>
                    <div style="margin-top:1rem;font-size:0.68rem;color:{COLORS['subtext']}">
                        Model: <b style="color:{COLORS['primary']}">{(metrics or {}).get('deployment_model',{}).get('selected_family','—')}</b>
                        &nbsp;·&nbsp; typical error on an unseen city (nested LOCO, macro RMSE):
                        <b style="color:{COLORS['primary']}">{f"{sel_metric(metrics,'macro_rmse'):.2f}" if metrics else '—'}°C</b>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Gauge
                figG = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    number={"suffix":" °C","font":{"size":32,"color":COLORS["text"]}},
                    gauge={
                        "axis":{"range":[min(-4.0, pred),max(8.0, pred)],"tickwidth":1,
                                "tickcolor":COLORS["subtext"],"nticks":9},
                        "bar":{"color":sc,"thickness":0.25},
                        "bgcolor":COLORS["card2"],
                        "bordercolor":COLORS["border"],"borderwidth":1,
                        "steps":[
                            {"range":[min(-4.0, pred),0],"color":"rgba(88,166,255,0.18)"},
                            {"range":[0,1],"color":"rgba(63,185,80,0.18)"},
                            {"range":[1,2],"color":"rgba(210,153,34,0.18)"},
                            {"range":[2,4],"color":"rgba(248,81,73,0.18)"},
                            {"range":[4,8],"color":"rgba(255,107,107,0.22)"},
                        ],
                        "threshold":{"line":{"color":COLORS["text"],"width":2},
                                     "thickness":0.8,"value":pred},
                    },
                    title={"text":"UHI Severity","font":{"color":COLORS["subtext"],"size":12}},
                ))
                figG.update_layout(height=255,paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color=COLORS["text"],family="Inter"),
                                   margin=dict(l=18,r=18,t=28,b=8))
                st.plotly_chart(figG, use_container_width=True)

                # Radar driving factors
                st.markdown(f'<div class="sec-head" style="font-size:1rem;margin-top:0.5rem">Key Drivers</div>',
                            unsafe_allow_html=True)
                fac = {
                    "Urban Fraction":  urban_frac,
                    "Temperature":     min(1.0,max(0.0,(temperature+10)/60)),
                    "Low Vegetation":  1-ndvi,
                    "High Humidity":   humidity/100,
                    "Low Wind":        1-min(1.0,wind_speed/15),
                    "No Clouds":       1-clouds/100,
                }
                figR2 = go.Figure(go.Scatterpolar(
                    r=list(fac.values())+[list(fac.values())[0]],
                    theta=list(fac.keys())+[list(fac.keys())[0]],
                    fill="toself",
                    fillcolor="rgba(88,166,255,0.1)",
                    line=dict(color=COLORS["primary"],width=2.5),
                    marker=dict(color=COLORS["primary"],size=7),
                ))
                figR2.update_layout(
                    polar=dict(
                        bgcolor=COLORS["card2"],
                        radialaxis=dict(visible=True,range=[0,1],
                                        tickcolor=COLORS["subtext"],
                                        gridcolor="#1c2230",linecolor="#1c2230"),
                        angularaxis=dict(tickcolor=COLORS["subtext"],
                                         gridcolor="#1c2230",linecolor="#1c2230")),
                    showlegend=False, height=290,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text"],family="Inter",size=10),
                    margin=dict(l=28,r=28,t=16,b=16))
                st.plotly_chart(figR2, use_container_width=True)

                # ── Feature contributions (scaled values) ──────────────────
                st.markdown(f'<div class="sec-head" style="font-size:1rem;margin-top:0.6rem">'
                            f'Standardised Feature Inputs (display only)</div>', unsafe_allow_html=True)
                scaled_vec = scaler.transform(np.atleast_2d(np.asarray(vec, dtype=float)))[0]
                render_feature_contributions(scaled_vec, feat_names)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        # ── Sensitivity analysis ──────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        sec("📐 Sensitivity Analysis",
            "How predicted UHI changes as one parameter varies — all others fixed at slider values")

        PARAM_MAP = {
            "temperature":"temp","humidity":"hum","wind_speed":"wind",
            "ndvi":"ndvi_val","urban_fraction":"uf",
            "pressure":"pressure","clouds":"clouds",
        }
        PARAM_RANGES = {
            "temperature":(-10.0,50.0),"urban_fraction":(0.0,1.0),
            "ndvi":(0.0,1.0),"humidity":(10.0,100.0),"wind_speed":(0.0,20.0),
            "pressure":(950.0,1050.0),"clouds":(0.0,100.0),
        }
        sc1,sc2 = st.columns([1,4])
        with sc1:
            vary = st.selectbox("Vary",list(PARAM_RANGES.keys()))
        if feat_names:
            lo,hi = PARAM_RANGES[vary]
            pr    = np.linspace(lo,hi,60)
            base  = dict(temp=temperature,hum=humidity,wind=wind_speed,
                         ndvi_val=ndvi,uf=urban_frac,lat=lat,lon=lon,
                         hour=hour,month=month,pressure=pressure,clouds=clouds)
            bkey  = PARAM_MAP[vary]
            psa   = predict_uhi([build_input(**{**base,bkey:v},feat_names=feat_names)
                                 for v in pr]).tolist()

            cur = base.get(bkey)
            figSA = go.Figure()
            # Fill area coloured by UHI level
            figSA.add_trace(go.Scatter(
                x=pr,y=psa,mode="lines",
                line=dict(color=COLORS["primary"],width=2.5),
                fillcolor="rgba(88,166,255,0.07)",
                hovertemplate=f"<b>{vary}</b>: %{{x:.2f}}<br>UHI: %{{y:.3f}}°C<extra></extra>",
            ))
            if cur is not None and lo<=cur<=hi:
                cur_p = float(predict_uhi([build_input(**{**base,bkey:cur},feat_names=feat_names)])[0])
                figSA.add_vline(x=cur,line_dash="dash",
                               line_color=COLORS["secondary"],line_width=2,
                               annotation_text=f"current {cur:.2f}",
                               annotation_font_color=COLORS["secondary"])
                figSA.add_trace(go.Scatter(
                    x=[cur],y=[cur_p],mode="markers",showlegend=False,
                    marker=dict(color=COLORS["secondary"],size=11,
                                line=dict(color=COLORS["background"],width=2)),
                    hovertemplate=f"Current: {cur:.2f}<br>UHI: {cur_p:.3f}°C<extra></extra>",
                ))
            figSA.update_layout(
                xaxis_title=vary.replace("_"," ").title(),
                yaxis_title="Predicted UHI (°C)",
                title=f"UHI Sensitivity — {vary.replace('_',' ').title()}",**_PL)
            with sc2:
                st.plotly_chart(figSA, use_container_width=True)

        # ── 3D Response Surface ───────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        sec("🌐 3D Response Surface",
            "How UHI changes with Temperature × Urban Fraction — all other parameters fixed at slider values")
        if feat_names:
            try:
                t_range  = np.linspace(-5.0, 50.0, 35)
                uf_range = np.linspace(0.0,  1.0,  35)
                T_grid, UF_grid = np.meshgrid(t_range, uf_range)
                base3d = dict(hum=humidity, wind=wind_speed, ndvi_val=ndvi,
                              lat=lat, lon=lon, hour=hour, month=month,
                              pressure=pressure, clouds=clouds)
                Z_grid = predict_uhi([
                    build_input(temp=float(t), uf=float(u), **base3d, feat_names=feat_names)
                    for t, u in zip(T_grid.ravel(), UF_grid.ravel())
                ]).reshape(T_grid.shape)

                fig3d = go.Figure(go.Surface(
                    x=T_grid, y=UF_grid, z=Z_grid,
                    colorscale=[[0,"#0d2b1a"],[0.25,COLORS["accent"]],
                                [0.55,COLORS["warning"]],[1,COLORS["danger"]]],
                    opacity=0.92,
                    contours=dict(
                        z=dict(show=True, usecolormap=True, highlightcolor="#e6edf3",
                               project_z=True, width=1),
                    ),
                    hovertemplate="Temp: %{x:.1f}°C<br>Urban Frac: %{y:.2f}<br>UHI: %{z:.2f}°C<extra></extra>",
                ))
                # Mark current slider position
                cur_z = float(predict_uhi([build_input(temp=temperature, uf=urban_frac,
                                                       **base3d, feat_names=feat_names)])[0])
                fig3d.add_trace(go.Scatter3d(
                    x=[temperature], y=[urban_frac], z=[cur_z+0.05],
                    mode="markers+text",
                    marker=dict(color=COLORS["secondary"], size=8,
                                symbol="circle",
                                line=dict(color=COLORS["background"], width=2)),
                    text=["← You"], textfont=dict(color=COLORS["secondary"], size=11),
                    showlegend=False,
                    hovertemplate=f"Your point<br>Temp: {temperature:.1f}°C<br>"
                                  f"Urban: {urban_frac:.2f}<br>UHI: {cur_z:.2f}°C<extra></extra>",
                ))
                fig3d.update_layout(
                    scene=dict(
                        xaxis=dict(title="Temperature (°C)", gridcolor="#1c2230",
                                   backgroundcolor=COLORS["card"], color=COLORS["subtext"]),
                        yaxis=dict(title="Urban Fraction", gridcolor="#1c2230",
                                   backgroundcolor=COLORS["card"], color=COLORS["subtext"]),
                        zaxis=dict(title="UHI (°C)", gridcolor="#1c2230",
                                   backgroundcolor=COLORS["card"], color=COLORS["subtext"]),
                        bgcolor=COLORS["card"],
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=COLORS["text"], family="Inter"),
                    margin=dict(l=0, r=0, t=36, b=0),
                    height=500,
                    title=dict(text="UHI = f(Temperature, Urban Fraction)",
                               font=dict(color=COLORS["text"], size=13)),
                )
                st.markdown('<div class="chart-3d-wrap">', unsafe_allow_html=True)
                st.plotly_chart(fig3d, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.info(f"3D surface unavailable: {e}")

        # ── City similarity + mitigation ──────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        sim_col, mit_col = st.columns(2)

        with sim_col:
            sec("🔍 City Similarity",
                "Which training city does your input climate profile most resemble?")
            if proc_df is not None and "city_name" in proc_df.columns and feat_names:
                try:
                    city_profiles = (proc_df.groupby("city_name")
                                     [[f for f in feat_names if f in proc_df.columns]]
                                     .mean()
                                     .reindex(columns=feat_names))
                    _has_obs = proc_df.groupby("city_name")["uhi_intensity"].count()
                    city_profiles = city_profiles.loc[_has_obs.loc[city_profiles.index] > 0]
                    # Scale to match user vector
                    user_scaled = np.array(scaler.transform(np.atleast_2d(np.asarray(vec, dtype=float)))[0])
                    city_scaled = scaler.transform(city_profiles.to_numpy(float))
                    distances   = np.linalg.norm(city_scaled - user_scaled, axis=1)
                    top5_idx    = distances.argsort()[:5]
                    cities_top5 = city_profiles.index[top5_idx]
                    dists_top5  = distances[top5_idx]
                    max_d       = dists_top5.max() or 1.0

                    pills = "".join(
                        f'<span class="city-pill">'
                        f'{CITY_EMOJI.get(c,"🏙")} {c}'
                        f'<span style="color:{COLORS["subtext"]};font-size:0.62rem;margin-left:0.3rem">'
                        f'{100*(1-d/max_d):.0f}%</span>'
                        f'</span>'
                        for c, d in zip(cities_top5, dists_top5)
                    )
                    st.markdown(
                        f'<div class="neon-card" style="padding:1rem 1.2rem">'
                        f'<div style="font-size:0.62rem;font-weight:800;text-transform:uppercase;'
                        f'letter-spacing:0.12em;color:{COLORS["subtext"]};margin-bottom:0.7rem">'
                        f'Most similar cities (feature-space distance)</div>'
                        f'<div style="display:flex;flex-wrap:wrap">{pills}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Show closest city UHI vs prediction
                    closest = cities_top5[0]
                    city_uhi = proc_df[proc_df["city_name"]==closest]["uhi_intensity"].mean()
                    delta_v  = pred - city_uhi
                    arrow    = "▲" if delta_v > 0 else "▼"
                    delta_c  = COLORS["danger"] if delta_v > 0 else COLORS["accent"]
                    st.markdown(f"""
                    <div class="insight-card" style="--ic:{COLORS['primary']};margin-top:0.6rem">
                        Closest match: <strong>{CITY_EMOJI.get(closest,"🏙")} {closest}</strong><br>
                        Mean observed UHI there: <strong>{city_uhi:.2f}°C</strong> ·
                        Your prediction: <strong>{pred:.2f}°C</strong>
                        <span style="color:{delta_c};font-weight:700"> {arrow} {abs(delta_v):.2f}°C
                        {"hotter" if delta_v>0 else "cooler"}</span>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.info(f"City similarity unavailable: {e}")

        with mit_col:
            sec("💡 Mitigation Insights",
                "Evidence-based strategies to reduce UHI at the predicted intensity level")
            mit_items = []
            if urban_frac > 0.6:
                mit_items.append(("🌳", "Increase urban greenery",
                    f"Urban fraction is {urban_frac:.0%}. Adding green roofs or parks "
                    f"could reduce NDVI gap and lower surface temperature by 1–3°C."))
            if ndvi < 0.25:
                mit_items.append(("🌿", "Vegetation cover",
                    f"NDVI of {ndvi:.2f} indicates sparse vegetation. "
                    f"Street trees and green corridors significantly reduce LST."))
            if wind_speed < 2.0:
                mit_items.append(("💨", "Improve ventilation corridors",
                    f"Wind speed {wind_speed:.1f} m/s is low. Urban canyon orientation "
                    f"and reduced building density improve air circulation."))
            if temperature > 35:
                mit_items.append(("🔆", "Cool pavements & roofs",
                    f"At {temperature:.0f}°C, high-albedo surfaces can reflect solar "
                    f"radiation and reduce ambient temperature by 0.5–2°C."))
            if clouds < 20:
                mit_items.append(("☁️", "Shading structures",
                    f"Cloud cover only {clouds}%. Cool corridors, pergolas, and "
                    f"reflective materials provide relief when natural shading is absent."))
            if humidity > 80:
                mit_items.append(("💧", "Reduce anthropogenic heat",
                    f"High humidity ({humidity}%) amplifies heat stress. Reducing "
                    f"AC exhaust, vehicle emissions, and industrial heat helps."))
            if not mit_items:
                mit_items.append(("✅", "Conditions are relatively favourable",
                    "Your parameter combination shows moderate UHI risk. "
                    "Maintain current green coverage and ventilation."))

            for icon, title, desc in mit_items[:4]:
                c_ic = COLORS["accent"] if icon in ("🌳","🌿","✅") else COLORS["warning"]
                st.markdown(f"""
                <div class="insight-card" style="--ic:{c_ic}">
                    <strong>{icon} {title}</strong><br>
                    <span style="font-size:0.73rem;color:{COLORS['subtext']}">{desc}</span>
                </div>""", unsafe_allow_html=True)

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


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║       🌡️ Urban Heat Island Prediction System            ║
║       ML Pipeline  •  Streamlit Dashboard                ║
╚══════════════════════════════════════════════════════════╝
    """)


def step_collect():
    log.info("──────────────────────────────────────────")
    log.info("STEP 1/4 – Data Collection")
    log.info("──────────────────────────────────────────")

    from data_collector import collect_data

    log.info("  🔄 Fetching fresh data from API...")

    try:
        collect_data(force=True)
    except TypeError:
        log.warning("  ⚠️ 'force' not supported, running default collect_data()")
        collect_data()


def step_preprocess():
    log.info("──────────────────────────────────────────")
    log.info("STEP 2/4 – Preprocessing")
    log.info("──────────────────────────────────────────")

    if PROCESSED_DATA_PATH.exists():
        log.info("  ✓ Processed data exists. Skipping.")
        return

    from preprocessor import preprocess
    preprocess()


def step_train():
    log.info("──────────────────────────────────────────")
    log.info("STEP 3/4 – Model Training")
    log.info("──────────────────────────────────────────")

    if BEST_MODEL_PATH.exists() and METRICS_PATH.exists():
        log.info("  ✓ Trained model exists. Skipping.")
        return

    from model_trainer import train
    metrics = train()
    log.info(f"  🏆 Best: {metrics['best_model']}  RMSE={metrics['best_rmse']}")


def step_dashboard():
    log.info("──────────────────────────────────────────")
    log.info("STEP 4/4 – Launching Dashboard")
    log.info("──────────────────────────────────────────")

    dashboard_path = Path("dashboard.py")

    if not dashboard_path.exists():
        log.error("  ❌ dashboard.py not found!")
        return

    log.info(f"  🚀 Starting Streamlit on http://localhost:{DASHBOARD_PORT}")
    log.info("  Press Ctrl+C to stop.\n")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", str(DASHBOARD_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.backgroundColor", "#F7F9FC",
            "--theme.primaryColor", "#A8C5DA",
            "--theme.textColor", "#2E2E2E",
        ], check=True)
    except subprocess.CalledProcessError:
        log.error("  ❌ Failed to start Streamlit. Is it installed?")
        log.info("  👉 Run: pip install streamlit")


if __name__ == "__main__":
    banner()
    try:
        step_collect()
        step_preprocess()
        step_train()
        step_dashboard()
    except KeyboardInterrupt:
        log.info("\n  Interrupted by user. Goodbye!")
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
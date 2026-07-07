"""Streamlit Cloud entrypoint.

Deploy this file on Streamlit Cloud. The local full pipeline still runs from
`python main.py`.
"""

import importlib
import sys

try:
    if "dashboard" in sys.modules:
        importlib.reload(sys.modules["dashboard"])
    else:
        import dashboard  # noqa: F401
except Exception as exc:
    import traceback

    import streamlit as st

    try:
        st.set_page_config(
            page_title="HeatFluxAI startup error",
            page_icon="!",
            layout="wide",
        )
    except Exception:
        pass
    st.error("HeatFluxAI could not finish starting.")
    st.exception(exc)
    st.code(traceback.format_exc(), language="python")

"""Headless render of every dashboard tab (streamlit.testing AppTest).

Requires the committed evaluation artifacts; the Prediction tab additionally
needs models/best_model.pkl (created by scripts/evaluate.py) and is skipped
otherwise.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

NAV = ["📊  Overview", "📁  Data Explorer", "🔬  Preprocessing",
       "🤖  Models", "🗺️  Heatmap", "🔮  Prediction"]

pytestmark = pytest.mark.skipif(not (ROOT / "evaluation" / "metrics.json").exists(),
                                reason="evaluation artifacts not generated")


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [w.value for w in at.warning] + \
            [e.value for e in at.error] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("tab", NAV)
def test_tab_renders_without_exceptions(tab, monkeypatch):
    if tab == NAV[-1] and not (ROOT / "models" / "best_model.pkl").exists():
        pytest.skip("deployment model not generated")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(tab).run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "Test residuals" not in text
    if tab == "🤖  Models":
        assert "Held-Out City Residuals (Nested Cross-Validation)" in text
        assert "No evaluation found" not in text
    if tab == "📊  Overview":
        assert "Held-out" in text

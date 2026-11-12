"""Smoke test: the dashboard renders without exceptions (trains FD001 models, about 15 s)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("matplotlib")

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_fd001():
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py"), default_timeout=240).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Engines monitored" in labels

"""API tests (train FD001 on first use, about 15 s)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from api.main import app
from src.data import load_subset

client = TestClient(app)


def _engine_rows(unit=1, n=60):
    _, test, _ = load_subset("FD001")
    g = test[test["unit"] == unit].head(n).drop(columns="unit")
    return g.to_dict(orient="records")


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_ordered_interval_and_status():
    r = client.post("/predict", json={"subset": "FD001", "readings": _engine_rows()})
    assert r.status_code == 200
    d = r.json()
    assert d["rul_q10"] <= d["rul_median"] <= d["rul_q90"]
    assert d["status"] in {"ALERT", "WATCH", "OK"}


def test_missing_columns_rejected():
    r = client.post("/predict", json={"subset": "FD001", "readings": [{"cycle": 1.0}]})
    assert r.status_code == 422


def test_bad_subset_rejected():
    r = client.post("/predict", json={"subset": "FD999", "readings": _engine_rows(n=5)})
    assert r.status_code == 422

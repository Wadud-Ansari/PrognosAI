"""PrognosAI prediction API.

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.alerts import choose_threshold
from src.baseline import window_features
from src.data import COLS, make_windows
from src.quantile import predict_q

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
WATCH_MARGIN = 20
REQUIRED = [c for c in COLS if c != "unit"]

app = FastAPI(
    title="PrognosAI",
    version="1.0",
    description="Remaining-useful-life prediction with 80% intervals and cost-based maintenance alerts "
    "for turbofan engines (NASA C-MAPSS, simulated data).",
)


class PredictRequest(BaseModel):
    subset: str = Field("FD001", description="C-MAPSS subset the engine belongs to: FD001 to FD004")
    readings: List[Dict[str, float]] = Field(
        ...,
        description="One engine's history, one object per cycle, oldest first. Each needs: "
        "cycle, op1, op2, op3, s1 ... s21. At least 30 cycles gives the most reliable prediction.",
    )


class PredictResponse(BaseModel):
    subset: str
    cycles_used: int
    rul_q10: float
    rul_median: float
    rul_q90: float
    alert_threshold: float
    status: str
    note: str = ""


@lru_cache(maxsize=4)
def get_bundle(subset: str):
    path = MODEL_DIR / f"{subset}.joblib"
    if path.exists():
        b = joblib.load(path)
    else:  # fall back to training on first use (about 15 s)
        from src.pipeline import fit_subset

        b = fit_subset(subset)
    b["threshold"] = choose_threshold(b["val_pred"][:, 0], b["val_true"])
    return b


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/subsets")
def subsets():
    return {"subsets": list(SUBSETS), "required_columns": REQUIRED}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.subset not in SUBSETS:
        raise HTTPException(422, f"subset must be one of {SUBSETS}")
    if not req.readings:
        raise HTTPException(422, "readings must contain at least one cycle")
    df = pd.DataFrame(req.readings)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(422, f"missing columns: {missing}")

    b = get_bundle(req.subset)
    df = df.sort_values("cycle").assign(unit=1)[COLS]
    Xw, _, _ = make_windows(b["pre"].transform(df), b["sensors"], b["window"], last_only=True)
    lo, med, hi = predict_q(b["models"], window_features(Xw))[0]

    T = b["threshold"]
    status = "ALERT" if lo <= T else ("WATCH" if lo <= T + WATCH_MARGIN else "OK")
    note = ""
    if len(df) < b["window"]:
        note = f"Only {len(df)} cycles supplied; history was padded to {b['window']}. Treat the estimate with caution."
    return PredictResponse(
        subset=req.subset, cycles_used=len(df), rul_q10=round(float(lo), 1), rul_median=round(float(med), 1),
        rul_q90=round(float(hi), 1), alert_threshold=T, status=status, note=note,
    )

"""Train and persist quantile models for serving.

Usage: python -m src.train_final [FD001 FD002 ...]   (default: all four subsets)
Writes models/<subset>.joblib, which api/main.py loads at startup.
"""
import sys
from pathlib import Path

import joblib

from src.pipeline import fit_subset

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
KEEP = ("fd", "pre", "models", "sensors", "window", "val_pred", "val_true")


def train_and_save(fd):
    b = fit_subset(fd)
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{fd}.joblib"
    joblib.dump({k: b[k] for k in KEEP}, path, compress=3)
    return path


if __name__ == "__main__":
    subsets = sys.argv[1:] or ["FD001", "FD002", "FD003", "FD004"]
    for fd in subsets:
        print("saved", train_and_save(fd))

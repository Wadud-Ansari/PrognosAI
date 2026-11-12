"""Shared fit/predict helpers used by the dashboard and explainability scripts."""
import numpy as np
from src.data import load_subset, add_train_rul, Preprocessor, make_windows, engine_split, RUL_CAP
from src.baseline import window_features, CONDS
from src.quantile import fit_quantiles, predict_q

WINDOW = 30
STATS = ("last", "mean", "std", "slope")


def feature_names(sensors):
    return [f"{s}_{st}" for st in STATS for s in sensors]


def fit_subset(fd, window=WINDOW):
    """Train quantile models; return everything needed for monitoring."""
    train, test, rul_true = load_subset(fd)
    train = add_train_rul(train)
    pre = Preprocessor(CONDS[fd]).fit(train)
    train_n, test_n = pre.transform(train), pre.transform(test)
    X, y, units = make_windows(train_n, pre.sensors_, window, rul_col="rul")
    tr_ids, va_ids = engine_split(units)
    tr, va = np.isin(units, list(tr_ids)), np.isin(units, list(va_ids))
    F = window_features(X)
    models = fit_quantiles(F[tr], y[tr], F[va], y[va])
    return {"fd": fd, "pre": pre, "models": models, "test_raw": test, "test_n": test_n,
            "rul_true": rul_true, "sensors": pre.sensors_, "window": window,
            "val_pred": predict_q(models, F[va]), "val_true": y[va]}


def engine_curve(bundle, unit):
    """Predicted [q10, q50, q90] RUL for every cycle (from cycle=window on) of one test engine."""
    g = bundle["test_n"][bundle["test_n"]["unit"] == unit]
    Xw, _, _ = make_windows(g, bundle["sensors"], bundle["window"])
    n_pad = len(g) - len(Xw)
    cycles = g["cycle"].to_numpy()[-len(Xw):]
    return cycles, predict_q(bundle["models"], window_features(Xw))


def fleet_table(bundle):
    """Last-window prediction for each test engine."""
    t = bundle["test_n"]
    Xl, _, units = make_windows(t, bundle["sensors"], bundle["window"], last_only=True)
    P = predict_q(bundle["models"], window_features(Xl))
    return units, P

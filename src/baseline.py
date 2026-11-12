"""LightGBM baseline on rolling-window features. Usage: python -m src.baseline FD001"""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb
from src.data import load_subset, add_train_rul, Preprocessor, make_windows, engine_split, RUL_CAP
from src.metrics import rmse, nasa_score

CONDS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


def window_features(X):
    """X [n, w, f] -> last value, mean, std, slope per sensor."""
    w = X.shape[1]
    t = np.arange(w) - (w - 1) / 2
    slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.concatenate([X[:, -1], X.mean(1), X.std(1), slope], axis=1)


def main(fd="FD001", window=30):
    train, test, rul_true = load_subset(fd)
    train = add_train_rul(train)
    pre = Preprocessor(CONDS[fd]).fit(train)
    train_n, test_n = pre.transform(train), pre.transform(test)
    feats = pre.sensors_

    X, y, units = make_windows(train_n, feats, window, rul_col="rul")
    tr_ids, va_ids = engine_split(units)          # split by ENGINE
    tr, va = np.isin(units, list(tr_ids)), np.isin(units, list(va_ids))
    F = window_features(X)

    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, verbose=-1)
    model.fit(F[tr], y[tr], eval_set=[(F[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    Xt, _, _ = make_windows(test_n, feats, window, last_only=True)  # last window / engine
    pred = np.clip(model.predict(window_features(Xt)), 0, RUL_CAP)
    true = np.minimum(rul_true, RUL_CAP) if False else rul_true     # score vs raw RUL
    res = {"subset": fd, "n_sensors_kept": len(feats), "val_rmse": rmse(y[va], model.predict(F[va])),
           "test_rmse": rmse(true, pred), "test_nasa_score": nasa_score(true, pred)}
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")

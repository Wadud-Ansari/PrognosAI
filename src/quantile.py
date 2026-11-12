"""Quantile LightGBM -> RUL prediction intervals + cost-based alerts.
Usage: python -m src.quantile FD001
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, lightgbm as lgb
from src.data import load_subset, add_train_rul, Preprocessor, make_windows, engine_split, RUL_CAP
from src.baseline import window_features, CONDS
from src.metrics import rmse, nasa_score
from src.alerts import choose_threshold, evaluate_policy, LEAD

QUANTILES = (0.1, 0.5, 0.9)


def fit_quantiles(F_tr, y_tr, F_va, y_va):
    models = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=1500, learning_rate=0.03,
                              num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1)
        m.fit(F_tr, y_tr, eval_set=[(F_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        models[q] = m
    return models


def predict_q(models, F):
    P = np.column_stack([np.clip(models[q].predict(F), 0, RUL_CAP) for q in QUANTILES])
    return np.sort(P, axis=1)          # prevent quantile crossing


def main(fd="FD001", window=30):
    train, test, rul_true = load_subset(fd)
    train = add_train_rul(train)
    pre = Preprocessor(CONDS[fd]).fit(train)
    train_n, test_n = pre.transform(train), pre.transform(test)
    feats = pre.sensors_
    X, y, units = make_windows(train_n, feats, window, rul_col="rul")
    tr_ids, va_ids = engine_split(units)
    tr, va = np.isin(units, list(tr_ids)), np.isin(units, list(va_ids))
    F = window_features(X)
    models = fit_quantiles(F[tr], y[tr], F[va], y[va])

    Pva = predict_q(models, F[va])
    Xt, _, _ = make_windows(test_n, feats, window, last_only=True)
    Pte = predict_q(models, window_features(Xt))
    lo, med, hi = Pte[:, 0], Pte[:, 1], Pte[:, 2]
    true_cap = np.minimum(rul_true, RUL_CAP)

    out = {"subset": fd,
           "median_rmse": rmse(rul_true, med), "median_nasa": nasa_score(rul_true, med),
           "interval_80_coverage": float(np.mean((true_cap >= lo) & (true_cap <= hi))),
           "mean_interval_width": float(np.mean(hi - lo))}

    # alert policies: threshold picked on validation windows, applied once to test
    y_va = y[va]
    policies = {"median": (Pva[:, 1], med), "lower_bound_q10": (Pva[:, 0], lo)}
    for name, (p_val, p_test) in policies.items():
        T = choose_threshold(p_val, y_va)
        out[f"alert_{name}"] = evaluate_policy(p_test, rul_true, T)
    out["alert_never"] = evaluate_policy(np.full(len(rul_true), 1e9), rul_true, -1)
    out["alert_always"] = evaluate_policy(np.zeros(len(rul_true)), rul_true, 1e9)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"reports/quantile_{fd}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")

import numpy as np
import pandas as pd
from src.data import add_train_rul, make_windows, engine_split, RUL_CAP
from src.metrics import rmse, nasa_score
from src.alerts import total_cost, choose_threshold


def _fake_engines(n_units=4, length=50):
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, length + 1):
            rows.append({"unit": u, "cycle": c, "s1": float(c), "s2": float(u)})
    return pd.DataFrame(rows)


def test_rmse_zero_for_perfect_prediction():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([50], [60]) > nasa_score([50], [40])


def test_rul_is_capped_and_ends_at_zero():
    df = add_train_rul(_fake_engines(length=300))
    assert df["rul"].max() == RUL_CAP
    assert (df.groupby("unit")["rul"].min() == 0).all()


def test_windows_shape_and_last_only():
    df = add_train_rul(_fake_engines(n_units=3, length=50))
    X, y, units = make_windows(df, ["s1", "s2"], window=30, rul_col="rul")
    assert X.shape == (3 * (50 - 30 + 1), 30, 2) and len(y) == len(units)
    Xl, _, ul = make_windows(df, ["s1", "s2"], window=30, last_only=True)
    assert Xl.shape == (3, 30, 2) and sorted(ul) == [1, 2, 3]


def test_short_history_is_padded():
    df = _fake_engines(n_units=1, length=10)
    X, _, _ = make_windows(df, ["s1"], window=30, last_only=True)
    assert X.shape == (1, 30, 1)


def test_engine_split_has_no_overlap():
    units = np.repeat(np.arange(100), 5)
    tr, va = engine_split(units, val_frac=0.2)
    assert tr.isdisjoint(va) and len(va) == 20


def test_alert_cost_and_threshold():
    true = np.array([5, 10, 80, 100])
    pred = np.array([5, 12, 85, 90])
    cost, miss, fa = total_cost(pred, true, T=20)
    assert miss == 0 and fa == 0 and cost == 0
    assert 12 <= choose_threshold(pred, true) < 85

"""C-MAPSS loading, preprocessing and windowing."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OP_COLS = [f"op{i}" for i in (1, 2, 3)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS
RUL_CAP = 125


def load_subset(fd: str):
    """Return train df, test df, true RUL of last test cycle per engine (array)."""
    kw = dict(sep=r"\s+", header=None, names=COLS)
    train = pd.read_csv(RAW / f"train_{fd}.txt", **kw)
    test = pd.read_csv(RAW / f"test_{fd}.txt", **kw)
    rul = pd.read_csv(RAW / f"RUL_{fd}.txt", header=None)[0].to_numpy()
    return train, test, rul


def add_train_rul(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """RUL = cycles to failure, capped (piecewise-linear target)."""
    train = train.copy()
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = (max_cycle - train["cycle"]).clip(upper=cap)
    return train


class Preprocessor:
    """Drops near-constant sensors; normalises per operating condition.

    FD001/FD003 have one condition -> single global scaler.
    FD002/FD004 have six conditions -> cluster op settings, scale per cluster.
    Fit on TRAIN only.
    """

    def __init__(self, n_conditions: int = 1, var_tol: float = 1e-4):
        self.n_conditions = n_conditions
        self.var_tol = var_tol

    def _cond(self, df):
        if self.n_conditions == 1:
            return np.zeros(len(df), dtype=int)
        return self.km_.predict(df[OP_COLS].to_numpy())

    def fit(self, train: pd.DataFrame):
        if self.n_conditions > 1:
            self.km_ = KMeans(self.n_conditions, n_init=10, random_state=0).fit(
                train[OP_COLS].to_numpy())
        cond = self._cond(train)
        # a sensor is useless if it is ~constant WITHIN every condition
        keep = []
        for s in SENSOR_COLS:
            within_std = train.groupby(cond)[s].std().max()
            if within_std > self.var_tol:
                keep.append(s)
        self.sensors_ = keep
        self.scalers_ = {
            c: StandardScaler().fit(train.loc[cond == c, keep]) for c in np.unique(cond)
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cond = self._cond(df)
        out = np.zeros((len(df), len(self.sensors_)))
        for c, sc in self.scalers_.items():
            m = cond == c
            if m.any():
                out[m] = sc.transform(df.loc[m, self.sensors_])
        df[self.sensors_] = out
        return df


def make_windows(df, features, window=30, rul_col=None, last_only=False):
    """Sliding windows per engine. Short histories are left-padded with the first row.

    last_only=True -> one window per engine (the NASA test protocol).
    Returns X [n, window, n_feat], y [n] (or None), unit ids [n].
    """
    X, y, units = [], [], []
    for u, g in df.groupby("unit"):
        arr = g[features].to_numpy(dtype=np.float32)
        if len(arr) < window:
            arr = np.vstack([np.repeat(arr[:1], window - len(arr), axis=0), arr])
        rul = g[rul_col].to_numpy() if rul_col else None
        ends = [len(arr) - 1] if last_only else range(window - 1, len(arr))
        # note: for padded engines len(arr) already >= window
        if rul is not None and len(rul) < len(arr):
            rul = np.concatenate([np.repeat(rul[:1], len(arr) - len(rul)), rul])
        for e in ends:
            X.append(arr[e - window + 1: e + 1])
            units.append(u)
            if rul is not None:
                y.append(rul[e])
    return np.stack(X), (np.array(y, dtype=np.float32) if rul_col else None), np.array(units)


def engine_split(units, val_frac=0.2, seed=0):
    """Split engine IDs (not rows) into train/val -> no leakage."""
    rng = np.random.default_rng(seed)
    u = np.array(sorted(set(units)))
    rng.shuffle(u)
    n_val = int(len(u) * val_frac)
    return set(u[n_val:]), set(u[:n_val])

"""1D-CNN / GRU RUL models (Keras), with seed ensembling.

Usage: python -m src.deep FD001 gru [epochs] [n_seeds] [window]
Each ensemble member uses a different engine-level validation split, so the
ensemble as a whole trains on every engine while early stopping stays leak-free.
"""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from src.data import load_subset, add_train_rul, Preprocessor, make_windows, engine_split, RUL_CAP
from src.metrics import rmse, nasa_score
from src.baseline import CONDS


def build_cnn(window, n_feat):
    return models.Sequential([
        layers.Input((window, n_feat)),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.Flatten(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"), layers.Dense(1)])


def build_gru(window, n_feat):
    return models.Sequential([
        layers.Input((window, n_feat)),
        layers.GRU(64, return_sequences=True, dropout=0.1),
        layers.GRU(64, dropout=0.1),
        layers.Dropout(0.3), layers.Dense(32, activation="relu"), layers.Dense(1)])


BUILDERS = {"cnn": build_cnn, "gru": build_gru}


def run(fd="FD001", kind="gru", epochs=40, n_seeds=3, window=30, bs=256, save=True):
    train, test, rul_true = load_subset(fd)
    train = add_train_rul(train)
    pre = Preprocessor(CONDS[fd]).fit(train)
    train_n, test_n = pre.transform(train), pre.transform(test)
    feats = pre.sensors_
    X, y, units = make_windows(train_n, feats, window, rul_col="rul")
    Xtest, _, _ = make_windows(test_n, feats, window, last_only=True)

    preds, members = [], []
    for seed in range(n_seeds):
        tf.keras.utils.set_random_seed(seed)
        tr_ids, va_ids = engine_split(units, seed=seed)
        tr, va = np.isin(units, list(tr_ids)), np.isin(units, list(va_ids))
        model = BUILDERS[kind](window, len(feats))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        cbs = [callbacks.EarlyStopping(patience=8, restore_best_weights=True),
               callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
        model.fit(X[tr], y[tr] / RUL_CAP, validation_data=(X[va], y[va] / RUL_CAP),
                  epochs=epochs, batch_size=bs, callbacks=cbs, verbose=0)
        p = np.clip(model.predict(Xtest, verbose=0).ravel() * RUL_CAP, 0, RUL_CAP)
        preds.append(p)
        members.append({"seed": seed, "test_rmse": rmse(rul_true, p)})
        if save:
            model.save(f"models/{kind}_{fd}_s{seed}.keras")
    ens = np.mean(preds, axis=0)
    res = {"subset": fd, "model": kind, "window": window, "n_seeds": n_seeds,
           "members": members, "test_rmse": rmse(rul_true, ens),
           "test_nasa_score": nasa_score(rul_true, ens)}
    print(json.dumps(res))
    np.save(f"models/pred_{kind}_{fd}.npy", ens)
    return res


if __name__ == "__main__":
    a = sys.argv
    run(a[1] if len(a) > 1 else "FD001", a[2] if len(a) > 2 else "gru",
        int(a[3]) if len(a) > 3 else 40, int(a[4]) if len(a) > 4 else 3,
        int(a[5]) if len(a) > 5 else 30)

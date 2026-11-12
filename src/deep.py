"""1D-CNN and GRU RUL models (Keras). Usage: python -m src.deep FD001 cnn|gru [epochs]"""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from src.data import load_subset, add_train_rul, Preprocessor, make_windows, engine_split, RUL_CAP
from src.metrics import rmse, nasa_score
from src.baseline import CONDS

tf.keras.utils.set_random_seed(0)


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


def run(fd="FD001", kind="cnn", epochs=30, window=30, bs=256):
    train, test, rul_true = load_subset(fd)
    train = add_train_rul(train)
    pre = Preprocessor(CONDS[fd]).fit(train)
    train_n, test_n = pre.transform(train), pre.transform(test)
    feats = pre.sensors_
    X, y, units = make_windows(train_n, feats, window, rul_col="rul")
    tr_ids, va_ids = engine_split(units)
    tr, va = np.isin(units, list(tr_ids)), np.isin(units, list(va_ids))
    Xtest, _, _ = make_windows(test_n, feats, window, last_only=True)

    model = BUILDERS[kind](window, len(feats))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cbs = [callbacks.EarlyStopping(patience=8, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(patience=3, factor=0.5)]
    model.fit(X[tr], y[tr] / RUL_CAP, validation_data=(X[va], y[va] / RUL_CAP),
              epochs=epochs, batch_size=bs, callbacks=cbs, verbose=0)
    val_pred = model.predict(X[va], verbose=0).ravel() * RUL_CAP
    pred = np.clip(model.predict(Xtest, verbose=0).ravel() * RUL_CAP, 0, RUL_CAP)
    res = {"subset": fd, "model": kind, "val_rmse": rmse(y[va], val_pred),
           "test_rmse": rmse(rul_true, pred), "test_nasa_score": nasa_score(rul_true, pred)}
    print(json.dumps(res))
    model.save(f"models/{kind}_{fd}.keras")
    return res


if __name__ == "__main__":
    fd = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    kind = sys.argv[2] if len(sys.argv) > 2 else "cnn"
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    run(fd, kind, ep)

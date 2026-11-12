"""Evaluate saved ensemble members: python -m src.evaluate_ens FD001 gru [window]"""
import os, sys, glob, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
from src.data import load_subset, add_train_rul, Preprocessor, make_windows, RUL_CAP
from src.metrics import rmse, nasa_score
from src.baseline import CONDS

fd, kind = sys.argv[1], sys.argv[2]
window = int(sys.argv[3]) if len(sys.argv) > 3 else 30
train, test, rul_true = load_subset(fd)
train = add_train_rul(train)
pre = Preprocessor(CONDS[fd]).fit(train)
Xtest, _, _ = make_windows(pre.transform(test), pre.sensors_, window, last_only=True)
files = sorted(glob.glob(f"models/{kind}_{fd}_s*.keras"))
preds = [np.clip(tf.keras.models.load_model(f).predict(Xtest, verbose=0).ravel() * RUL_CAP, 0, RUL_CAP) for f in files]
ens = np.mean(preds, axis=0)
print(json.dumps({"subset": fd, "model": kind, "n_members": len(files),
                  "member_rmse": [round(rmse(rul_true, p), 2) for p in preds],
                  "ens_rmse": rmse(rul_true, ens), "ens_nasa": nasa_score(rul_true, ens)}))

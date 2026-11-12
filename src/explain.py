"""Which sensors drive predicted RUL? Exact TreeSHAP values via LightGBM pred_contrib.
Usage: python -m src.explain FD001
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import make_windows
from src.baseline import window_features
from src.pipeline import fit_subset, feature_names, STATS


def sensor_importance(bundle, n_max=4000):
    t = bundle["test_n"]
    X, _, _ = make_windows(t, bundle["sensors"], bundle["window"])
    F = window_features(X)[:: max(1, len(X) // n_max)]
    contrib = bundle["models"][0.5].predict(F, pred_contrib=True)[:, :-1]   # drop bias column
    names = feature_names(bundle["sensors"])
    per_feature = np.abs(contrib).mean(0)
    per_sensor = {s: float(sum(per_feature[i] for i, n in enumerate(names) if n.rsplit("_", 1)[0] == s))
                  for s in bundle["sensors"]}
    return dict(sorted(per_sensor.items(), key=lambda kv: -kv[1]))


def main(fd="FD001"):
    b = fit_subset(fd)
    imp = sensor_importance(b)
    json.dump(imp, open(f"reports/sensor_importance_{fd}.json", "w"), indent=1)
    top = list(imp.items())[:10][::-1]
    plt.figure(figsize=(6, 4))
    plt.barh([k for k, _ in top], [v for _, v in top], color="#2a6fdb")
    plt.xlabel("mean |SHAP| contribution to predicted RUL (cycles)")
    plt.title(f"PrognosAI {fd}: top sensors driving RUL")
    plt.tight_layout(); plt.savefig(f"reports/sensor_importance_{fd}.png", dpi=150)
    print(json.dumps(dict(list(imp.items())[:8]), indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FD001")

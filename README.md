# PrognosAI

*Predict engine failure before it happens: RUL forecasting with uncertainty and cost-based alerts.*

Predictive maintenance on the NASA C-MAPSS turbofan dataset.

Predict remaining useful life (RUL) of turbofan engines from multivariate sensor time series, and turn the predictions into maintenance alerts.

## Status
- [x] Data loading, per-condition normalisation, capped RUL target, windowing, engine-level split
- [x] Metrics: RMSE + NASA asymmetric score
- [x] LightGBM baseline (rolling mean/std/slope features)
- [x] 1D-CNN and GRU on 30-cycle windows (Keras), first pass on FD001 only
- [x] Uncertainty: quantile LightGBM (10th/50th/90th percentile) -> RUL intervals
- [x] Cost-based alert threshold (chosen on validation engines, applied once to test)
- [x] Unit tests + GitHub Actions CI
- [x] Explainability: exact TreeSHAP sensor importance (LightGBM pred_contrib)
- [x] Streamlit dashboard (fleet view, engine view, sensor drivers)
- [ ] FastAPI endpoint + Docker

## Run
    pip install -r requirements.txt
    python -m src.baseline FD001
    python -m src.deep FD001 gru 30   # or cnn
    python -m src.quantile FD001      # intervals + alert policies
    python -m src.explain FD001       # sensor importance chart
    streamlit run app/dashboard.py    # interactive dashboard
    python -m pytest -q

Data: NASA C-MAPSS (simulated turbofan run-to-failure data), in data/raw/.

## Results (test set, last window per engine)

| Subset | Model | RMSE | NASA score |
|---|---|---|---|
| FD001 | LightGBM (window features) | 13.3 | 237 |
| FD001 | GRU | 15.7 | 371 |
| FD001 | GRU ensemble (2 seeds) | 15.9 | 348 |
| FD001 | 1D-CNN | 19.0 | 977 |
| FD001 | 0.7 LightGBM + 0.3 GRU blend | 13.5 | 237 |
| FD002 | LightGBM | 26.8 | 8291 |
| FD003 | LightGBM | 14.4 | 338 |
| FD004 | LightGBM | 27.7 | 5561 |

Deep models are untuned first passes (single seed, 80/20 engine split for early stopping). Tuned results will replace these.


## Uncertainty and cost-based alerts (LightGBM quantile models)

Median RMSE is on par with the point model. An alert fires when the predicted RUL (lower bound or median) is at or below a threshold T.
Assumptions: lead time 30 cycles; a missed failure (no alert, true RUL <= 30) costs 10; a false alarm (alert, true RUL > 30) costs 1.
T is tuned on held-out validation engines, never on the test set. Cost is per test engine (lower is better).

| Subset | Median RMSE | 80% interval coverage | Never alert | Always alert | Alert on median | Alert on lower bound (q10) |
|---|---|---|---|---|---|---|
| FD001 | 13.2 | 0.64 | 2.50 | 0.75 | 0.04 | 0.02 |
| FD002 | 26.1 | 0.69 | 2.36 | 0.76 | 0.04 | 0.05 |
| FD003 | 14.9 | 0.70 | 2.00 | 0.80 | 0.02 | 0.02 |
| FD004 | 27.2 | 0.66 | 2.14 | 0.79 | 0.12 | 0.10 |

Limitations:
- The nominal 80% intervals cover only 64-70% of true values, so they are overconfident; calibration (e.g. conformal prediction) is future work.
- Costs and lead time are illustrative assumptions, not real maintenance data, and the test sets have only 100-259 engines, so differences between the two alert policies are small.
- C-MAPSS is simulated data.


## Dashboard

`streamlit run app/dashboard.py` trains the quantile models for the chosen subset (about 15 seconds, cached) and shows:
- **Fleet view:** every engine with predicted RUL, an 80% interval and an ALERT / WATCH / OK status, sorted by the RUL lower bound.
- **Engine view:** predicted RUL over time with its interval against the benchmark ground truth, plus the top sensor trends.
- **Sensor drivers:** which sensors push the predicted RUL up or down (TreeSHAP).
- Sidebar sliders for lead time and the costs of missed failures and false alarms; the alert threshold is re-tuned on validation engines each time.

![Top sensors](reports/sensor_importance_FD001.png)

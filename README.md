# PrognosAI

*Predict engine failure before it happens: RUL forecasting with uncertainty and cost-based alerts.*

Predictive maintenance on the NASA C-MAPSS turbofan dataset.

Predict remaining useful life (RUL) of turbofan engines from multivariate sensor time series, and turn the predictions into maintenance alerts.

## Status
- [x] Data loading, per-condition normalisation, capped RUL target, windowing, engine-level split
- [x] Metrics: RMSE + NASA asymmetric score
- [x] LightGBM baseline (rolling mean/std/slope features)
- [x] 1D-CNN and GRU on 30-cycle windows (Keras), first pass on FD001 only
- [ ] Uncertainty (quantile regression / MC dropout)
- [ ] Cost-based alert threshold
- [ ] SHAP explainability
- [ ] Streamlit dashboard, FastAPI + Docker

## Run
    pip install -r requirements.txt
    python -m src.baseline FD001
    python -m src.deep FD001 gru 30   # or cnn

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

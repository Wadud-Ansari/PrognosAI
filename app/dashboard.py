"""PrognosAI dashboard. Run:  streamlit run app/dashboard.py"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.alerts import choose_threshold, total_cost
from src.data import RUL_CAP
from src.explain import sensor_importance
from src.pipeline import engine_curve, fit_subset, fleet_table

st.set_page_config(page_title="PrognosAI", page_icon="🛠️", layout="wide")

COLORS = {"ALERT": "#d62728", "WATCH": "#ff9f1c", "OK": "#2ca02c"}


@st.cache_resource(show_spinner="Training quantile models (about 15 s)...")
def get_bundle(fd):
    b = fit_subset(fd)
    b["importance"] = sensor_importance(b)
    b["fleet"] = fleet_table(b)
    return b


# ---------------- sidebar ----------------
st.sidebar.title("PrognosAI")
st.sidebar.caption("Predict engine failure before it happens.")
fd = st.sidebar.selectbox("Dataset (C-MAPSS subset)", ["FD001", "FD002", "FD003", "FD004"])
lead = st.sidebar.slider("Maintenance lead time (cycles)", 10, 60, 30)
c_fail = st.sidebar.slider("Cost of a missed failure", 1, 50, 10)
c_early = st.sidebar.slider("Cost of a false alarm", 1, 10, 1)
watch_margin = st.sidebar.slider("WATCH margin above alert threshold", 5, 40, 20)

b = get_bundle(fd)
units, P = b["fleet"]
units = list(units)
lo, med, hi = P[:, 0], P[:, 1], P[:, 2]
true = b["rul_true"]

# threshold is chosen on VALIDATION engines only
T = choose_threshold(b["val_pred"][:, 0], b["val_true"], lead=lead, c_fail=c_fail, c_early=c_early)
status = np.where(lo <= T, "ALERT", np.where(lo <= T + watch_margin, "WATCH", "OK"))

st.title("PrognosAI: fleet health and maintenance alerts")
st.caption(
    "NASA C-MAPSS turbofan data (simulated). Alert rule: predicted RUL lower bound (10th percentile) "
    "is at or below the threshold. The threshold is tuned on held-out validation engines."
)

# ---------------- KPIs ----------------
kw = dict(lead=lead, c_fail=c_fail, c_early=c_early)
cost_alert = total_cost(lo, true, T, **kw)
cost_never = total_cost(np.full(len(true), 1e9), true, -1, **kw)
cost_always = total_cost(np.zeros(len(true)), true, 1e9, **kw)
k = st.columns(5)
k[0].metric("Engines monitored", len(units))
k[1].metric("ALERT", int((status == "ALERT").sum()))
k[2].metric("WATCH", int((status == "WATCH").sum()))
k[3].metric("Alert threshold (cycles)", f"{T:.0f}")
k[4].metric(
    "Cost / engine (this policy)",
    f"{cost_alert[0] / len(true):.2f}",
    help=(
        f"Never alert: {cost_never[0] / len(true):.2f} | Alert on every engine: "
        f"{cost_always[0] / len(true):.2f}. Missed failures: {cost_alert[1]}, "
        f"false alarms: {cost_alert[2]}. Evaluated with benchmark ground truth."
    ),
)

tab_fleet, tab_engine, tab_sensors = st.tabs(["Fleet view", "Engine view", "Why? Sensor drivers"])

# ---------------- fleet ----------------
with tab_fleet:
    df = pd.DataFrame(
        {
            "engine": units,
            "status": status,
            "RUL q10": lo.round(0),
            "RUL median": med.round(0),
            "RUL q90": hi.round(0),
            "true RUL (benchmark)": np.minimum(true, RUL_CAP),
        }
    ).sort_values("RUL q10").reset_index(drop=True)

    fig = go.Figure()
    for s in ("ALERT", "WATCH", "OK"):
        d = df[df.status == s]
        fig.add_trace(
            go.Scatter(
                x=d["engine"].astype(str),
                y=d["RUL median"],
                mode="markers",
                name=s,
                marker=dict(color=COLORS[s], size=7),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(d["RUL q90"] - d["RUL median"]),
                    arrayminus=(d["RUL median"] - d["RUL q10"]),
                    color=COLORS[s],
                    thickness=1,
                ),
            )
        )
    fig.add_hline(y=T, line_dash="dash", annotation_text=f"alert threshold {T:.0f}")
    fig.update_layout(
        height=420,
        xaxis=dict(title="Engine (sorted by RUL lower bound)", type="category", showticklabels=False),
        yaxis_title="Predicted RUL (cycles), 80% interval",
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, height=320)

# ---------------- engine ----------------
with tab_engine:
    default_unit = int(df.iloc[0]["engine"])
    sorted_units = sorted(units)
    unit = st.selectbox("Engine", sorted_units, index=sorted_units.index(default_unit))
    i = units.index(unit)
    cycles, Pc = engine_curve(b, unit)
    last_cycle = int(b["test_raw"][b["test_raw"]["unit"] == unit]["cycle"].max())
    true_curve = np.minimum(true[i] + (last_cycle - cycles), RUL_CAP)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycles, y=Pc[:, 2], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=cycles, y=Pc[:, 0], fill="tonexty", fillcolor="rgba(42,111,219,0.2)",
            line=dict(width=0), name="80% interval",
        )
    )
    fig.add_trace(go.Scatter(x=cycles, y=Pc[:, 1], line=dict(color="#2a6fdb"), name="Predicted RUL (median)"))
    fig.add_trace(
        go.Scatter(x=cycles, y=true_curve, line=dict(color="black", dash="dot"), name="True RUL (benchmark, capped)")
    )
    fig.add_hline(y=T, line_dash="dash", line_color="#d62728", annotation_text="alert threshold")
    fig.update_layout(height=380, xaxis_title="Cycle", yaxis_title="RUL (cycles)", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"Current status of engine {unit}: **{status[i]}**")

    top = list(b["importance"])[:4]
    g = b["test_n"][b["test_n"]["unit"] == unit]
    fig2 = go.Figure()
    for s in top:
        fig2.add_trace(go.Scatter(x=g["cycle"], y=g[s], name=s))
    fig2.update_layout(
        height=300, xaxis_title="Cycle", yaxis_title="Sensor (normalised)",
        title="Top sensor trends for this engine", margin=dict(t=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- sensors ----------------
with tab_sensors:
    imp = list(b["importance"].items())[:12][::-1]
    fig = go.Figure(
        go.Bar(x=[v for _, v in imp], y=[k_ for k_, _ in imp], orientation="h", marker_color="#2a6fdb")
    )
    fig.update_layout(height=420, xaxis_title="mean |SHAP| contribution to predicted RUL (cycles)", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Exact TreeSHAP values from the median LightGBM model, summed over each sensor's window "
        "features (last value, mean, std, slope)."
    )

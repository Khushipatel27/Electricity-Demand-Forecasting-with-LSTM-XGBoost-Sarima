"""Streamlit dashboard — Electricity Demand Forecasting (Panama 2015–2020).

Four tabs:
    1. Overview    — KPI cards, dataset summary, quick pattern charts
    2. Explorer    — interactive time-series explorer with multi-overlay
    3. Forecast    — live LSTM / XGBoost inference on any test date
    4. Comparison  — side-by-side model metrics, interactive charts
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import (
    TARGET_COL,
    engineer_features,
    get_feature_columns,
    load_data,
    temporal_train_test_split,
)
from src.utils import MODELS_DIR, OUTPUTS_DIR, load_json

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Metric card styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #16304f 100%);
        border: 1px solid #2d5986;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] > div { color: #e8f4fd; }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #90b4d4 !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #ffffff !important; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Info/success/warning boxes */
    .insight-box {
        background: linear-gradient(135deg, #0d3349 0%, #0a2535 100%);
        border-left: 4px solid #3d9df0;
        border-radius: 0 10px 10px 0;
        padding: 14px 20px;
        margin: 12px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .model-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin: 2px;
    }
    .badge-lstm   { background: #1a5276; color: #7fb3d3; border: 1px solid #2980b9; }
    .badge-xgb    { background: #1a4a2e; color: #7dcea0; border: 1px solid #27ae60; }
    .badge-sarima { background: #4a3728; color: #d4a96a; border: 1px solid #d35400; }
    .badge-ok     { background: #1a4a2e; color: #7dcea0; border: 1px solid #27ae60; }
    .badge-miss   { background: #4a1a1a; color: #e59090; border: 1px solid #c0392b; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #7fb3d3;
        border-bottom: 2px solid #2d5986;
        padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0d1b2a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset...")
def load_full_data() -> pd.DataFrame:
    df = load_data()
    return engineer_features(df)


@st.cache_resource(show_spinner="Loading LSTM model...")
def load_lstm_model():
    import joblib
    paths = [MODELS_DIR / "lstm_best.keras",
             MODELS_DIR / "feature_scaler.pkl",
             MODELS_DIR / "target_scaler.pkl"]
    if not all(p.exists() for p in paths):
        return None, None, None
    import tensorflow as tf
    return (tf.keras.models.load_model(str(paths[0])),
            joblib.load(paths[1]),
            joblib.load(paths[2]))


@st.cache_resource(show_spinner="Loading XGBoost model...")
def load_xgb_model():
    import joblib
    path = MODELS_DIR / "xgboost_model.pkl"
    return joblib.load(path) if path.exists() else None


def load_metrics(name: str) -> dict:
    path = OUTPUTS_DIR / f"{name}_metrics.json"
    return load_json(path) if path.exists() else {}


# ─────────────────────────────────────────────────────────────
# Load data & split
# ─────────────────────────────────────────────────────────────
df = load_full_data()
train_df, test_df = temporal_train_test_split(df)
feature_cols = get_feature_columns(df)

lstm_metrics   = load_metrics("lstm")
sarima_metrics = load_metrics("sarima")
xgb_metrics    = load_metrics("xgboost")

MODEL_COLORS = {
    "Actual":  "#ffffff",
    "LSTM":    "#4a9fd4",
    "SARIMA":  "#e8974a",
    "XGBoost": "#5dbf8a",
}

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Demand Forecasting")
    st.markdown("**Panama National Grid · 2015–2020**")
    st.divider()

    st.markdown("### Model Status")

    def status_badge(label: str, metrics: dict, color_cls: str) -> str:
        if metrics:
            mape = metrics.get("MAPE_pct", "?")
            return (f'<span class="model-badge badge-ok">✓ {label}</span> '
                    f'<span style="font-size:0.78rem;color:#aaa;">MAPE: {mape}%</span>')
        return f'<span class="model-badge badge-miss">✗ {label} — not trained</span>'

    st.markdown(status_badge("LSTM", lstm_metrics, "lstm"), unsafe_allow_html=True)
    st.markdown(status_badge("XGBoost", xgb_metrics, "xgb"), unsafe_allow_html=True)
    st.markdown(status_badge("SARIMA", sarima_metrics, "sarima"), unsafe_allow_html=True)

    st.divider()
    st.markdown("### Dataset Info")
    st.markdown(f"- **Records:** {len(df):,} hourly rows")
    st.markdown(f"- **Range:** {df.index.min().date()} → {df.index.max().date()}")
    st.markdown(f"- **Features:** {len(feature_cols)}")
    st.markdown(f"- **Train:** 2015 – 2019 ({len(train_df):,} rows)")
    st.markdown(f"- **Test:** Jan – Jun 2020 ({len(test_df):,} rows)")

    st.divider()
    st.markdown("### Run Training")
    st.code("python -m src.train_lstm\npython -m src.train_xgboost\npython -m src.train_sarima",
            language="bash")

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Data Explorer",
    "🔮 Live Forecast",
    "📈 Model Comparison",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Panama Electricity Demand Forecasting")
    st.markdown(
        "Multi-model forecasting pipeline — **LSTM** (deep learning, 24-hour multi-step), "
        "**XGBoost** (gradient boosting, 1-step-ahead), and **SARIMA** (statistical baseline) "
        "trained on 48K hourly records from Panama's national grid."
    )

    # KPI row
    mean_d = df[TARGET_COL].mean()
    max_d  = df[TARGET_COL].max()
    min_d  = df[TARGET_COL].min()
    std_d  = df[TARGET_COL].std()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Demand",  f"{mean_d:,.0f} MW")
    c2.metric("Peak Demand", f"{max_d:,.0f} MW")
    c3.metric("Min Demand",  f"{min_d:,.0f} MW")
    c4.metric("Volatility",  f"{std_d:,.0f} MW",  help="Standard deviation of hourly demand")
    c5.metric("Total Hours", f"{len(df):,}")

    st.divider()

    # Best model callout
    if lstm_metrics or xgb_metrics:
        best_model, best_mape = ("XGBoost", xgb_metrics.get("MAPE_pct")) if xgb_metrics else ("LSTM", lstm_metrics.get("MAPE_pct"))
        best_mae = (xgb_metrics if xgb_metrics else lstm_metrics).get("MAE_MW", "—")
        st.markdown(
            f'<div class="insight-box">🏆 <strong>Best model: {best_model}</strong> — '
            f'achieves <strong>{best_mape}% MAPE</strong> on the 2020 holdout set, '
            f'equivalent to an average error of <strong>{best_mae} MW</strong> against a mean demand of '
            f'<strong>{mean_d:,.0f} MW</strong>.</div>',
            unsafe_allow_html=True,
        )

    # Two-column layout: full series + daily pattern
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-header">Full Demand History</div>', unsafe_allow_html=True)
        daily = df[TARGET_COL].resample("D").mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily.values,
            fill="tozeroy", fillcolor="rgba(74,159,212,0.15)",
            line=dict(color="#4a9fd4", width=1.2),
            name="Daily avg demand",
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.0f} MW<extra></extra>",
        ))
        fig.add_shape(
            type="rect",
            x0="2020-01-01", x1=str(daily.index.max().date()),
            y0=0, y1=1, yref="paper",
            fillcolor="rgba(93,191,138,0.1)", line_width=0,
        )
        fig.add_annotation(
            x="2020-03-15", y=0.95, yref="paper",
            text="Test period", showarrow=False,
            font=dict(color="#5dbf8a", size=11),
        )
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title=None, yaxis_title="Demand (MW)",
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cccccc"),
            xaxis=dict(gridcolor="#1e3a5f"),
            yaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Average by Hour of Day</div>', unsafe_allow_html=True)
        hourly_avg = df.groupby("hour")[TARGET_COL].mean().reset_index()
        fig2 = go.Figure(go.Scatter(
            x=hourly_avg["hour"], y=hourly_avg[TARGET_COL],
            mode="lines+markers",
            line=dict(color="#4a9fd4", width=2.5),
            marker=dict(size=7, color="#4a9fd4"),
            fill="tozeroy", fillcolor="rgba(74,159,212,0.12)",
            hovertemplate="Hour %{x}:00<br>Avg: %{y:.0f} MW<extra></extra>",
        ))
        fig2.update_layout(
            height=300, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title="Hour of Day", tickvals=list(range(0, 24, 3)), gridcolor="#1e3a5f"),
            yaxis=dict(title="Avg Demand (MW)", gridcolor="#1e3a5f"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cccccc"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: hour × day_of_week
    st.markdown('<div class="section-header">Demand Heatmap — Hour of Day vs Day of Week</div>',
                unsafe_allow_html=True)
    pivot = df.groupby(["day_of_week", "hour"])[TARGET_COL].mean().unstack()
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig3 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        colorbar=dict(title=dict(text="MW", font=dict(color="#ccc")), tickfont=dict(color="#ccc")),
        hovertemplate="%{y} %{x}<br>Avg: %{z:.0f} MW<extra></extra>",
    ))
    fig3.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Hour", tickfont=dict(color="#ccc")),
        yaxis=dict(title=None, tickfont=dict(color="#ccc")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Weekday evenings peak highest. Weekends show a flatter, lower profile throughout the day.")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Data Explorer
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Data Explorer")
    st.caption("Select a date range and variables to explore the raw dataset interactively.")

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        date_range = st.date_input(
            "Date range",
            value=(df.index.min().date(), df.index.max().date()),
            min_value=df.index.min().date(),
            max_value=df.index.max().date(),
        )
    with ctrl2:
        resample_opt = st.selectbox(
            "Aggregation",
            ["Hourly (raw)", "Daily average", "Weekly average", "Monthly average"],
        )
    with ctrl3:
        overlay_temp = st.toggle("Overlay temperature", value=False)

    if len(date_range) == 2:
        start_d, end_d = date_range
        mask = (df.index.date >= start_d) & (df.index.date <= end_d)
        view = df[mask].copy()
    else:
        view = df.copy()

    if view.empty:
        st.warning("No data in selected range — adjust the date picker.")
    else:
        # Resample
        freq_map = {
            "Hourly (raw)": None,
            "Daily average": "D",
            "Weekly average": "W",
            "Monthly average": "ME",
        }
        freq = freq_map[resample_opt]
        plot_demand = view[TARGET_COL].resample(freq).mean() if freq else view[TARGET_COL]
        plot_temp   = view["T2M_toc"].resample(freq).mean() if freq else view["T2M_toc"]

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Min",  f"{view[TARGET_COL].min():,.0f} MW")
        k2.metric("Max",  f"{view[TARGET_COL].max():,.0f} MW")
        k3.metric("Mean", f"{view[TARGET_COL].mean():,.0f} MW",
                  delta=f"{view[TARGET_COL].mean() - df[TARGET_COL].mean():+.0f} vs overall avg")
        k4.metric("Std",  f"{view[TARGET_COL].std():,.0f} MW")

        # Main time series chart
        if overlay_temp:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=plot_demand.index, y=plot_demand.values,
                           name="Demand (MW)", line=dict(color="#4a9fd4", width=1.5),
                           hovertemplate="%{x|%b %d %Y %H:%M}<br>%{y:.0f} MW<extra></extra>"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=plot_temp.index, y=plot_temp.values,
                           name="Temp Tocumen (°C)", line=dict(color="#e87a5a", width=1.2, dash="dot"),
                           hovertemplate="%{x|%b %d}<br>%{y:.1f} °C<extra></extra>"),
                secondary_y=True,
            )
            fig.update_yaxes(title_text="Demand (MW)", secondary_y=False,
                             gridcolor="#1e3a5f", color="#4a9fd4")
            fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True,
                             gridcolor="rgba(0,0,0,0)", color="#e87a5a")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_demand.index, y=plot_demand.values,
                name="Demand (MW)", fill="tozeroy",
                fillcolor="rgba(74,159,212,0.12)",
                line=dict(color="#4a9fd4", width=1.5),
                hovertemplate="%{x|%b %d %Y %H:%M}<br>%{y:.0f} MW<extra></extra>",
            ))

        fig.update_layout(
            title=f"National Electricity Demand — {resample_opt}",
            height=420, margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(gridcolor="#1e3a5f", rangeslider=dict(visible=True), rangeselector=dict(
                buttons=[
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#1e3a5f", activecolor="#2d5986", font=dict(color="#ccc"),
            )),
            yaxis=dict(gridcolor="#1e3a5f"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cccccc"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Two sub-charts
        sub1, sub2 = st.columns(2)
        with sub1:
            st.markdown('<div class="section-header">Avg Demand by Day of Week</div>',
                        unsafe_allow_html=True)
            dow_avg = view.groupby("day_of_week")[TARGET_COL].mean()
            dow_avg.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig_dow = go.Figure(go.Bar(
                x=dow_avg.index, y=dow_avg.values,
                marker_color=["#4a9fd4"]*5 + ["#e87a5a"]*2,
                hovertemplate="%{x}<br>%{y:.0f} MW<extra></extra>",
            ))
            fig_dow.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="#1e3a5f"), xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), showlegend=False,
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        with sub2:
            st.markdown('<div class="section-header">Holiday vs Non-Holiday Distribution</div>',
                        unsafe_allow_html=True)
            fig_box = go.Figure()
            for flag, label, color in [(0, "Non-Holiday", "#4a9fd4"), (1, "Holiday", "#e87a5a")]:
                subset = view[view["holiday"] == flag][TARGET_COL]
                fig_box.add_trace(go.Box(
                    y=subset.values, name=label,
                    marker_color=color, line_color=color,
                    boxmean=True,
                    hovertemplate=f"{label}<br>%{{y:.0f}} MW<extra></extra>",
                ))
            fig_box.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(title="Demand (MW)", gridcolor="#1e3a5f"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with st.expander("Raw data table"):
            show_cols = [TARGET_COL, "T2M_toc", "T2M_san", "T2M_dav",
                         "holiday", "school", "hour", "day_of_week"]
            show_cols = [c for c in show_cols if c in view.columns]
            st.dataframe(view[show_cols].head(500).round(2), height=300)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Live Forecast
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Live Forecast")
    st.caption("Pick any date in the test period, choose a model, and generate a forecast instantly.")

    model_lstm, feat_scaler, tgt_scaler = load_lstm_model()
    model_xgb = load_xgb_model()

    available_models = []
    if model_lstm is not None:
        available_models.append("LSTM (24-step ahead)")
    if model_xgb is not None:
        available_models.append("XGBoost (1-step ahead)")

    if not available_models:
        st.warning("No trained models found. Run the training scripts shown in the sidebar.")
    else:
        ctrl_a, ctrl_b, ctrl_c = st.columns([2, 2, 1])
        with ctrl_a:
            selected_model = st.selectbox("Select model", available_models)
        with ctrl_b:
            selected_date = st.date_input(
                "Forecast start date",
                value=test_df.index.min().date(),
                min_value=test_df.index.min().date(),
                max_value=test_df.index.max().date(),
                help="Choose any date in the 2020 test period",
            )
        with ctrl_c:
            horizon_h = st.selectbox("Horizon (hours)", [24, 48, 72, 168],
                                     help="How many hours ahead to show")

        run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

        if run_btn:
            selected_dt = pd.Timestamp(selected_date)
            pos = df.index.get_indexer([selected_dt], method="nearest")[0]

            with st.spinner("Running inference..."):
                if "LSTM" in selected_model and model_lstm is not None:
                    from src.train_lstm import SEQ_LEN, HORIZON

                    if pos < SEQ_LEN:
                        st.error(f"Need at least {SEQ_LEN} hours of history before the selected date.")
                    else:
                        all_feat_scaled = feat_scaler.transform(df[feature_cols].values)
                        X_in = all_feat_scaled[pos - SEQ_LEN: pos][np.newaxis, ...]
                        y_scaled = model_lstm.predict(X_in, verbose=0)
                        y_mw = tgt_scaler.inverse_transform(
                            y_scaled.reshape(-1, 1)
                        ).flatten()

                        n = min(horizon_h, len(y_mw),
                                len(df) - pos)
                        actual_w = df.iloc[pos: pos + n][TARGET_COL].values
                        pred_w   = y_mw[:n]
                        idx_w    = df.index[pos: pos + n]
                        model_label = "LSTM"
                        pred_color  = MODEL_COLORS["LSTM"]

                elif "XGBoost" in selected_model and model_xgb is not None:
                    n = min(horizon_h, len(df) - pos - 1)
                    X_window = df.iloc[pos: pos + n][feature_cols].values
                    pred_w   = model_xgb.predict(X_window)
                    actual_w = df.iloc[pos + 1: pos + 1 + n][TARGET_COL].values
                    n = min(len(pred_w), len(actual_w))
                    pred_w   = pred_w[:n]
                    actual_w = actual_w[:n]
                    idx_w    = df.index[pos + 1: pos + 1 + n]
                    model_label = "XGBoost"
                    pred_color  = MODEL_COLORS["XGBoost"]

            if n > 0:
                # ── Forecast chart
                error_band_upper = pred_w * 1.05
                error_band_lower = pred_w * 0.95

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=list(idx_w) + list(idx_w[::-1]),
                    y=list(error_band_upper) + list(error_band_lower[::-1]),
                    fill="toself", fillcolor=f"rgba(74,159,212,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="±5% band", showlegend=True,
                    hoverinfo="skip",
                ))
                fig_fc.add_trace(go.Scatter(
                    x=idx_w, y=actual_w, name="Actual",
                    line=dict(color="#ffffff", width=2.5),
                    hovertemplate="%{x|%b %d %H:%M}<br>Actual: %{y:.0f} MW<extra></extra>",
                ))
                fig_fc.add_trace(go.Scatter(
                    x=idx_w, y=pred_w, name=model_label,
                    line=dict(color=pred_color, width=2.5, dash="dash"),
                    hovertemplate="%{x|%b %d %H:%M}<br>Forecast: %{y:.0f} MW<extra></extra>",
                ))
                fig_fc.update_layout(
                    title=f"{model_label} Forecast — {selected_date} ({n}-hour window)",
                    height=420, margin=dict(l=0, r=0, t=50, b=0),
                    xaxis=dict(gridcolor="#1e3a5f", title=None),
                    yaxis=dict(gridcolor="#1e3a5f", title="Demand (MW)"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"), legend=dict(bgcolor="rgba(0,0,0,0)"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                # ── Window metrics
                from sklearn.metrics import r2_score as r2
                abs_e = np.abs(actual_w - pred_w)
                mape_w  = float(np.mean(abs_e / np.abs(actual_w)) * 100)
                mae_w   = float(np.mean(abs_e))
                rmse_w  = float(np.sqrt(np.mean((actual_w - pred_w) ** 2)))
                r2_w    = float(r2(actual_w, pred_w))
                w5_w    = float(np.mean(abs_e / np.abs(actual_w) < 0.05) * 100)
                w10_w   = float(np.mean(abs_e / np.abs(actual_w) < 0.10) * 100)

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("MAE",       f"{mae_w:.1f} MW")
                m2.metric("RMSE",      f"{rmse_w:.1f} MW")
                m3.metric("MAPE",      f"{mape_w:.2f}%")
                m4.metric("R²",        f"{r2_w:.3f}")
                m5.metric("Within 5%", f"{w5_w:.1f}%")
                m6.metric("Within 10%",f"{w10_w:.1f}%")

                # ── Error breakdown chart
                st.markdown('<div class="section-header">Hourly Prediction Error</div>',
                            unsafe_allow_html=True)
                errors = actual_w - pred_w
                colors_err = [("#e87a5a" if e < 0 else "#5dbf8a") for e in errors]
                fig_err = go.Figure(go.Bar(
                    x=list(range(1, n + 1)), y=errors,
                    marker_color=colors_err,
                    hovertemplate="Hour +%{x}<br>Error: %{y:.1f} MW<extra></extra>",
                ))
                fig_err.add_hline(y=0, line_dash="dash", line_color="#888")
                fig_err.update_layout(
                    height=220, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(title="Forecast step (hours ahead)", gridcolor="#1e3a5f"),
                    yaxis=dict(title="Error (MW)", gridcolor="#1e3a5f"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccc"),
                )
                st.plotly_chart(fig_err, use_container_width=True)

        # ── Feature importance (always visible if available)
        fi_path = OUTPUTS_DIR / "feature_importance.csv"
        if fi_path.exists():
            st.divider()
            st.markdown('<div class="section-header">Feature Importance (XGBoost)</div>',
                        unsafe_allow_html=True)
            fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=True).tail(15)
            colors_fi = ["#4a9fd4" if "lag" in f or "rolling" in f
                         else "#5dbf8a" if "T2M" in f or "QV2M" in f or "W2M" in f
                         else "#e8974a"
                         for f in fi_df["feature"]]
            fig_fi = go.Figure(go.Bar(
                x=fi_df["importance"], y=fi_df["feature"],
                orientation="h", marker_color=colors_fi,
                hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
            ))
            fig_fi.update_layout(
                height=480, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title="Feature Importance Score", gridcolor="#1e3a5f"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("Blue = lag/rolling features · Green = weather features · Orange = calendar features")

        # ── Per-horizon LSTM breakdown
        if lstm_metrics and "per_horizon" in lstm_metrics:
            st.divider()
            st.markdown('<div class="section-header">LSTM Accuracy by Forecast Horizon</div>',
                        unsafe_allow_html=True)
            ph = lstm_metrics["per_horizon"]
            horizons = list(ph.keys())
            mapes_h  = [ph[h]["MAPE_pct"] for h in horizons]
            r2s_h    = [ph[h]["R2"] for h in horizons]
            maes_h   = [ph[h]["MAE_MW"] for h in horizons]

            fig_ph = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ph.add_trace(go.Bar(
                x=horizons, y=mapes_h, name="MAPE (%)",
                marker_color="#4a9fd4", opacity=0.8,
                hovertemplate="%{x}<br>MAPE: %{y:.2f}%<extra></extra>",
            ), secondary_y=False)
            fig_ph.add_trace(go.Scatter(
                x=horizons, y=r2s_h, name="R²",
                mode="lines+markers", line=dict(color="#5dbf8a", width=2.5),
                marker=dict(size=8),
                hovertemplate="%{x}<br>R²: %{y:.3f}<extra></extra>",
            ), secondary_y=True)
            fig_ph.update_yaxes(title_text="MAPE (%)", secondary_y=False,
                                 gridcolor="#1e3a5f", color="#4a9fd4")
            fig_ph.update_yaxes(title_text="R²", secondary_y=True,
                                 gridcolor="rgba(0,0,0,0)", color="#5dbf8a")
            fig_ph.update_layout(
                height=300, margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(title="Forecast Horizon", gridcolor="rgba(0,0,0,0)"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_ph, use_container_width=True)
            st.caption("LSTM is most accurate at short horizons. R² degrades gracefully as horizon increases.")


# ══════════════════════════════════════════════════════════════
# TAB 4 — Model Comparison
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Model Comparison")
    st.caption("Side-by-side evaluation of all trained models on the 2020 holdout test set.")

    any_metrics = lstm_metrics or sarima_metrics or xgb_metrics
    if not any_metrics:
        st.warning("No model metrics found. Train at least one model first.")
    else:
        # ── Metrics table
        metric_keys = [
            ("MAE_MW", "MAE (MW)", "lower is better"),
            ("RMSE_MW", "RMSE (MW)", "lower is better"),
            ("MAPE_pct", "MAPE (%)", "lower is better"),
            ("R2", "R²", "higher is better"),
            ("within_5pct", "Within 5% (%)", "higher is better"),
            ("within_10pct", "Within 10% (%)", "higher is better"),
        ]
        all_model_metrics = {}
        if lstm_metrics:   all_model_metrics["LSTM"]    = lstm_metrics
        if xgb_metrics:    all_model_metrics["XGBoost"] = xgb_metrics
        if sarima_metrics: all_model_metrics["SARIMA"]  = sarima_metrics

        # Build styled DataFrame
        rows = []
        for key, label, direction in metric_keys:
            row = {"Metric": label, "Direction": direction}
            for mname, mdict in all_model_metrics.items():
                row[mname] = mdict.get(key, None)
            rows.append(row)

        cmp_df = pd.DataFrame(rows).set_index("Metric")

        st.markdown('<div class="section-header">Metrics Summary</div>', unsafe_allow_html=True)

        # Bold the best value in each row
        model_cols = [c for c in cmp_df.columns if c != "Direction"]

        def highlight_best(row):
            direction = row.get("Direction", "lower is better")
            vals = {c: row[c] for c in model_cols if row[c] is not None}
            if not vals:
                return [""] * len(row)
            best = min(vals, key=vals.get) if "lower" in direction else max(vals, key=vals.get)
            styles = []
            for c in row.index:
                if c == best:
                    styles.append("background-color: #1a4a2e; color: #7dcea0; font-weight: bold;")
                else:
                    styles.append("")
            return styles

        display_df = cmp_df[model_cols].copy()
        for col in model_cols:
            display_df[col] = display_df[col].apply(
                lambda v: f"{v:.3f}" if isinstance(v, float) else "—"
            )

        st.dataframe(display_df, use_container_width=True)

        # ── Note on comparison fairness
        if xgb_metrics:
            st.markdown(
                '<div class="insight-box">ℹ️ <strong>Comparison note:</strong> '
                'XGBoost predicts <strong>1 step ahead (t+1)</strong> — a simpler task. '
                'LSTM predicts <strong>24 steps ahead</strong>. SARIMA forecasts <strong>daily averages</strong>. '
                'XGBoost\'s lower MAPE reflects its easier task, not a fair head-to-head. '
                'For a fair 1-step comparison, LSTM Hour +1 metrics are shown in the Forecast tab.</div>',
                unsafe_allow_html=True,
            )

        # ── Business insight
        if lstm_metrics:
            mape  = lstm_metrics.get("MAPE_pct", "?")
            mae   = lstm_metrics.get("MAE_MW", "?")
            mean_d = df[TARGET_COL].mean()
            st.markdown(
                f'<div class="insight-box">📊 <strong>Business translation:</strong> '
                f'LSTM achieves <strong>{mape}% MAPE</strong> on a 24-hour multi-step forecast — '
                f'equivalent to predicting <strong>{mae} MW</strong> error on an average demand of '
                f'<strong>{mean_d:,.0f} MW</strong>. For grid operators, this is the difference between '
                f'optimal dispatch and costly over/under-generation.</div>',
                unsafe_allow_html=True,
            )

        # ── Interactive grouped bar chart
        st.markdown('<div class="section-header">Metric Comparison (Interactive Bar Chart)</div>',
                    unsafe_allow_html=True)

        bar_metrics = ["MAE_MW", "RMSE_MW", "MAPE_pct"]
        bar_labels  = ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]
        colors_bar  = [MODEL_COLORS["LSTM"], MODEL_COLORS["XGBoost"], MODEL_COLORS["SARIMA"]]

        fig_bar = go.Figure()
        for (mname, mdict), color in zip(all_model_metrics.items(),
                                          [MODEL_COLORS.get(m, "#aaa") for m in all_model_metrics]):
            vals = [mdict.get(k) for k in bar_metrics]
            fig_bar.add_trace(go.Bar(
                name=mname, x=bar_labels, y=vals,
                marker_color=color,
                text=[f"{v:.2f}" if v is not None else "—" for v in vals],
                textposition="outside",
                hovertemplate=f"{mname}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            ))
        fig_bar.update_layout(
            barmode="group", height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(gridcolor="#1e3a5f"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── R² and within-% comparison
        r2_col, w_col = st.columns(2)
        with r2_col:
            st.markdown('<div class="section-header">R² Score by Model</div>',
                        unsafe_allow_html=True)
            r2_vals = {m: d.get("R2") for m, d in all_model_metrics.items() if d.get("R2") is not None}
            fig_r2 = go.Figure(go.Bar(
                x=list(r2_vals.keys()),
                y=list(r2_vals.values()),
                marker_color=[MODEL_COLORS.get(m, "#aaa") for m in r2_vals],
                text=[f"{v:.3f}" for v in r2_vals.values()],
                textposition="outside",
                hovertemplate="%{x}<br>R²: %{y:.4f}<extra></extra>",
            ))
            fig_r2.add_hline(y=0, line_dash="dash", line_color="#888", annotation_text="baseline (mean predictor)")
            fig_r2.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="#1e3a5f", title="R²"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), showlegend=False,
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with w_col:
            st.markdown('<div class="section-header">Predictions Within Error Threshold</div>',
                        unsafe_allow_html=True)
            fig_w = go.Figure()
            for mname, mdict in all_model_metrics.items():
                color = MODEL_COLORS.get(mname, "#aaa")
                fig_w.add_trace(go.Bar(
                    name=mname,
                    x=["Within 5%", "Within 10%"],
                    y=[mdict.get("within_5pct"), mdict.get("within_10pct")],
                    marker_color=color,
                    hovertemplate=f"{mname}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
                ))
            fig_w.update_layout(
                barmode="group", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="#1e3a5f", title="% of predictions"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        # ── Saved PNG plots
        png_plots = [
            ("Full Test Period — All Models",       "fig1_full_test_period.png"),
            ("First 2 Weeks Zoomed",                "fig2_two_week_zoom.png"),
            ("Scatter: Actual vs Predicted",        "fig3_scatter.png"),
            ("Residuals Over Time",                 "fig4_residuals.png"),
            ("Error Distribution Comparison",       "fig5_error_distribution.png"),
        ]
        available_pngs = [(t, OUTPUTS_DIR / f) for t, f in png_plots if (OUTPUTS_DIR / f).exists()]

        if available_pngs:
            st.markdown('<div class="section-header">Evaluation Plots</div>', unsafe_allow_html=True)
            for title, path in available_pngs:
                with st.expander(title, expanded=(title == "First 2 Weeks Zoomed")):
                    st.image(str(path), use_container_width=True)
        else:
            st.info("No evaluation plots found. Run `python -m src.evaluate` after training all models.")

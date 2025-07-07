import streamlit as st
import torch
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

from visualization.toast import inject_toast_js, show_toast
from inference.multi_timeframe import multi_timeframe_prediction

import config
from data.fetcher import fetch_enhanced_data
from preprocessing.anomalies import detect_anomalies
from preprocessing.indicators import calculate_heikin_ashi
from visualization.charts import display_live_chart
from training.trainer import start_training
from analysis.sentiment import analyze_market_sentiment
from analysis.signals import calculate_risk_metrics, generate_trading_signals
from models.ensemble import EnsemblePredictor

# ───── Config ─────────────────────────────────
st.set_page_config(config.PAGE_TITLE, layout=config.PAGE_LAYOUT)
st.markdown(config.APP_CSS, unsafe_allow_html=True)
st.markdown(f'<div class="app-header"><h1>{config.PAGE_TITLE}</h1></div>', unsafe_allow_html=True)

# ───── Sidebar Inputs ─────────────────────────
with st.sidebar:
    SYMBOL               = st.text_input("Binance Symbol", "BTCUSDT")
    INTERVAL             = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"])
    CANDLE_TYPE          = st.selectbox("Candle Type", ["Regular", "Heikin-Ashi"])
    LIVE_BARS            = st.slider("Live bars", 100, 500, 200)
    HISTORY_BARS         = st.slider("History bars", min_value=500, max_value=3000, value=3000, step=500)
    SEQ_LEN              = st.slider("Sequence Length", min_value=16, max_value=256, value=64, step=16)
    ENSEMBLE_SIZE        = st.slider("Ensemble size", 3, 7, 5)
    REFRESH_SEC          = st.slider("Refresh (s)", 1, 60, 10)
    CONFIDENCE_THRESHOLD = st.slider("Confidence thres", 0.5, 0.95, 0.8)
    RISK_TOLERANCE       = st.slider("Risk tol", 0.1, 2.0, 1.0, 0.1)
    STOP_LOSS_PCT        = st.slider("Stop loss %", 1.0, 10.0, 3.0, 0.5)
    TAKE_PROFIT_PCT      = st.slider("Take profit %", 2.0, 20.0, 6.0, 0.5)

    with st.expander("⚙️ Train Other Intervals"):
        train_symbol = st.text_input("Train Symbol", SYMBOL)
        train_interval = st.selectbox("Interval to Train", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
        if st.button("🔔 Train Now"):
            alt_params = {
                "SYMBOL": train_symbol,
                "INTERVAL": train_interval,
                "MTF_INTERVALS": ["1m", "5m", "15m", "1h"],
                "HISTORY_BARS": HISTORY_BARS,
                "SEQ_LEN": SEQ_LEN,
                "ENSEMBLE_SIZE": ENSEMBLE_SIZE,
                "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
                "RISK_TOLERANCE": RISK_TOLERANCE,
                "STOP_LOSS_PCT": STOP_LOSS_PCT,
                "TAKE_PROFIT_PCT": TAKE_PROFIT_PCT,
            }
            inject_toast_js()
            show_toast(f"🔔 Started training for {train_symbol} [{train_interval}]")
            start_training(st.session_state, alt_params)
            st.success(f"✅ Training started for {train_symbol} ({train_interval})")

params = {
    "SYMBOL": SYMBOL,
    "INTERVAL": INTERVAL,
    "HISTORY_BARS": HISTORY_BARS,
    "SEQ_LEN": SEQ_LEN,
    "ENSEMBLE_SIZE": ENSEMBLE_SIZE,
    "CONFIDENCE_THRESHOLD": CONFIDENCE_THRESHOLD,
    "RISK_TOLERANCE": RISK_TOLERANCE,
    "STOP_LOSS_PCT": STOP_LOSS_PCT,
    "TAKE_PROFIT_PCT": TAKE_PROFIT_PCT,
}
RETRAIN_INTERVAL_HOURS = 6

# ───── State Initialization ───────────────────
for k, v in {
    "training": False, "model_trained": False, "model": None,
    "scaler": None, "train_losses": [], "val_losses": [],
    "trained_at": None, "training_error": None,
    "last_retrain_check": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───── Auto-refresh ──────────────────────────
st_autorefresh(interval=REFRESH_SEC * 1000, key="refresh_key")

# ───── UI Placeholders ───────────────────────
progress_ph = st.empty()
status_ph = st.empty()
chart_slot = st.empty()

# ───── Load or Train Model ───────────────────
MTF_INTERVALS = ["1m", "5m", "15m", "1h"]
interval_key = '_'.join(MTF_INTERVALS)

model_path  = f"models/trained_model/{SYMBOL}/{SYMBOL}_model_{interval_key}.pt"
scaler_path = f"models/trained_model/{SYMBOL}/{SYMBOL}_scaler_{interval_key}.pkl"

def try_load_model():
    try:
        model = EnsemblePredictor(15, ENSEMBLE_SIZE)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        scaler = joblib.load(scaler_path)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.model_trained = True
        st.session_state.training = False
        st.session_state.trained_at = datetime.now()
        return True
    except Exception as e:
        print("⚠️ Model load failed:", e)
        return False

if not st.session_state.model_trained and os.path.exists(model_path):
    try_load_model()

# ───── Retrain If Needed ─────────────────────
now = datetime.now()
last_train = st.session_state.trained_at or datetime.min
elapsed = (now - last_train).total_seconds() / 3600

if elapsed >= RETRAIN_INTERVAL_HOURS and not st.session_state.training:
    print(f"🔁 Retraining required (last trained {elapsed:.2f} hours ago)…")
    st.session_state.model_trained = False
    start_training(st.session_state, params)

# ───── Display Training Errors ────────────────
if st.session_state.training_error:
    st.error(f"❌ Training Error: {st.session_state.training_error}")

# ───── Fetch Live Data ───────────────────────
df = fetch_enhanced_data(SYMBOL, INTERVAL, LIVE_BARS)
if df.empty:
    st.warning("No live data.")
    st.stop()

df = detect_anomalies(df)
ha_df = calculate_heikin_ashi(df)
df = df.join(ha_df[["ha_open", "ha_high", "ha_low", "ha_close"]])

# ───── Training Status ───────────────────────
if st.session_state.training:
    progress_ph.progress(0.5, "⏳ Training in progress…")
    chart_slot.plotly_chart(display_live_chart(df, CANDLE_TYPE), use_container_width=True)
    st.stop()

# ───── Multi-Timeframe Prediction ────────────
pred, conf = multi_timeframe_prediction(SYMBOL, SEQ_LEN)
if pred is None:
    st.error("❌ Could not compute prediction using any timeframe.")
    st.stop()

status_ph.success("✅ Multi-timeframe model ready")

# ───── Display Predictions ───────────────────
price = df["close"].iloc[-1]
future_price = price * (1 + pred)

chart_slot.plotly_chart(display_live_chart(df, CANDLE_TYPE, prediction=future_price), use_container_width=True)

st.metric("📈 Predicted Price", f"${future_price:.2f}", delta=f"{pred*100:.2f}%")
st.metric("🔐 Confidence", f"{conf*100:.1f}%")

score, label, color, _ = analyze_market_sentiment(df)
risk = calculate_risk_metrics(df, pred, conf, params)
sigs = generate_trading_signals(df, pred, conf, risk, params)

st.subheader("💬 Market Sentiment")
st.markdown(f"<span style='color:{color}; font-weight:bold'>{label}: {score:.1f}</span>", unsafe_allow_html=True)

st.subheader("📢 Trading Signals")
for s in sigs:
    col = "#4CAF50" if s["type"] == "BUY" else "#F44336"
    st.markdown(f"<span style='color:{col}'><b>{s['type']} ({s['strength']}):</b> {s['reason']}</span>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;font-size:0.8em;'>© 2025 Advanced Crypto Predictor Pro</div>", unsafe_allow_html=True)

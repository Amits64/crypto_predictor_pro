import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd

import config
from data.fetcher import fetch_enhanced_data
from preprocessing.anomalies import detect_anomalies
from visualization.charts import display_live_chart
from training.trainer import start_training
from analysis.sentiment import analyze_market_sentiment
from analysis.signals import calculate_risk_metrics, generate_trading_signals

# UI Setup
st.set_page_config(config.PAGE_TITLE, layout=config.PAGE_LAYOUT)
st.markdown(config.APP_CSS, unsafe_allow_html=True)
st.markdown('<div class="app-header"><h1>'+config.PAGE_TITLE+'</h1></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    SYMBOL             = st.text_input("Binance Symbol", "BTCUSDT")
    INTERVAL           = st.selectbox("Interval", ["1m","5m","15m","30m","1h","4h","1d"])
    CANDLE_TYPE        = st.selectbox("Candle Type",["Regular","Heikin-Ashi"])
    LIVE_BARS          = st.slider("Live bars",100,500,200)
    HISTORY_BARS       = st.slider("History bars",500,3000,1000)
    REFRESH_SEC        = st.slider("Refresh (s)",1,60,10)
    SEQ_LEN            = st.slider("Seq length",50,300,100)
    ENSEMBLE_SIZE      = st.slider("Ensemble size",3,7,5)
    CONFIDENCE_THRESHOLD = st.slider("Confidence thres",0.5,0.95,0.8)
    RISK_TOLERANCE     = st.slider("Risk tol",0.1,2.0,1.0,0.1)
    STOP_LOSS_PCT      = st.slider("Stop loss %",1.0,10.0,3.0,0.5)
    TAKE_PROFIT_PCT    = st.slider("Take profit %",2.0,20.0,6.0,0.5)

# Auto-refresh
st_autorefresh(interval=REFRESH_SEC*1000, key="ticker")

# Session state defaults
for v in ["training","model_trained","model","scaler","train_losses","val_losses","trained_at","training_error"]:
    if v not in st.session_state: st.session_state[v] = False if v in ["training","model_trained"] else None

# Fetch data
df = fetch_enhanced_data(SYMBOL, INTERVAL, LIVE_BARS)
if df.empty:
    st.warning("⏳ Waiting for live data…")
    st.stop()

df = detect_anomalies(df)

# Plot
st.subheader("Real-time Market Data")
chart = display_live_chart(df, CANDLE_TYPE, prediction=None)
ph = st.empty()
ph.plotly_chart(chart, use_container_width=True)

# Start training
if not st.session_state.training and not st.session_state.model_trained:
    start_training(st.session_state, config)

# Training status
if st.session_state.training:
    st.info("🔄 Training in background…")
elif st.session_state.model_trained:
    st.success("🎯 Model Ready!")

    # Prediction
    pred, conf, _, _ = st.session_state.model(
        st.session_state.scaler.transform(df.tail(SEQ_LEN)[[
            "close","volume","SMA_10","SMA_20","EMA_12","EMA_26",
            "RSI","MACD","ATR","ADX","return_1","volatility","volume_ratio"
        ]].astype(float))
    )
    pred, conf = pred.item(), conf.item()
    price = df.close.iloc[-1]
    predicted_price = price * (1 + pred)

    # Update chart with pred
    ph.plotly_chart(display_live_chart(df, CANDLE_TYPE, predicted_price), use_container_width=True)

    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Predicted Price", f"${predicted_price:.2f}", delta=f"{pred*100:.2f}%")
    col2.metric("Confidence", f"{conf*100:.1f}%")

    # Sentiment & signals
    sent_score, sent_lbl, sent_col, comps = analyze_market_sentiment(df)
    risk = calculate_risk_metrics(df, pred, conf, config)
    sigs = generate_trading_signals(df, pred, conf, risk, config)

    st.subheader("Market Sentiment")
    st.markdown(f"<div style='color:{sent_col}'>{sent_lbl}: {sent_score:.1f}/100</div>", unsafe_allow_html=True)

    st.subheader("Trading Signals")
    for s in sigs:
        clr = "#4CAF50" if s["type"]=="BUY" else "#F44336"
        st.markdown(f"**{s['type']}** ({s['strength']}): {s['reason']}", unsafe_allow_html=True)

else:
    st.warning("⚠️ Model not ready yet…")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;font-size:0.8em;'>© 2025 Advanced Trading Systems</div>", unsafe_allow_html=True)

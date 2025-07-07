import os
import torch
import joblib
import numpy as np
import pandas as pd
from models.ensemble import EnsemblePredictor
from data.fetcher import fetch_enhanced_data
from preprocessing.anomalies import detect_anomalies
from preprocessing.indicators import calculate_heikin_ashi

INTERVAL_WEIGHTS = {
    "1m": 0.2,
    "5m": 0.5,
    "15m": 1.0,
    "1h": 3.0,
    "4h": 5.0,
    "1d": 10.0
}

def predict_for_interval(symbol, interval, seq_len, base_model_path):
    try:
        df = fetch_enhanced_data(symbol, interval, limit=seq_len + 50)
        if df is None or df.empty:
            return None, None, 0

        df = detect_anomalies(df)
        ha_df = calculate_heikin_ashi(df)
        df = df.join(ha_df[["ha_close"]])

        required_cols = [
            "close", "volume", "SMA_10", "SMA_20", "EMA_12", "EMA_26",
            "RSI", "MACD", "ATR", "ADX", "return_1", "volatility",
            "volume_ratio", "anomaly", "ha_close"
        ]
        if not all(col in df.columns for col in required_cols):
            return None, None, 0

        # Prepare input
        X = df.tail(seq_len)[required_cols].astype(float)
        X_df = pd.DataFrame(X, columns=required_cols)  # Preserve column names
        scaler_path = f"{base_model_path}/{symbol}_scaler_{interval}.pkl"
        model_path = f"{base_model_path}/{symbol}_model_{interval}.pt"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, 0

        # Load model & scaler
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_df)  # Suppresses feature name warning
        X_tensor = torch.tensor(X_scaled.reshape(1, seq_len, -1), dtype=torch.float32)

        model = EnsemblePredictor(input_dim=X_tensor.shape[2], ensemble_size=5)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            pred, _, P, _ = model(X_tensor)
            conf = model.get_confidence(P)

        return pred.item(), conf, df["volume"].iloc[-1]

    except Exception as e:
        print(f"❌ Prediction error for {symbol} ({interval}): {e}")
        return None, None, 0


def multi_timeframe_prediction(symbol, seq_len):
    base_model_path = f"models/trained_model/{symbol}"
    results = []

    for interval in INTERVAL_WEIGHTS:
        pred, conf, volume = predict_for_interval(symbol, interval, seq_len, base_model_path)
        if pred is not None:
            weight = INTERVAL_WEIGHTS[interval]
            score = weight * volume
            results.append((pred, conf, score))

    if not results:
        return None, None

    preds, confs, scores = zip(*results)
    total_score = sum(scores)
    weights = [s / total_score for s in scores]

    weighted_pred = sum(p * w for p, w in zip(preds, weights))
    weighted_conf = sum(c * w for c, w in zip(confs, weights))

    return weighted_pred, weighted_conf

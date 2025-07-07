import numpy as np

def analyze_market_sentiment(df):
    """Compute a simple fear/greed index from RSI, volume_ratio & volatility."""
    rsi   = df.RSI.iloc[-1]
    volr  = df.volume_ratio.iloc[-1]
    vol   = df.volatility.iloc[-1]
    trend = df.close.pct_change(5).iloc[-1]

    comp = {
        "momentum": 100 - rsi if rsi>50 else rsi,
        "volume":  min(100, volr*50),
        "volatility": max(0, 100 - vol*1000),
        "trend": (trend+1)*50
    }
    score = np.mean(list(comp.values()))

    if score>=75:
        label, color = "Extreme Greed 🔥", "#FF4444"
    elif score>=60:
        label, color = "Greed 📈", "#FF8800"
    elif score>=40:
        label, color = "Neutral ⚖️", "#FFBB00"
    elif score>=25:
        label, color = "Fear 📉", "#88FF88"
    else:
        label, color = "Extreme Fear 💚", "#00FF00"
    return score, label, color, comp

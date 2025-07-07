import pandas as pd
import talib
from streamlit import cache_data
from utils import safe_get

@cache_data(ttl=30)
def fetch_enhanced_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch kline data from Binance + compute TA indicators."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    raw = safe_get(url)
    cols = ["open_time","o","h","l","c","v","close_time","qa","tr","tba","tqa","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["datetime"] = pd.to_datetime(df.open_time, unit="ms")
    df[["open","high","low","close","volume"]] = df[["o","h","l","c","v"]].astype(float)

    # Technical Indicators
    df["SMA_10"] = talib.SMA(df.close, 10)
    df["SMA_20"] = talib.SMA(df.close, 20)
    df["EMA_12"] = talib.EMA(df.close, 12)
    df["EMA_26"] = talib.EMA(df.close, 26)
    df["RSI"]     = talib.RSI(df.close, 14)
    df["MACD"], df["MACD_signal"], _ = talib.MACD(df.close)
    df["ATR"]     = talib.ATR(df.high, df.low, df.close, 14)
    df["ADX"]     = talib.ADX(df.high, df.low, df.close, 14)

    # Price action features
    df["return_1"]      = df.close.pct_change(1)
    df["volatility"]    = df.return_1.rolling(20).std()
    df["volume_sma"]    = df.volume.rolling(20).mean()
    df["volume_ratio"]  = df.volume / df.volume_sma
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

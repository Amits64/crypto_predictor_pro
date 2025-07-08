import pandas as pd
import pandas_ta as ta
from streamlit import cache_data
from utils import safe_get

@cache_data(ttl=30)
def fetch_enhanced_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch kline data from Binance + compute TA indicators."""
    url = f"https://api.binance.me/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    raw = safe_get(url)
    cols = ["open_time", "o", "h", "l", "c", "v", "close_time", "qa", "tr", "tba", "tqa", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["datetime"] = pd.to_datetime(df.open_time, unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["o", "h", "l", "c", "v"]].astype(float)

    # Technical Indicators using pandas-ta
    df["SMA_10"] = ta.sma(df["close"], length=10)
    df["SMA_20"] = ta.sma(df["close"], length=20)
    df["EMA_12"] = ta.ema(df["close"], length=12)
    df["EMA_26"] = ta.ema(df["close"], length=26)

    rsi = ta.rsi(df["close"], length=14)
    df["RSI"] = rsi

    macd = ta.macd(df["close"])
    if not macd.empty:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
    else:
        df["MACD"] = df["MACD_signal"] = None

    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    # Price action features
    df["return_1"] = df["close"].pct_change(1)
    df["volatility"] = df["return_1"].rolling(20).std()
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    df.dropna(inplace=True)
    return df.reset_index(drop=True)

def fetch_multi_timeframe_data(symbol, intervals, history_size):
    dfs = {}
    for interval in intervals:
        df = fetch_enhanced_data(symbol, interval, history_size * 3)
        if not df.empty:
            # Preserve 'datetime' without suffix to allow merging
            columns = []
            for col in df.columns:
                if col == "datetime":
                    columns.append(col)
                else:
                    columns.append(f"{col}_{interval}")
            df.columns = columns
            dfs[interval] = df
    return dfs

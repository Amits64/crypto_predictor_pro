import pandas as pd

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi candle values."""
    ha = df.copy()
    ha["ha_close"] = (df.open + df.high + df.low + df.close) / 4
    ha["ha_open"] = 0.0
    for i in range(len(ha)):
        if i == 0:
            ha.at[i, "ha_open"] = (df.open.iloc[0] + df.close.iloc[0]) / 2
        else:
            ha.at[i, "ha_open"] = (ha.ha_open.iloc[i-1] + ha.ha_close.iloc[i-1]) / 2
    ha["ha_high"] = ha[["high","ha_open","ha_close"]].max(axis=1)
    ha["ha_low"]  = ha[["low","ha_open","ha_close"]].min(axis=1)
    return ha

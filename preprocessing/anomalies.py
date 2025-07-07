import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame, contamination=0.05) -> pd.DataFrame:
    """Mark anomalies using IsolationForest."""
    features = df[["return_1","volatility","volume_ratio"]].fillna(0)
    iso = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = iso.fit_predict(features)
    df["anomaly"] = (df["anomaly"] == -1).astype(int)
    return df

from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    # Dynamically detect columns for anomalies
    feature_cols = [col for col in df.columns if any(metric in col for metric in ["return_1", "volatility", "volume_ratio"])]

    if len(feature_cols) < 3:
        raise ValueError(f"❌ Missing expected anomaly columns. Found: {feature_cols}")

    features = df[feature_cols].fillna(0)
    clf = IsolationForest(contamination=0.01, random_state=42)
    scores = clf.fit_predict(features)
    df["anomaly_score"] = scores
    return df
import threading
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from data.fetcher import fetch_multi_timeframe_data
from preprocessing.indicators import calculate_heikin_ashi
from preprocessing.anomalies import detect_anomalies
from models.ensemble import EnsemblePredictor
from models.loss import robust_loss
from sklearn.preprocessing import RobustScaler
import joblib
import os
import traceback

def background_training(state, params, update_progress=None):
    try:
        print("🎯 Entered background_training()")

        symbol = params["SYMBOL"]
        intervals = params.get("MTF_INTERVALS", [params["INTERVAL"]])
        model_dir = f"models/trained_model/{symbol}"
        os.makedirs(model_dir, exist_ok=True)
        interval_key = '_'.join(intervals)
        model_path = f"{model_dir}/{symbol}_model_{interval_key}.pt"
        scaler_path = f"{model_dir}/{symbol}_scaler_{interval_key}.pkl"

        # Step 1: Fetch MTF Data
        dfs = fetch_multi_timeframe_data(symbol, intervals, params["HISTORY_BARS"] + 500)
        if not dfs:
            raise ValueError("❌ No data returned for MTF intervals.")

        df = dfs[intervals[0]]

        # Merge strictly first
        for interval in intervals[1:]:
            df = df.merge(dfs[interval], on="datetime", how="inner")

        print("✅ MTF Data merged (inner):", df.shape)

        # Fallback: use merge_asof if data is insufficient
        if len(df) < params["SEQ_LEN"] + 50:
            print(f"⚠️ Too few rows after inner merge: {len(df)}. Retrying with asof merge…")
            df = dfs[intervals[0]].sort_values("datetime")
            for interval in intervals[1:]:
                df = pd.merge_asof(
                    df,
                    dfs[interval].sort_values("datetime"),
                    on="datetime",
                    direction="nearest",
                    tolerance=pd.Timedelta("3min")
                )
            print("✅ MTF Data merged (asof):", df.shape)

        if len(df) < params["SEQ_LEN"] + 50:
            raise ValueError(f"❌ Not enough data after MTF merge. Only {len(df)} rows.")

        # Step 2: Anomaly detection
        df = detect_anomalies(df)

        # Step 3: Heikin-Ashi on base interval
        base = intervals[0]
        required_cols = [f"{c}_{base}" for c in ["open", "high", "low", "close"]]
        df_ha_input = df[required_cols].copy()
        df_ha_input.columns = ["open", "high", "low", "close"]
        ha_df = calculate_heikin_ashi(df_ha_input)
        df = df.join(ha_df[["ha_open", "ha_high", "ha_low", "ha_close"]])

        # Step 4: Feature Engineering
        feature_cols = [col for col in df.columns if col not in ["datetime"]]
        feats = df[feature_cols].ffill().bfill().astype(np.float32)
        print("🧩 Features processed.")

        # Step 5: Scaling
        scaler = RobustScaler().fit(feats)
        X_all = scaler.transform(feats)

        # Step 6: Sequence generation
        X, y = [], []
        close_col = f"close_{base}"
        for i in range(params["SEQ_LEN"], len(X_all) - 5):
            X.append(X_all[i - params["SEQ_LEN"]:i])
            future = df[close_col].iloc[i:i + 5].values
            curr = df[close_col].iloc[i - 1]
            y.append(((future - curr) / curr).mean())

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        if len(X) < 100:
            raise ValueError("❌ Too few samples after sequencing.")

        # Step 7: DataLoader
        split = int(0.8 * len(X))
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split])), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:])), batch_size=32)

        # Step 8: Model setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnsemblePredictor(X.shape[2], params["ENSEMBLE_SIZE"]).to(device)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"🔁 Loaded previous weights from {model_path}")

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

        best_val = float("inf")
        counter = 0
        patience = 10
        losses, val_losses = [], []

        for epoch in range(10):
            model.train()
            tot_train = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                pred, conf, _, _ = model(bx)
                loss = robust_loss(pred, by, conf)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot_train += loss.item()
            train_loss = tot_train / len(train_loader)
            losses.append(train_loss)

            model.eval()
            tot_val = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred, conf, _, _ = model(bx)
                    tot_val += robust_loss(pred, by, conf).item()
            val_loss = tot_val / len(val_loader)
            val_losses.append(val_loss)
            sched.step()

            print(f"📊 Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                torch.save(model.state_dict(), model_path)
                joblib.dump(scaler, scaler_path)
                print(f"💾 Saved improved model to {model_path}")
            else:
                counter += 1
                if counter >= patience:
                    print("⏹️ Early stopping.")
                    break

            if update_progress:
                update_progress(epoch + 1, losses, val_losses)

        model.load_state_dict(torch.load(model_path))
        model.eval()
        scaler = joblib.load(scaler_path)

        state.model = model
        state.scaler = scaler
        state.train_losses = losses
        state.val_losses = val_losses
        state.trained_at = datetime.now()
        state.model_trained = True
        state.training = False
        print("✅ Training complete.")

    except Exception as e:
        print("❌ Training failed:", str(e))
        traceback.print_exc()
        state.training_error = str(e)
        state.training = False

def start_training(state, params, update_progress=None):
    print("🚀 Launching training thread…")
    state.training = True
    state.model_trained = False

    def wrapped_training():
        try:
            background_training(state, params, update_progress)
        except ValueError as e:
            if "Too few samples after sequencing" in str(e) and params["SEQ_LEN"] > 16:
                old_seq = params["SEQ_LEN"]
                params["SEQ_LEN"] = max(16, old_seq // 2)
                print(f"⚠️ Retrying training with reduced SEQ_LEN: {old_seq} → {params['SEQ_LEN']}")
                background_training(state, params, update_progress)
            else:
                raise e

    thread = threading.Thread(target=wrapped_training, daemon=True)
    thread.start()

import threading
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from data.fetcher import fetch_enhanced_data
from preprocessing.indicators import calculate_heikin_ashi
from preprocessing.anomalies import detect_anomalies
from models.ensemble import EnsemblePredictor
from models.loss import robust_loss
from sklearn.preprocessing import RobustScaler


def background_training(state, symbol, interval, history_bars, seq_len, ensemble_size, config, progress_callback=None):
    """Trains the ensemble model in background thread with logs."""
    try:
        print("🎯 Entered background_training()")

        df = fetch_enhanced_data(symbol, interval, history_bars + 100)
        print("✅ Data fetched:", df.shape)

        if len(df) < seq_len + 50:
            state.training_error = "❌ Not enough data for training."
            state.training = False
            return

        df = detect_anomalies(df)
        print("🧠 Anomaly detection complete.")

        ha_df = calculate_heikin_ashi(df)
        df = df.join(ha_df[["ha_open", "ha_high", "ha_low", "ha_close"]])

        feature_cols = [
            "close", "volume",
            "SMA_10", "SMA_20", "EMA_12", "EMA_26",
            "RSI", "MACD", "ATR", "ADX",
            "return_1", "volatility", "volume_ratio",
            "anomaly", "ha_close"
        ]
        feats = df[feature_cols].ffill().bfill().astype(np.float32)
        print("🧩 Features processed.")

        scaler = RobustScaler().fit(feats)
        X_all = scaler.transform(feats)

        X, y = [], []
        for i in range(seq_len, len(X_all) - 5):
            X.append(X_all[i - seq_len:i])
            future = df.close.iloc[i:i+5].values
            curr = df.close.iloc[i - 1]
            y.append(((future - curr) / curr).mean())
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        if len(X) < 100:
            state.training_error = "❌ Too few samples after sequencing."
            state.training = False
            return

        split = int(0.8 * len(X))
        train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
        val_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnsemblePredictor(X.shape[2], ensemble_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

        best_val = float("inf")
        patience = 10
        counter = 0
        losses, val_losses = [], []

        for epoch in range(100):
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
            losses.append(tot_train / len(train_loader))

            model.eval()
            tot_val = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred, conf, _, _ = model(bx)
                    tot_val += robust_loss(pred, by, conf).item()
            val_losses.append(tot_val / len(val_loader))
            sched.step()

            if progress_callback:
                progress_callback((epoch + 1) / 100)

            print(f"📊 Epoch {epoch+1}: train={losses[-1]:.4f}, val={val_losses[-1]:.4f}")

            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    print("⏹️ Early stopping triggered.")
                    break

        model.load_state_dict(torch.load("best_model.pt"))
        model.eval()

        state.model = model
        state.scaler = scaler
        state.train_losses = losses
        state.val_losses = val_losses
        state.trained_at = datetime.now()
        state.model_trained = True
        state.training = False
        print("✅ Model training completed successfully.")

    except Exception as e:
        print("❌ Training failed:", str(e))
        state.training_error = str(e)
        state.training = False


def start_training(state, symbol, interval, history_bars, seq_len, ensemble_size, config, progress_callback=None):
    """Starts training in background daemon thread."""
    print("🚀 Launching training thread…")
    state.training = True
    thread = threading.Thread(
        target=background_training,
        args=(state, symbol, interval, history_bars, seq_len, ensemble_size, config, progress_callback),
        daemon=True
    )
    thread.start()

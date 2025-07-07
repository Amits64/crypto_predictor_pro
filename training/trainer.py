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

def background_training(state, config):
    """Thread target that trains the ensemble model."""
    try:
        # 1) Fetch raw data
        df = fetch_enhanced_data(
            config.SYMBOL, config.INTERVAL, config.HISTORY_BARS + 100
        )
        if len(df) < config.SEQ_LEN + 50:
            state.training_error = "Not enough data"
            return

        # 2) Anomaly detection
        df = detect_anomalies(df)

        # 3) Heikin‐Ashi transform (adds ha_open/ha_high/ha_low/ha_close)
        ha_df = calculate_heikin_ashi(df)
        # merge only the new HA cols
        df = df.join(ha_df[["ha_open", "ha_high", "ha_low", "ha_close"]])

        # 4) Feature columns now include anomaly and HA close
        feature_cols = [
            "close", "volume",
            "SMA_10", "SMA_20", "EMA_12", "EMA_26",
            "RSI", "MACD", "ATR", "ADX",
            "return_1", "volatility", "volume_ratio",
            "anomaly",    # new
            "ha_close"    # new
        ]

        # fill and type‐cast
        feats = df[feature_cols].ffill().bfill().astype(np.float32)

        # 5) Scale
        scaler = RobustScaler().fit(feats)
        X_all = scaler.transform(feats)

        # 6) Sequence / target assembly
        X, y = [], []
        for i in range(config.SEQ_LEN, len(X_all) - 5):
            X.append(X_all[i - config.SEQ_LEN:i])
            future = df.close.iloc[i : i + 5].values
            curr   = df.close.iloc[i - 1]
            y.append(((future - curr) / curr).mean())
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        # 7) Train/Val split
        split = int(0.8 * len(X))
        train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
        val_ds   = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=32)

        # 8) Model, optimizer, scheduler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = EnsemblePredictor(X.shape[2], config.ENSEMBLE_SIZE).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

        # 9) Training loop with early stopping
        best_val = float("inf")
        patience = 10
        counter  = 0
        losses, val_losses = [], []

        for epoch in range(100):
            # train
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

            # validate
            model.eval()
            tot_val = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred, conf, _, _ = model(bx)
                    tot_val += robust_loss(pred, by, conf).item()
            val_losses.append(tot_val / len(val_loader))
            sched.step()

            # early stop check
            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                counter = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    break

        # 10) Load best & finalize
        model.load_state_dict(torch.load("best_model.pt"))
        model.eval()

        # 11) Save into state
        state.model         = model
        state.scaler        = scaler
        state.train_losses  = losses
        state.val_losses    = val_losses
        state.trained_at    = datetime.now()
        state.model_trained = True
        state.training      = False

    except Exception as e:
        state.training_error = str(e)
        state.training       = False

def start_training(state, config):
    """Launch training in a daemon thread."""
    state.training = True
    t = threading.Thread(target=background_training, args=(state, config), daemon=True)
    t.start()

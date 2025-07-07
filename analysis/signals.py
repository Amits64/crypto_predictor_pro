def calculate_risk_metrics(df, pred_return, confidence, params):
    """Position sizing + SL/TP and R:R based on ATR and Kelly criterion."""
    price = df.close.iloc[-1]
    atr   = df.ATR.iloc[-1]
    vol   = df.volatility.iloc[-1]

    win_rate = 0.6
    avg_win  = abs(pred_return) or 0.02
    avg_loss = avg_win * 0.75

    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    kelly = max(0, min(kelly, 0.25))  # limit Kelly to 25%
    size  = kelly * confidence * params["RISK_TOLERANCE"]

    stop_loss_pct = params["STOP_LOSS_PCT"]
    take_profit_pct = params["TAKE_PROFIT_PCT"]

    slp = price - 2 * atr if pred_return > 0 else price + 2 * atr
    tp1 = price * (1 + take_profit_pct / 100) if pred_return > 0 else price * (1 - take_profit_pct / 100)

    rr = abs(tp1 - price) / abs(price - slp) if slp != price else 0

    return {
        "position_size": size,
        "stop_loss": slp,
        "take_profit": tp1,
        "risk_reward": rr,
        "kelly": kelly
    }

def generate_trading_signals(df, pred_return, confidence, risk_metrics, params):
    """Return list of BUY/SELL signals with strength and reasons."""
    signals = []

    if confidence >= params["CONFIDENCE_THRESHOLD"]:
        if pred_return > 0.001:
            signals.append({
                "type": "BUY",
                "strength": min(5, int(confidence * 5)),
                "reason": f"AI predicts {pred_return * 100:.2f}% return at {confidence * 100:.1f}% confidence"
            })
        elif pred_return < -0.001:
            signals.append({
                "type": "SELL",
                "strength": min(5, int(confidence * 5)),
                "reason": f"AI predicts {pred_return * 100:.2f}% return at {confidence * 100:.1f}% confidence"
            })

    # RSI/MACD filter
    rsi = df.RSI.iloc[-1]
    macd, macd_sig = df.MACD.iloc[-1], df.MACD_signal.iloc[-1]

    if rsi < 30 and macd > macd_sig:
        signals.append({
            "type": "BUY",
            "strength": 3,
            "reason": f"RSI oversold ({rsi:.1f}) + MACD↑"
        })
    if rsi > 70 and macd < macd_sig:
        signals.append({
            "type": "SELL",
            "strength": 3,
            "reason": f"RSI overbought ({rsi:.1f}) + MACD↓"
        })

    # volume confirmation
    if df.volume_ratio.iloc[-1] > 1.5 and signals:
        signals[-1]["strength"] = min(5, signals[-1]["strength"] + 1)
        signals[-1]["reason"] += " + high volume"

    # poor R:R penalty
    if risk_metrics["risk_reward"] < 2:
        for s in signals:
            s["strength"] = max(1, s["strength"] - 1)
            s["reason"] += " (low R:R)"

    return signals

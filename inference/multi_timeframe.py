import os
import torch
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from models.ensemble import EnsemblePredictor
from data.fetcher import fetch_enhanced_data
from preprocessing.anomalies import detect_anomalies
from preprocessing.indicators import calculate_heikin_ashi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERVAL_WEIGHTS = {
    "1m": 0.2,
    "5m": 0.5,
    "15m": 1.0,
    "1h": 3.0,
    "4h": 5.0,
    "1d": 10.0
}

def predict_for_interval(symbol: str, interval: str, seq_len: int, base_model_path: str) -> Tuple[Optional[float], Optional[float], float]:
    """
    Generate prediction for a specific interval.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1m', '5m', '1h')
        seq_len: Sequence length for model input
        base_model_path: Base path for model files
        
    Returns:
        Tuple of (prediction, confidence, volume)
    """
    try:
        logger.info(f"Predicting for {symbol} on {interval} interval")
        
        # Fetch data with retry logic
        df = fetch_enhanced_data(symbol, interval, limit=seq_len + 50)
        if df is None or df.empty:
            logger.warning(f"No data fetched for {symbol} {interval}")
            return None, None, 0.0
        
        # Detect anomalies
        try:
            df = detect_anomalies(df)
        except Exception as e:
            logger.warning(f"Anomaly detection failed for {symbol} {interval}: {e}")
            df['anomaly'] = 0  # Default anomaly column
        
        # Calculate Heikin-Ashi
        try:
            ha_df = calculate_heikin_ashi(df)
            df = df.join(ha_df[["ha_close"]])
        except Exception as e:
            logger.warning(f"Heikin-Ashi calculation failed for {symbol} {interval}: {e}")
            df['ha_close'] = df['close']  # Fallback to regular close
        
        # Define required columns
        required_cols = [
            "close", "volume", "SMA_10", "SMA_20", "EMA_12", "EMA_26",
            "RSI", "MACD", "ATR", "ADX", "return_1", "volatility",
            "volume_ratio", "anomaly", "ha_close"
        ]
        
        # Check for missing columns and handle them
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {symbol} {interval}: {missing_cols}")
            # Fill missing columns with default values
            for col in missing_cols:
                if col == 'anomaly':
                    df[col] = 0
                elif col == 'ha_close':
                    df[col] = df['close']
                else:
                    df[col] = df['close']  # Fallback to close price
        
        # Prepare input data
        X = df.tail(seq_len)[required_cols].astype(float)
        
        # Handle NaN values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if X.empty or len(X) < seq_len:
            logger.warning(f"Insufficient data for {symbol} {interval}: got {len(X)}, need {seq_len}")
            return None, None, 0.0
        
        X_df = pd.DataFrame(X, columns=required_cols)
        
        # Load scaler and model
        scaler_path = f"{base_model_path}/{symbol}_scaler_{interval}.pkl"
        model_path = f"{base_model_path}/{symbol}_model_{interval}.pt"
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return None, None, 0.0
            
        if not os.path.exists(scaler_path):
            logger.warning(f"Scaler not found: {scaler_path}")
            return None, None, 0.0
        
        # Load scaler and transform data
        try:
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X_df)
        except Exception as e:
            logger.error(f"Failed to load/apply scaler for {symbol} {interval}: {e}")
            return None, None, 0.0
        
        # Prepare tensor input
        X_tensor = torch.tensor(X_scaled.reshape(1, seq_len, -1), dtype=torch.float32)
        
        # Load and run model
        try:
            model = EnsemblePredictor(input_dim=X_tensor.shape[2], ensemble_size=5)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            with torch.no_grad():
                output = model(X_tensor)
                
                # Handle different output formats
                if isinstance(output, tuple) and len(output) >= 3:
                    pred, _, P = output[0], output[1], output[2]
                    conf = model.get_confidence(P)
                elif isinstance(output, tuple) and len(output) == 2:
                    pred, conf = output
                else:
                    pred = output
                    conf = 0.5  # Default confidence
                
                # Extract scalar values
                pred_value = pred.item() if hasattr(pred, 'item') else float(pred)
                conf_value = conf.item() if hasattr(conf, 'item') else float(conf)
                
                # Get volume
                volume = df["volume"].iloc[-1] if not df["volume"].empty else 0.0
                
                logger.info(f"Prediction for {symbol} {interval}: {pred_value:.4f}, confidence: {conf_value:.4f}")
                return pred_value, conf_value, volume
                
        except Exception as e:
            logger.error(f"Model inference failed for {symbol} {interval}: {e}")
            return None, None, 0.0
            
    except Exception as e:
        logger.error(f"Prediction error for {symbol} ({interval}): {e}")
        return None, None, 0.0


def multi_timeframe_prediction(symbol: str, seq_len: int = 60) -> Dict[str, Any]:
    """
    Generate multi-timeframe predictions with enhanced error handling and metadata.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        seq_len: Sequence length for model input
        
    Returns:
        Dictionary containing prediction results and metadata
    """
    try:
        logger.info(f"Starting multi-timeframe prediction for {symbol}")
        
        base_model_path = f"models/trained_model/{symbol}"
        
        # Check if model directory exists
        if not os.path.exists(base_model_path):
            logger.error(f"Model directory not found: {base_model_path}")
            return _create_empty_result(symbol, f"Model directory not found: {base_model_path}")
        
        results = []
        interval_details = {}
        
        # Process each interval
        for interval in INTERVAL_WEIGHTS:
            pred, conf, volume = predict_for_interval(symbol, interval, seq_len, base_model_path)
            
            interval_details[interval] = {
                'prediction': pred,
                'confidence': conf,
                'volume': volume,
                'weight': INTERVAL_WEIGHTS[interval],
                'success': pred is not None
            }
            
            if pred is not None:
                weight = INTERVAL_WEIGHTS[interval]
                score = weight * volume
                results.append((pred, conf, score, interval))
                logger.info(f"✓ {interval}: pred={pred:.4f}, conf={conf:.4f}, score={score:.2f}")
            else:
                logger.warning(f"✗ {interval}: prediction failed")
        
        if not results:
            logger.error(f"No successful predictions for {symbol}")
            return _create_empty_result(symbol, "No successful predictions from any interval")
        
        # Calculate weighted prediction
        preds, confs, scores, intervals = zip(*results)
        total_score = sum(scores)
        
        if total_score == 0:
            logger.warning("Total score is zero, using equal weights")
            weights = [1.0 / len(results)] * len(results)
        else:
            weights = [s / total_score for s in scores]
        
        weighted_pred = sum(p * w for p, w in zip(preds, weights))
        weighted_conf = sum(c * w for c, w in zip(confs, weights))
        
        # Calculate additional metrics
        pred_std = np.std(preds) if len(preds) > 1 else 0.0
        conf_std = np.std(confs) if len(confs) > 1 else 0.0
        
        # Generate trading signals
        signals = _generate_trading_signals(weighted_pred, weighted_conf, pred_std)
        
        # Create comprehensive result
        result = {
            'success': True,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'prediction': weighted_pred,
            'confidence': weighted_conf,
            'prediction_std': pred_std,
            'confidence_std': conf_std,
            'intervals_used': list(intervals),
            'total_intervals': len(INTERVAL_WEIGHTS),
            'success_rate': len(results) / len(INTERVAL_WEIGHTS),
            'interval_details': interval_details,
            'signals': signals,
            'weights': dict(zip(intervals, weights)),
            'raw_predictions': list(preds),
            'raw_confidences': list(confs)
        }
        
        logger.info(f"✓ Multi-timeframe prediction complete for {symbol}: {weighted_pred:.4f} (conf: {weighted_conf:.4f})")
        return result
        
    except Exception as e:
        logger.error(f"Multi-timeframe prediction failed for {symbol}: {e}")
        return _create_empty_result(symbol, f"Prediction failed: {str(e)}")


def _generate_trading_signals(prediction: float, confidence: float, prediction_std: float) -> Dict[str, Any]:
    """Generate trading signals based on prediction and confidence."""
    try:
        signals = {
            'buy': False,
            'sell': False,
            'hold': True,
            'signal_strength': 0.0,
            'reason': 'Insufficient confidence or unclear signal'
        }
        
        # Confidence threshold
        min_confidence = 0.6
        
        # Signal strength threshold
        min_signal_strength = 0.02  # 2% minimum change
        
        if confidence < min_confidence:
            signals['reason'] = f'Low confidence ({confidence:.2f} < {min_confidence})'
            return signals
        
        # Calculate signal strength (normalized prediction adjusted for uncertainty)
        signal_strength = abs(prediction) * confidence / (1 + prediction_std)
        
        if signal_strength < min_signal_strength:
            signals['reason'] = f'Weak signal strength ({signal_strength:.4f} < {min_signal_strength})'
            return signals
        
        # Generate signals
        if prediction > min_signal_strength:
            signals.update({
                'buy': True,
                'sell': False,
                'hold': False,
                'signal_strength': signal_strength,
                'reason': f'Strong buy signal (pred: {prediction:.4f}, conf: {confidence:.2f})'
            })
        elif prediction < -min_signal_strength:
            signals.update({
                'buy': False,
                'sell': True,
                'hold': False,
                'signal_strength': signal_strength,
                'reason': f'Strong sell signal (pred: {prediction:.4f}, conf: {confidence:.2f})'
            })
        else:
            signals['reason'] = f'Neutral signal (pred: {prediction:.4f})'
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        return {
            'buy': False,
            'sell': False,
            'hold': True,
            'signal_strength': 0.0,
            'reason': f'Error generating signals: {str(e)}'
        }


def _create_empty_result(symbol: str, error_message: str) -> Dict[str, Any]:
    """Create empty result structure for failed predictions."""
    return {
        'success': False,
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'prediction': None,
        'confidence': None,
        'prediction_std': 0.0,
        'confidence_std': 0.0,
        'intervals_used': [],
        'total_intervals': len(INTERVAL_WEIGHTS),
        'success_rate': 0.0,
        'interval_details': {},
        'signals': {
            'buy': False,
            'sell': False,
            'hold': True,
            'signal_strength': 0.0,
            'reason': error_message
        },
        'weights': {},
        'raw_predictions': [],
        'raw_confidences': [],
        'error': error_message
    }


def get_prediction_summary(symbol: str, seq_len: int = 60) -> str:
    """
    Get a human-readable summary of the multi-timeframe prediction.
    
    Args:
        symbol: Trading symbol
        seq_len: Sequence length for model input
        
    Returns:
        Formatted string summary
    """
    try:
        result = multi_timeframe_prediction(symbol, seq_len)
        
        if not result['success']:
            return f"❌ Prediction failed for {symbol}: {result.get('error', 'Unknown error')}"
        
        summary = f"""
🔮 Multi-Timeframe Prediction for {symbol}
{'='*50}
📊 Prediction: {result['prediction']:.4f}
🎯 Confidence: {result['confidence']:.2f}
📈 Success Rate: {result['success_rate']:.1%} ({len(result['intervals_used'])}/{result['total_intervals']} intervals)
⏰ Timestamp: {result['timestamp']}

🔔 Trading Signals:
• Buy: {'✅' if result['signals']['buy'] else '❌'}
• Sell: {'✅' if result['signals']['sell'] else '❌'}
• Hold: {'✅' if result['signals']['hold'] else '❌'}
• Signal Strength: {result['signals']['signal_strength']:.4f}
• Reason: {result['signals']['reason']}

📋 Interval Details:
"""
        
        for interval, details in result['interval_details'].items():
            status = "✅" if details['success'] else "❌"
            pred_str = f"{details['prediction']:.4f}" if details['prediction'] is not None else "N/A"
            conf_str = f"{details['confidence']:.2f}" if details['confidence'] is not None else "N/A"
            summary += f"• {interval}: {status} pred={pred_str}, conf={conf_str}\n"
        
        return summary
        
    except Exception as e:
        return f"❌ Error generating prediction summary: {str(e)}"


# Legacy compatibility function
def legacy_multi_timeframe_prediction(symbol: str, seq_len: int = 60) -> Tuple[Optional[float], Optional[float]]:
    """
    Legacy function that returns only prediction and confidence for backward compatibility.
    
    Args:
        symbol: Trading symbol
        seq_len: Sequence length
        
    Returns:
        Tuple of (prediction, confidence) or (None, None) if failed
    """
    try:
        result = multi_timeframe_prediction(symbol, seq_len)
        
        if result['success']:
            return result['prediction'], result['confidence']
        else:
            return None, None
            
    except Exception as e:
        logger.error(f"Legacy prediction failed for {symbol}: {e}")
        return None, None

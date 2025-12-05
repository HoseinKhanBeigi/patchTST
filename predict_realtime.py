"""
Real-time Bitcoin Price Prediction using PatchTST

This script continuously fetches live Bitcoin data and makes predictions.
"""

import pandas as pd
import numpy as np
import torch
from transformers import PatchTSTForPrediction
import os
import time
from datetime import datetime
import ccxt
from alert_handler import AlertHandler


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram


def fetch_latest_data(exchange_name='bybit', timeframe='5m', limit=200):
    """
    Fetch the latest Bitcoin data from an exchange.
    """
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange_config = {
            'enableRateLimit': True,
        }
        
        # Add exchange-specific options
        if exchange_name == 'binance':
            exchange_config['options'] = {'defaultType': 'spot'}
        elif exchange_name == 'coinbasepro':
            exchange_config['options'] = {'sandbox': False}
        
        exchange = exchange_class(exchange_config)
        
        # Determine symbol
        symbol = 'BTC/USDT'
        if exchange_name == 'coinbasepro':
            symbol = 'BTC/USD'
        elif exchange_name == 'kraken':
            symbol = 'BTC/USD'
        elif exchange_name == 'bitfinex':
            symbol = 'BTC/USD'
        
        # Fetch latest OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) == 0:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        print(f"Error fetching data from {exchange_name}: {e}")
        return None


def fetch_latest_data_with_fallback(timeframe='5m', limit=200):
    """
    Try multiple exchanges in order until one works.
    """
    # Try exchanges in order of preference (most reliable first)
    exchanges_to_try = [
        'bybit',      # Usually works well
        'kraken',     # Reliable
        'coinbasepro', # Coinbase Pro
        'bitfinex',   # Bitfinex
        'gate',       # Gate.io
        'huobi',      # Huobi
        'okx',        # OKX (may be blocked)
        'binance',    # Binance (may be geo-blocked)
    ]
    
    for exchange_name in exchanges_to_try:
        print(f"Trying {exchange_name}...", end=' ')
        df = fetch_latest_data(exchange_name, timeframe, limit)
        if df is not None and len(df) > 0:
            print(f"✓ Success!")
            return df, exchange_name
        print("Failed")
    
    return None, None


def load_and_preprocess_data(df):
    """
    Add technical indicators to the data.
    """
    df = df.copy()
    df['RSI_14'] = calculate_rsi(df['close'], period=14)
    df['MACDh_12_26_9'] = calculate_macd(df['close'], fast=12, slow=26, signal=9)
    df = df.dropna()
    return df


def prepare_prediction_input(df, context_length=128, features=['close', 'high', 'low', 'open', 'volume', 'RSI_14', 'MACDh_12_26_9']):
    """
    Prepare the most recent data for prediction.
    """
    recent_data = df[features].iloc[-context_length:].values
    past_values = recent_data.T.T  # Ensure correct shape
    return torch.tensor(past_values, dtype=torch.float32).unsqueeze(0)


def make_predictions(model, past_values, prediction_length=6):
    """
    Make predictions using the trained model.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(past_values=past_values)
        predictions = outputs.prediction_outputs
    return predictions.squeeze(0).cpu().numpy()


def update_csv_with_latest(csv_file, new_df):
    """
    Update CSV file with latest data, keeping only recent data.
    """
    try:
        # Load existing data
        existing_df = pd.read_csv(csv_file)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        existing_df.set_index('timestamp', inplace=True)
        
        # Combine and remove duplicates
        combined_df = pd.concat([existing_df, new_df], ignore_index=False)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        # Keep only last 2 years to manage file size
        if len(combined_df) > 200000:  # ~2 years of 5-min data
            combined_df = combined_df.iloc[-200000:]
        
        # Save
        combined_df.to_csv(csv_file)
        return combined_df
    except Exception as e:
        print(f"Error updating CSV: {e}")
        return new_df


def analyze_signal(current_price, predictions, features):
    """
    Analyze predictions to determine buy/sell signal strength.
    Returns: (signal_type, signal_strength, confidence, reasons)
    """
    next_5min_pred = predictions[0]
    predicted_close_5min = next_5min_pred[0]
    change_5min = predicted_close_5min - current_price
    change_pct_5min = (change_5min / current_price) * 100
    
    # Get feature indices
    close_idx = features.index('close')
    high_idx = features.index('high')
    low_idx = features.index('low')
    volume_idx = features.index('volume')
    rsi_idx = features.index('RSI_14')
    macd_idx = features.index('MACDh_12_26_9')
    
    # Extract values
    predicted_high = next_5min_pred[high_idx]
    predicted_low = next_5min_pred[low_idx]
    predicted_rsi = next_5min_pred[rsi_idx]
    predicted_macd = next_5min_pred[macd_idx]
    predicted_volume = next_5min_pred[volume_idx]
    
    # Calculate momentum (price change over 30 min)
    predicted_close_30min = predictions[5, 0]
    change_30min = predicted_close_30min - current_price
    change_pct_30min = (change_30min / current_price) * 100
    
    # Signal scoring system
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # 1. Price momentum (5 min)
    if change_pct_5min > 0.1:
        buy_score += 3
        reasons.append(f"Strong upward momentum (+{change_pct_5min:.2f}% in 5min)")
    elif change_pct_5min > 0.05:
        buy_score += 2
        reasons.append(f"Moderate upward momentum (+{change_pct_5min:.2f}% in 5min)")
    elif change_pct_5min > 0:
        buy_score += 1
        reasons.append(f"Slight upward movement (+{change_pct_5min:.2f}% in 5min)")
    elif change_pct_5min < -0.1:
        sell_score += 3
        reasons.append(f"Strong downward momentum ({change_pct_5min:.2f}% in 5min)")
    elif change_pct_5min < -0.05:
        sell_score += 2
        reasons.append(f"Moderate downward momentum ({change_pct_5min:.2f}% in 5min)")
    elif change_pct_5min < 0:
        sell_score += 1
        reasons.append(f"Slight downward movement ({change_pct_5min:.2f}% in 5min)")
    
    # 2. RSI analysis
    if predicted_rsi < 30:
        buy_score += 2
        reasons.append(f"RSI oversold ({predicted_rsi:.1f}) - potential bounce")
    elif predicted_rsi < 40:
        buy_score += 1
        reasons.append(f"RSI below neutral ({predicted_rsi:.1f})")
    elif predicted_rsi > 70:
        sell_score += 2
        reasons.append(f"RSI overbought ({predicted_rsi:.1f}) - potential pullback")
    elif predicted_rsi > 60:
        sell_score += 1
        reasons.append(f"RSI above neutral ({predicted_rsi:.1f})")
    
    # 3. MACD analysis
    if predicted_macd > 20:
        buy_score += 2
        reasons.append(f"MACD bullish ({predicted_macd:.2f})")
    elif predicted_macd > 0:
        buy_score += 1
        reasons.append(f"MACD positive ({predicted_macd:.2f})")
    elif predicted_macd < -20:
        sell_score += 2
        reasons.append(f"MACD bearish ({predicted_macd:.2f})")
    elif predicted_macd < 0:
        sell_score += 1
        reasons.append(f"MACD negative ({predicted_macd:.2f})")
    
    # 4. Price range analysis (high-low spread)
    price_range = predicted_high - predicted_low
    range_pct = (price_range / current_price) * 100
    if range_pct > 0.15:  # High volatility
        if predicted_close_5min > current_price:
            buy_score += 1
            reasons.append(f"High volatility with upward bias")
        else:
            sell_score += 1
            reasons.append(f"High volatility with downward bias")
    
    # 5. 30-minute trend
    if change_pct_30min > 0.2:
        buy_score += 2
        reasons.append(f"Strong 30-min trend (+{change_pct_30min:.2f}%)")
    elif change_pct_30min > 0.1:
        buy_score += 1
        reasons.append(f"Positive 30-min trend (+{change_pct_30min:.2f}%)")
    elif change_pct_30min < -0.2:
        sell_score += 2
        reasons.append(f"Strong 30-min downtrend ({change_pct_30min:.2f}%)")
    elif change_pct_30min < -0.1:
        sell_score += 1
        reasons.append(f"Negative 30-min trend ({change_pct_30min:.2f}%)")
    
    # Determine signal
    signal_diff = buy_score - sell_score
    
    if signal_diff >= 5:
        signal_type = "STRONG BUY"
        signal_strength = signal_diff
        confidence = "HIGH"
    elif signal_diff >= 3:
        signal_type = "BUY"
        signal_strength = signal_diff
        confidence = "MEDIUM"
    elif signal_diff >= 1:
        signal_type = "WEAK BUY"
        signal_strength = signal_diff
        confidence = "LOW"
    elif signal_diff <= -5:
        signal_type = "STRONG SELL"
        signal_strength = abs(signal_diff)
        confidence = "HIGH"
    elif signal_diff <= -3:
        signal_type = "SELL"
        signal_strength = abs(signal_diff)
        confidence = "MEDIUM"
    elif signal_diff <= -1:
        signal_type = "WEAK SELL"
        signal_strength = abs(signal_diff)
        confidence = "LOW"
    else:
        signal_type = "NEUTRAL"
        signal_strength = 0
        confidence = "LOW"
    
    return signal_type, signal_strength, confidence, reasons


def display_prediction(current_price, predictions, features, timestamp):
    """
    Display the prediction results with buy/sell signals.
    """
    # Clear screen (works on Unix/Mac)
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 70)
    print("REAL-TIME BITCOIN PRICE PREDICTION")
    print("=" * 70)
    print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Analyze signal
    signal_type, signal_strength, confidence, reasons = analyze_signal(
        current_price, predictions, features
    )
    
    # Next 5 minutes
    next_5min_pred = predictions[0]
    predicted_close_5min = next_5min_pred[0]
    change_5min = predicted_close_5min - current_price
    change_pct_5min = (change_5min / current_price) * 100
    
    # Display signal prominently
    print("=" * 70)
    print("TRADING SIGNAL (5-MINUTE TIMEFRAME)")
    print("=" * 70)
    
    # Color coding (using text formatting)
    if "STRONG BUY" in signal_type:
        signal_display = f"🟢 {signal_type}"
    elif "BUY" in signal_type:
        signal_display = f"🟢 {signal_type}"
    elif "STRONG SELL" in signal_type:
        signal_display = f"🔴 {signal_type}"
    elif "SELL" in signal_type:
        signal_display = f"🔴 {signal_type}"
    else:
        signal_display = f"🟡 {signal_type}"
    
    print(f"Signal:        {signal_display}")
    print(f"Strength:      {signal_strength}/10")
    print(f"Confidence:    {confidence}")
    print()
    
    print("Signal Reasons:")
    print("-" * 70)
    for reason in reasons[:5]:  # Show top 5 reasons
        print(f"  • {reason}")
    print()
    
    print("=" * 70)
    print("NEXT 5 MINUTES PREDICTION")
    print("=" * 70)
    print(f"Current Price:     ${current_price:,.2f}")
    print(f"Predicted Price:   ${predicted_close_5min:,.2f}")
    print(f"Expected Change:   ${change_5min:,.2f} ({change_pct_5min:+.2f}%)")
    print()
    
    print("Predicted Values:")
    print("-" * 70)
    for j, feature in enumerate(features):
        value = next_5min_pred[j]
        if feature in ['close', 'high', 'low', 'open']:
            print(f"  {feature:15s}: ${value:,.2f}")
        elif feature == 'volume':
            print(f"  {feature:15s}: {value:,.0f}")
        else:
            print(f"  {feature:15s}: {value:,.4f}")
    print()
    
    # 30-minute forecast summary
    print("=" * 70)
    print("30-MINUTE FORECAST")
    print("=" * 70)
    for i in range(6):
        pred_close = predictions[i, 0]
        period_change = pred_close - current_price
        period_change_pct = (period_change / current_price) * 100
        minutes = (i + 1) * 5
        print(f"  +{minutes:2d} min: ${pred_close:,.2f} ({period_change_pct:+.2f}%)")
    print()
    
    predicted_close_30min = predictions[5, 0]
    change_30min = predicted_close_30min - current_price
    change_pct_30min = (change_30min / current_price) * 100
    
    print(f"30-min Target: ${predicted_close_30min:,.2f} ({change_pct_30min:+.2f}%)")
    print()
    print("=" * 70)
    print("Press Ctrl+C to stop")
    print("=" * 70)


def main():
    """
    Main real-time prediction loop.
    """
    print("=" * 70)
    print("Real-Time Bitcoin Price Prediction")
    print("=" * 70)
    print()
    
    # Load model
    model_path = "./vast_outputs/patchtst_bitcoin_output/final_model"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print("Loading model...")
    model = PatchTSTForPrediction.from_pretrained(model_path)
    print("✓ Model loaded")
    print()
    
    # Initialize alert handler
    alert_handler = AlertHandler()
    if alert_handler.enabled:
        print("✓ Telegram alerts enabled")
        print(f"  Chat IDs: {alert_handler.chat_ids}")
    else:
        print("⚠ Telegram alerts disabled")
    print()
    
    # Try to load existing CSV or fetch fresh data
    csv_file = "btc_5m_5years.csv"
    df = None
    
    if os.path.exists(csv_file):
        print(f"Loading existing data from {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            print(f"✓ Loaded {len(df)} historical samples")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            df = None
    
    # Configuration
    context_length = 128
    features = ['close', 'high', 'low', 'open', 'volume', 'RSI_14', 'MACDh_12_26_9']
    update_interval = 300  # Update every 5 minutes (300 seconds)
    
    print()
    print("Starting real-time predictions...")
    print(f"Update interval: {update_interval} seconds (5 minutes)")
    print("Fetching live data from exchange...")
    print()
    
    current_exchange = None
    
    try:
        while True:
            # Fetch latest data (try multiple exchanges)
            print("Fetching latest data...", end=' ')
            live_df, exchange_used = fetch_latest_data_with_fallback(timeframe='5m', limit=200)
            
            if live_df is None or len(live_df) == 0:
                print("All exchanges failed. Retrying in 60 seconds...")
                time.sleep(60)
                continue
            
            if exchange_used != current_exchange:
                current_exchange = exchange_used
                print(f"Using exchange: {current_exchange}")
            
            print(f"✓ Got {len(live_df)} candles")
            
            # Update CSV if we have existing data
            if df is not None:
                df = update_csv_with_latest(csv_file, live_df)
            else:
                df = live_df
            
            # Preprocess
            df_processed = load_and_preprocess_data(df)
            
            if len(df_processed) < context_length:
                print(f"Not enough data (need {context_length}, have {len(df_processed)}). Waiting...")
                time.sleep(60)
                continue
            
            # Prepare input and predict
            past_values = prepare_prediction_input(df_processed, context_length=context_length, features=features)
            predictions = make_predictions(model, past_values, prediction_length=6)
            
            # Display results
            current_price = df_processed['close'].iloc[-1]
            last_timestamp = df_processed.index[-1]
            
            # Analyze signal for alerts
            signal_type, signal_strength, confidence, reasons = analyze_signal(
                current_price, predictions, features
            )
            
            # Send Telegram alert if needed
            next_5min_pred = predictions[0]
            predicted_close_5min = next_5min_pred[0]
            change_5min = predicted_close_5min - current_price
            change_pct_5min = (change_5min / current_price) * 100
            
            # Always try to send alert (handler will check if it should)
            try:
                alert_sent = alert_handler.send_signal_alert(
                    signal_type, signal_strength, confidence,
                    current_price, predicted_close_5min, change_pct_5min,
                    reasons, last_timestamp
                )
                if alert_sent:
                    print(f"\n📱 Telegram alert sent for {signal_type}")
            except Exception as e:
                print(f"\n⚠ Error sending Telegram alert: {e}")
            
            # Display results
            display_prediction(current_price, predictions, features, last_timestamp)
            
            # Wait before next update
            print(f"\nNext update in {update_interval} seconds...")
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


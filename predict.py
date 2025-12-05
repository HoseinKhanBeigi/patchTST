"""
Use the trained PatchTST model to make predictions on Bitcoin data.

This script loads the trained model and makes predictions for future price movements.
"""

import pandas as pd
import numpy as np
import torch
from transformers import PatchTSTForPrediction
import os


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


def load_and_preprocess_data(file_path):
    """
    Load Bitcoin data and add technical indicators.
    """
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Add technical indicators (without pandas-ta)
    df['RSI_14'] = calculate_rsi(df['close'], period=14)
    df['MACDh_12_26_9'] = calculate_macd(df['close'], fast=12, slow=26, signal=9)
    
    df = df.dropna()
    return df


def prepare_prediction_input(df, context_length=128, features=['close', 'high', 'low', 'open', 'volume', 'RSI_14', 'MACDh_12_26_9']):
    """
    Prepare the most recent data for prediction.
    Returns the last context_length periods as input.
    """
    # Get the last context_length periods
    recent_data = df[features].iloc[-context_length:].values  # Shape: (context_length, num_features)
    
    # Transpose to (num_features, context_length) then to (context_length, num_features)
    past_values = recent_data.T  # (num_features, context_length)
    past_values = past_values.T  # (context_length, num_features) - correct format
    
    return torch.tensor(past_values, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


def make_predictions(model, past_values, prediction_length=6):
    """
    Make predictions using the trained model.
    """
    model.eval()
    
    with torch.no_grad():
        # Model expects past_values shape: (batch_size, sequence_length, num_features)
        outputs = model(past_values=past_values)
        
        # Get predictions
        predictions = outputs.prediction_outputs  # Shape: (batch_size, prediction_length, num_features)
        
    return predictions.squeeze(0).cpu().numpy()  # Remove batch dimension


def main():
    """
    Main prediction function.
    """
    print("=" * 60)
    print("PatchTST Bitcoin Price Prediction")
    print("=" * 60)
    print()
    
    # Load model
    model_path = "./vast_outputs/patchtst_bitcoin_output/final_model"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please make sure you've downloaded the model.")
        return
    
    print(f"Loading model from {model_path}...")
    model = PatchTSTForPrediction.from_pretrained(model_path)
    print("✓ Model loaded successfully")
    print()
    
    # Load data
    data_file = "btc_5m_5years.csv"
    if not os.path.exists(data_file):
        print(f"Warning: Data file {data_file} not found.")
        print("Please provide the path to your Bitcoin data CSV file.")
        data_file = input("Enter path to CSV file (or press Enter to skip): ").strip()
        if not data_file or not os.path.exists(data_file):
            print("Cannot proceed without data file.")
            return
    
    print(f"Loading data from {data_file}...")
    df = load_and_preprocess_data(data_file)
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print()
    
    # Prepare input
    print("Preparing prediction input...")
    context_length = 128
    features = ['close', 'high', 'low', 'open', 'volume', 'RSI_14', 'MACDh_12_26_9']
    
    past_values = prepare_prediction_input(df, context_length=context_length, features=features)
    print(f"✓ Input shape: {past_values.shape}")
    print()
    
    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(model, past_values, prediction_length=6)
    print(f"✓ Predictions shape: {predictions.shape}")
    print()
    
    # Display results
    feature_names = features
    last_timestamp = df.index[-1]
    current_close = df['close'].iloc[-1]
    
    # NEXT 5 MINUTES PREDICTION (Most Important)
    print("=" * 60)
    print("NEXT 5 MINUTES PREDICTION")
    print("=" * 60)
    print()
    
    next_5min_time = last_timestamp + pd.Timedelta(minutes=5)
    next_5min_pred = predictions[0]  # First prediction
    
    print(f"Current Time: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Prediction Time: {next_5min_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("PREDICTED VALUES:")
    print("-" * 60)
    
    predicted_close_5min = next_5min_pred[0]
    change_5min = predicted_close_5min - current_close
    change_pct_5min = (change_5min / current_close) * 100
    
    for j, feature in enumerate(feature_names):
        value = next_5min_pred[j]
        if feature == 'close':
            print(f"  {feature:15s}: ${value:,.2f} ({change_pct_5min:+.2f}%)")
        elif feature == 'high':
            print(f"  {feature:15s}: ${value:,.2f}")
        elif feature == 'low':
            print(f"  {feature:15s}: ${value:,.2f}")
        elif feature == 'open':
            print(f"  {feature:15s}: ${value:,.2f}")
        elif feature == 'volume':
            print(f"  {feature:15s}: {value:,.0f}")
        else:
            print(f"  {feature:15s}: {value:,.4f}")
    
    print()
    print("=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"Current Price:     ${current_close:,.2f}")
    print(f"Predicted (5min):  ${predicted_close_5min:,.2f}")
    print(f"Expected Change:   ${change_5min:,.2f} ({change_pct_5min:+.2f}%)")
    print()
    
    # Show all 6 periods (30 minutes) if user wants details
    print("=" * 60)
    print("FULL PREDICTIONS (Next 30 minutes - 6 periods)")
    print("=" * 60)
    print()
    
    for i in range(6):
        period_time = last_timestamp + pd.Timedelta(minutes=5 * (i + 1))
        pred_close = predictions[i, 0]
        period_change = pred_close - current_close
        period_change_pct = (period_change / current_close) * 100
        
        print(f"Period {i+1} ({period_time.strftime('%H:%M:%S')}): "
              f"${pred_close:,.2f} ({period_change_pct:+.2f}%)")
    
    print()
    print("=" * 60)
    print("30-MINUTE SUMMARY")
    print("=" * 60)
    predicted_close_30min = predictions[5, 0]  # Last prediction
    change_30min = predicted_close_30min - current_close
    change_pct_30min = (change_30min / current_close) * 100
    
    print(f"Current Price:        ${current_close:,.2f}")
    print(f"Predicted (30min):    ${predicted_close_30min:,.2f}")
    print(f"Expected Change:      ${change_30min:,.2f} ({change_pct_30min:+.2f}%)")
    print()


if __name__ == "__main__":
    main()


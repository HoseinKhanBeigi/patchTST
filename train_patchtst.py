"""
PatchTST Model Training Script for Bitcoin Multi-variate Time Series Forecasting

This script trains a PatchTST model from Hugging Face transformers library
for predicting future Bitcoin price movements using historical OHLCV data
and technical indicators.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
from transformers import (
    PatchTSTForPrediction,
    PatchTSTConfig,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset


def load_and_preprocess_data(file_path):
    """
    Load Bitcoin data and add technical indicators.
    
    Args:
        file_path: Path to the CSV file containing Bitcoin data
        
    Returns:
        DataFrame with timestamp index and technical indicators
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Set timestamp as datetime index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Add technical indicators using pandas-ta
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df['MACD_12_26_9'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    df['MACDh_12_26_9'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACDh_12_26_9']
    
    # Drop any NaN values
    df = df.dropna()
    
    return df


def create_sliding_windows(df, context_length=128, prediction_length=6, 
                          features=['close', 'high', 'low', 'open', 'volume', 
                                   'RSI_14', 'MACDh_12_26_9']):
    """
    Create sliding windows for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        context_length: Number of historical periods to use as input
        prediction_length: Number of future periods to predict
        features: List of feature column names to include
        
    Returns:
        List of dictionaries, each containing 'start' and 'target' keys
    """
    windows = []
    total_length = context_length + prediction_length
    
    # Extract feature columns
    feature_data = df[features].values  # Shape: (num_samples, num_features)
    
    # Create sliding windows
    for i in range(len(df) - total_length + 1):
        # Extract window data
        window_data = feature_data[i:i + total_length]  # Shape: (total_length, num_features)
        
        # Transpose to get shape (num_features, total_length)
        window_data = window_data.T
        
        # Get the start timestamp
        start_timestamp = df.index[i]
        
        # Create dictionary for this window
        window_dict = {
            'start': pd.Timestamp(start_timestamp),
            'target': window_data.astype(np.float32)
        }
        
        windows.append(window_dict)
    
    return windows




def main():
    """
    Main training function.
    """
    # Data Loading and Feature Engineering
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('btc_5m_5years.csv')
    print(f"Data loaded: {len(df)} samples")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Define features
    features = ['close', 'high', 'low', 'open', 'volume', 'RSI_14', 'MACDh_12_26_9']
    
    # Verify all features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Data Preparation for Hugging Face
    # Split the data into training set (first 80%) and validation set (last 20%) BEFORE creating windows
    print("\nSplitting data into train/validation sets...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]  # Use remaining 20% for validation
    
    print(f"Training data: {len(train_df)} samples")
    print(f"Validation data: {len(val_df)} samples")
    
    # Create sliding windows
    print("\nCreating sliding windows...")
    context_length = 128
    prediction_length = 6
    
    # Create windows from training and validation sets separately
    train_windows = create_sliding_windows(
        train_df, 
        context_length=context_length,
        prediction_length=prediction_length,
        features=features
    )
    
    val_windows = create_sliding_windows(
        val_df, 
        context_length=context_length,
        prediction_length=prediction_length,
        features=features
    )
    
    print(f"Training windows: {len(train_windows)}")
    print(f"Validation windows: {len(val_windows)}")
    
    # Model Configuration
    print("\nConfiguring PatchTST model...")
    config = PatchTSTConfig(
        prediction_length=prediction_length,
        context_length=context_length,
        num_input_channels=7,  # Number of features (channels)
        input_size=7,  # Also set input_size for compatibility
        patch_length=16,
        patch_stride=8,
        num_time_features=0,  # No separate time feature encoding
        loss="mse"  # Mean Squared Error for regression
    )
    
    # Verify config
    print(f"Config - num_input_channels: {config.num_input_channels if hasattr(config, 'num_input_channels') else 'N/A'}")
    print(f"Config - input_size: {config.input_size}")
    
    # Instantiate the Model
    print("Instantiating PatchTST model...")
    model = PatchTSTForPrediction(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training Setup
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./patchtst_bitcoin_output",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=False,  # Disable until eval_loss is properly computed
        # metric_for_best_model="eval_loss",  # Commented out until eval_loss works
        # greater_is_better=False,
        save_total_limit=3,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,  # Use mixed precision training if available
    )
    
    # Create a dataset class for Hugging Face Trainer
    class TimeSeriesDataset(Dataset):
        def __init__(self, windows, context_length, prediction_length):
            self.windows = windows
            self.context_length = context_length
            self.prediction_length = prediction_length
        
        def __len__(self):
            return len(self.windows)
        
        def __getitem__(self, idx):
            window = self.windows[idx]
            target = window['target']  # Shape: (num_features, context_length + prediction_length)
            
            # Split into past_values (context) and future_values (ground truth)
            past_values = target[:, :self.context_length]  # Shape: (num_features, context_length)
            future_values = target[:, self.context_length:]  # Shape: (num_features, prediction_length)
            
            # PatchTST expects (sequence_length, num_features), so transpose
            past_values = past_values.T  # Shape: (context_length, num_features)
            future_values = future_values.T  # Shape: (prediction_length, num_features)
            
            return {
                'start': window['start'],
                'past_values': torch.tensor(past_values, dtype=torch.float32),
                'future_values': torch.tensor(future_values, dtype=torch.float32),
            }
    
    train_dataset = TimeSeriesDataset(train_windows, context_length, prediction_length)
    val_dataset = TimeSeriesDataset(val_windows, context_length, prediction_length)
    
    # Create a custom data collator for PatchTST
    def patchtst_data_collator(features):
        """
        Custom data collator for PatchTST that batches the data correctly.
        """
        if not features:
            return {}
        
        batch = {}
        
        # Stack past_values and future_values
        if 'past_values' in features[0]:
            batch['past_values'] = torch.stack([f['past_values'] for f in features])
        if 'future_values' in features[0]:
            batch['future_values'] = torch.stack([f['future_values'] for f in features])
        
        # Keep start timestamps if available (optional)
        if len(features) > 0 and 'start' in features[0]:
            batch['start'] = [f.get('start') for f in features]
        
        return batch
    
    # Define compute_metrics function to compute eval_loss
    # Note: For PatchTST, the Trainer automatically computes loss during evaluation
    # This function is a fallback in case it's needed
    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics.
        For PatchTST, the loss is computed automatically by the model.
        This function can be used to add additional metrics if needed.
        """
        # The eval_pred format depends on what the model returns
        # For now, return empty dict - loss should be computed automatically
        return {}
    
    # Instantiate Trainer
    print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=patchtst_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the Model
    print("\nStarting training...")
    print("=" * 50)
    trainer.train()
    
    # Save the final model
    print("\nSaving model...")
    trainer.save_model("./patchtst_bitcoin_output/final_model")
    print("Training completed!")
    print(f"Model saved to: ./patchtst_bitcoin_output/final_model")


if __name__ == "__main__":
    main()


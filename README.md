# PatchTST Bitcoin Forecasting Training

This repository contains a script to train a PatchTST model for multi-variate time series forecasting on Bitcoin data.

## Step 0: Download Bitcoin Data

Before training, you need to download the Bitcoin OHLCV data using CCXT:

```bash
# Install CCXT if not already installed
pip install ccxt

# Download 5 years of 5-minute Bitcoin data
python download_btc_data.py
```

This will create `btc_5m_5years.csv` with the required data format.

**Note:** The download script will try multiple exchanges (Binance, Coinbase, Kraken, etc.) automatically if one fails. The download may take several minutes depending on your internet connection.

## Setup on Vast.ai GPU Instance

### 1. Connect to Your Instance

You can connect via SSH:
```bash
ssh -p 17638 root@36.34.82.195
```

Or access Jupyter notebook at:
```
http://36.34.82.195:17523
```
(Check which port is mapped to Jupyter - it might be one of the other ports)

### 2. Transfer Files to Instance

From your local machine, transfer the training script and data:

```bash
# Transfer the training script
scp -P 17638 train_patchtst.py root@36.34.82.195:/root/

# Transfer the data download script (if downloading on instance)
scp -P 17638 download_btc_data.py root@36.34.82.195:/root/

# Transfer the CSV data file (if downloaded locally)
scp -P 17638 btc_5m_5years.csv root@36.34.82.195:/root/

# Transfer requirements file
scp -P 17638 requirements.txt root@36.34.82.195:/root/
```

**Alternative:** You can also download the data directly on the Vast.ai instance:
```bash
# On the instance
python download_btc_data.py
```

### 3. Install Dependencies

Once connected to the instance:

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually:
pip install pandas numpy pandas-ta torch transformers
```

### 4. Run Training

```bash
python train_patchtst.py
```

### 5. Monitor Training

The script will:
- Print progress to console
- Save checkpoints to `./patchtst_bitcoin_output/`
- Save logs to `./logs/`
- Save the final model to `./patchtst_bitcoin_output/final_model/`

### 6. Download Results

After training completes, download the model:

```bash
# From your local machine
scp -P 17638 -r root@36.34.82.195:/root/patchtst_bitcoin_output ./local_output/
```

## GPU Usage

The script will automatically use the GPU if available. You can verify GPU usage with:

```bash
nvidia-smi
```

## Notes

- The RTX 5090 has 31.8 GB VRAM, which is more than sufficient for this model
- Training time will depend on dataset size, but should be relatively fast on this GPU
- The script uses mixed precision training (fp16) by default for faster training
- Adjust `per_device_train_batch_size` if you encounter out-of-memory errors


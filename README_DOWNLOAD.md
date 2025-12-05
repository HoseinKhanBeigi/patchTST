# Download Trained Model from Vast.ai

After training completes, download your model and outputs using one of these methods:

## Option 1: Use the Download Script (Recommended)

```bash
./download_from_vast.sh
```

This will download:
- The trained model (`patchtst_bitcoin_output/`)
- Training logs (`logs/`)
- Optionally the data file (`btc_5m_5years.csv`)

## Option 2: Manual Download

### Download the trained model:
```bash
scp -P 17638 -r root@36.34.82.195:/root/patchtst_bitcoin_output ./
```

### Download training logs:
```bash
scp -P 17638 -r root@36.34.82.195:/root/logs ./
```

### Download the data file (optional):
```bash
scp -P 17638 root@36.34.82.195:/root/btc_5m_5years.csv ./
```

## What You'll Get

### Model Directory Structure:
```
patchtst_bitcoin_output/
├── final_model/          # Your trained model
│   ├── config.json       # Model configuration
│   ├── pytorch_model.bin # Model weights
│   └── ...
├── checkpoint-*/         # Training checkpoints (if any)
└── ...
```

### Logs Directory:
```
logs/
└── events.out.tfevents.* # Training logs (viewable with TensorBoard)
```

## Using the Downloaded Model

Once downloaded, you can load and use the model:

```python
from transformers import PatchTSTForPrediction
import torch

# Load the trained model
model = PatchTSTForPrediction.from_pretrained("./patchtst_bitcoin_output/final_model")
model.eval()

# Use for predictions (example)
# past_values shape: (batch_size, sequence_length=128, num_features=7)
# predictions shape: (batch_size, prediction_length=6, num_features=7)
```

## File Sizes

- Model: ~2-5 MB (compressed)
- Logs: ~1-10 MB (depending on training duration)
- Data file: ~32 MB (if downloaded)


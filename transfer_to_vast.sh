#!/bin/bash
# Script to transfer files to Vast.ai instance

VAST_IP="36.34.82.195"
VAST_PORT="17638"
REMOTE_USER="root"
REMOTE_DIR="/root"

echo "Transferring files to Vast.ai instance..."

# Transfer training script
echo "Transferring train_patchtst.py..."
scp -P $VAST_PORT train_patchtst.py ${REMOTE_USER}@${VAST_IP}:${REMOTE_DIR}/

# Transfer data download script
echo "Transferring download_btc_data.py..."
scp -P $VAST_PORT download_btc_data.py ${REMOTE_USER}@${VAST_IP}:${REMOTE_DIR}/

# Transfer requirements file
echo "Transferring requirements.txt..."
scp -P $VAST_PORT requirements.txt ${REMOTE_USER}@${VAST_IP}:${REMOTE_DIR}/

# Transfer data file (if it exists)
if [ -f "btc_5m_5years.csv" ]; then
    echo "Transferring btc_5m_5years.csv..."
    scp -P $VAST_PORT btc_5m_5years.csv ${REMOTE_USER}@${VAST_IP}:${REMOTE_DIR}/
else
    echo "Warning: btc_5m_5years.csv not found. Please transfer it manually."
fi

echo "Transfer complete!"
echo ""
echo "Next steps:"
echo "1. SSH into the instance: ssh -p $VAST_PORT ${REMOTE_USER}@${VAST_IP}"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Download Bitcoin data (if not transferred): python download_btc_data.py"
echo "4. Run training: python train_patchtst.py"


#!/bin/bash
# Script to download trained model and outputs from Vast.ai instance

VAST_IP="36.34.82.195"
VAST_PORT="17638"
REMOTE_USER="root"
REMOTE_DIR="/root"
LOCAL_DIR="./vast_outputs"

echo "Downloading files from Vast.ai instance..."
echo ""

# Create local directory
mkdir -p "$LOCAL_DIR"

# Download the trained model
echo "Downloading trained model..."
scp -P $VAST_PORT -r ${REMOTE_USER}@${VAST_IP}:${REMOTE_DIR}/patchtst_bitcoin_output "$LOCAL_DIR/"

# Download training logs
if ssh -p $VAST_PORT ${REMOTE_USER}@${VAST_IP} "[ -d ${REMOTE_DIR}/logs ]"; then
    echo "Downloading training logs..."
    scp -P $VAST_PORT -r ${REMOTE_USER}@${VAST_IP}:${REMOTE_DIR}/logs "$LOCAL_DIR/"
else
    echo "No logs directory found."
fi

# Download the data file (optional)
read -p "Download the Bitcoin data file (btc_5m_5years.csv)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading Bitcoin data file..."
    scp -P $VAST_PORT ${REMOTE_USER}@${VAST_IP}:${REMOTE_DIR}/btc_5m_5years.csv "$LOCAL_DIR/" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Data file downloaded"
    else
        echo "✗ Data file not found on server"
    fi
fi

echo ""
echo "Download complete!"
echo "Files saved to: $LOCAL_DIR"
echo ""
echo "Contents:"
ls -lh "$LOCAL_DIR"


#!/usr/bin/env bash
# Downloads the FER-2013 dataset from Kaggle and unpacks it into data/fer2013/.
# Supports both ~/.kaggle/kaggle.json and ~/.kaggle/access_token formats.

set -euo pipefail

DEST="data/fer2013"

if ! command -v kaggle &>/dev/null; then
    echo "Error: kaggle CLI not found. Install it with: pip install kaggle"
    exit 1
fi

if [ -f "$HOME/.kaggle/access_token" ]; then
    export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
elif [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Error: No Kaggle credentials found."
    echo "Save your token to ~/.kaggle/access_token or ~/.kaggle/kaggle.json"
    echo "Get your token at: https://www.kaggle.com/settings -> API"
    exit 1
fi

mkdir -p "$DEST"

echo "Downloading FER-2013 (CSV version)..."
# Use the CSV dataset which contains fer2013.csv (deadskull7/fer2013)
kaggle datasets download -d deadskull7/fer2013 -p "$DEST"

echo "Unzipping..."
unzip -q "$DEST/fer2013.zip" -d "$DEST"
rm "$DEST/fer2013.zip"

echo "Done. Dataset available at ./$DEST"
echo "If t
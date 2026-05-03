#!/usr/bin/env bash
# Downloads the FER-2013 dataset from Kaggle and unpacks it into data/fer2013/.
# Requires the Kaggle CLI and a valid ~/.kaggle/kaggle.json API key.
# See README.md for setup instructions.

set -euo pipefail

DEST="data/fer2013"

if ! command -v kaggle &>/dev/null; then
    echo "Error: kaggle CLI not found. Install it with: pip install kaggle"
    exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Error: Kaggle API key not found at ~/.kaggle/kaggle.json"
    echo "Go to https://www.kaggle.com/settings -> API -> Create New Token"
    exit 1
fi

mkdir -p "$DEST"

echo "Downloading FER-2013..."
kaggle datasets download -d msambare/fer2013 -p "$DEST"

echo "Unzipping..."
unzip -q "$DEST/fer2013.zip" -d "$DEST"
rm "$DEST/fer2013.zip"

echo "Done. Dataset available at ./$DEST"
echo "Expected structure:"
echo "  $DEST/train/<class>/*.jpg"
echo "  $DEST/test/<class>/*.jpg"

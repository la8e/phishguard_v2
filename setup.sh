#!/usr/bin/env bash

set -e

REPO_URL="https://github.com/la8e/phishguard_v2.git"
REPO_NAME="phishguard_v2"

echo "[1/6] Cloning repository..."
if [ ! -d "$REPO_NAME" ]; then
    git clone "$REPO_URL"
else
    echo "Repository already exists. Skipping clone."
fi

cd "$REPO_NAME"

echo "[2/6] Creating virtual environment..."
python3 -m venv .venv

echo "[3/6] Activating virtual environment..."
source .venv/bin/activate

echo "[4/6] Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[5/6] Creating FastText directory..."
mkdir -p src/features/fastText

if [ ! -f src/features/fastText/cc.en.300.bin ]; then
    echo "[6/6] Downloading FastText model..."
    wget \
        https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz \
        -O src/features/fastText/cc.en.300.bin.gz
    gunzip src/features/fastText/cc.en.300.bin.gz
else
    echo "FastText model already exists."
fi

echo
echo "Setup completed."
echo
echo "Activate later with:"
echo "source .venv/bin/activate"

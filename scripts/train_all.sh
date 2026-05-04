#!/usr/bin/env bash
# Train all three FER-2013 models in sequence, then finalize results for each.
# Run from the repo root:  bash scripts/train_all.sh
#
# Approximate runtimes on a single GPU (T4 / A100):
#   SimpleCNN  : ~10 min  (48×48, 50 epochs)
#   ResNet-50  : ~45 min  (224×224, 50 epochs)
#   ViT-B/16   : ~90 min  (224×224, 50 epochs)

set -euo pipefail

echo "============================================================"
echo " FER-2013 full training pipeline"
echo "============================================================"

# ---------------------------------------------------------------------------
# SimpleCNN — native 48×48 grayscale
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Training SimpleCNN ..."
python -m src.training.train \
    --config configs/default.yaml \
    --model  simple_cnn \
    --image-size 48 \
    --channels   1

echo "[1/3] Finalizing SimpleCNN results ..."
python scripts/finalize_results.py \
    --model      simple_cnn \
    --checkpoint checkpoints/simple_cnn/best.pt

echo "DONE: simple_cnn"

# ---------------------------------------------------------------------------
# ResNet-50 — pretrained ImageNet, 224×224 RGB
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Training ResNet-50 ..."
python -m src.training.train \
    --config configs/default.yaml \
    --model  resnet \
    --image-size 224 \
    --channels   3

echo "[2/3] Finalizing ResNet-50 results ..."
python scripts/finalize_results.py \
    --model      resnet \
    --checkpoint checkpoints/resnet/best.pt

echo "DONE: resnet"

# ---------------------------------------------------------------------------
# ViT-B/16 — pretrained ImageNet-21k, 224×224 RGB
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Training ViT-B/16 ..."
python -m src.training.train \
    --config configs/default.yaml \
    --model  vit \
    --image-size 224 \
    --channels   3

echo "[3/3] Finalizing ViT-B/16 results ..."
python scripts/finalize_results.py \
    --model      vit \
    --checkpoint checkpoints/vit/best.pt

echo "DONE: vit"

echo ""
echo "============================================================"
echo " All models trained. Generating figures ..."
echo "============================================================"

python scripts/generate_figures.py
python scripts/generate_saliency_grid.py

echo ""
echo "All done. Results in results/ and results/figures/"

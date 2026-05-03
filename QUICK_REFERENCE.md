# Quick Reference Card

## 🚀 Start Here (One Command)
```bash
python scripts/run_full_pipeline.py
```

---

## ⚡ Individual Commands

```bash
# 1. Verify setup is correct (5 min)
python scripts/verify_setup.py

# 2. Train all 3 models (2-4 hours)
python scripts/train_all_models.py

# 3. Evaluate models on test set (5 min)
python scripts/collect_results.py

# 4. Create visualizations (2 min)
python scripts/visualize_results.py

# 5. Detailed analysis (optional, 2 min)
python scripts/analyze_results.py
```

---

## 📊 Results After Training

```
results/
├── 00_summary_metrics.txt       ← Quick overview
├── evaluation_results.json      ← Raw data
└── visualizations/
    ├── 01_accuracy_comparison.png          ← Best model check
    ├── 02_per_class_f1_scores.png          ← Per-emotion scores
    ├── 03_confusion_matrix_*.png           ← Error patterns
    └── 04_metrics_heatmap.png              ← All metrics
```

---

## 🎯 Expected Accuracies

| Model | Accuracy | Time |
|-------|----------|------|
| SimpleCNN | ~55-62% | 20-40 min |
| ResNet-50 | ~68-73% | 45-75 min |
| ViT | ~72-76% | 60-120 min |

---

## ⚙️ Key Configuration (configs/default.yaml)

```yaml
image_size: 48          # 48 for SimpleCNN, 224 for ResNet/ViT
channels: 1             # 1 for SimpleCNN, 3 for ResNet/ViT
batch_size: 64          # Reduce if out of memory
epochs: 50              # Training iterations
lr: 3.0e-4              # Learning rate
early_stopping_patience: 10  # Stop if no improvement
```

---

## 📍 Where Things Are

```
src/models/              Models (SimpleCNN, ResNet, ViT)
src/training/train.py    Training loop
src/evaluation/          Metrics computation
configs/                 Configuration files
scripts/                 Training/evaluation scripts
checkpoints/             Saved model weights
results/                 Evaluation outputs
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| Out of memory | Reduce `batch_size` |
| No GPU | Install CUDA/CuDNN (default falls back to CPU) |
| No dataset | Run `bash scripts/download_data.sh` |
| Import errors | `pip install -r requirements.txt` |

---

## 📚 Full Documentation

- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md) (5-minute setup)
- **Pipeline Details**: See [PIPELINE.md](PIPELINE.md) (step-by-step walkthrough)
- **Full Docs**: See [README.md](README.md) (complete reference)

---

**TL;DR**: `python scripts/run_full_pipeline.py` then read `results/00_summary_metrics.txt`


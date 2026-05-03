# FERreal - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. **Setup** (1 minute)
```bash
# Clone & navigate
git clone https://github.com/aathithc/FERreal.git
cd FERreal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Download Data** (5-30 minutes depending on internet)
```bash
bash scripts/download_data.sh
```

### 3. **Verify Everything Works**
```bash
python scripts/verify_setup.py
```

This checks:
- ✓ All packages installed
- ✓ GPU availability
- ✓ Dataset accessible
- ✓ Models can load
- ✓ Data pipeline works

### 4. **Train All Models** (2-4 hours with GPU)
```bash
python scripts/run_full_pipeline.py
```

This will:
1. Train SimpleCNN (48×48 grayscale)
2. Train ResNet-50 (224×224 RGB, ImageNet pretrained)
3. Train ViT (224×224 RGB, ImageNet pretrained)
4. Evaluate all models on test set
5. Generate visualizations

---

## 📊 What You'll Get

After training completes:

```
results/
├── 00_summary_metrics.txt              # Quick reference table
├── evaluation_results.json             # Raw metrics in JSON
└── visualizations/
    ├── 01_accuracy_comparison.png      # Accuracy bar chart
    ├── 02_per_class_f1_scores.png      # Per-emotion F1 comparison
    ├── 03_confusion_matrix_*.png       # 3 confusion matrices
    └── 04_metrics_heatmap.png          # Performance heatmap

checkpoints/
├── simple_cnn/best.pt                  # Trained SimpleCNN weights
├── resnet/best.pt                      # Trained ResNet-50 weights
└── vit/best.pt                         # Trained ViT weights
```

---

## ⚡ Manual Steps (If You Want More Control)

### Train Individual Models
```bash
# SimpleCNN - fast baseline
python -m src.training.train --config configs/default.yaml --model simple_cnn

# ResNet-50 - traditional CNN
python -m src.training.train --config configs/default.yaml --model resnet

# Vision Transformer - state-of-the-art
python -m src.training.train --config configs/default.yaml --model vit
```

### Evaluate & Collect Results
```bash
# Compute metrics on test set
python scripts/collect_results.py
```

### Generate Visualizations Only
```bash
# Create charts from results
python scripts/visualize_results.py
```

---

## 🛠️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model settings
image_size: 48          # 48 for SimpleCNN, 224 for ResNet/ViT
channels: 1             # 1 for SimpleCNN, 3 for ResNet/ViT

# Training settings
batch_size: 64
epochs: 50
lr: 3.0e-4
scheduler: cosine       # or 'step', 'onecycle'

# Data settings
use_weighted_sampler: true   # Handle class imbalance
use_class_weights: true      # Weighted loss
```

---

## 🔍 What Each Model Does

| Model | Input | Pros | Cons |
|-------|-------|------|------|
| **SimpleCNN** | 48×48 grayscale | Fast, simple, ~100K params | Lower accuracy |
| **ResNet-50** | 224×224 RGB | Good accuracy, proven, ~23M params | Slower training |
| **ViT** | 224×224 RGB | Best accuracy, modern, ~86M params | Requires GPU |

---

## 📈 Expected Performance

*Typical results on FER-2013 test set:*

- **SimpleCNN**: ~55-60% accuracy
- **ResNet-50**: ~65-70% accuracy  
- **ViT**: ~70-75% accuracy

(Actual results depend on training time, hardware, random seed)

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU not found | Install CUDA toolkit, or edit script to force CPU |
| Out of memory | Reduce `batch_size` in config |
| Dataset not found | Run `bash scripts/download_data.sh` |
| Import errors | Reinstall requirements: `pip install -r requirements.txt` |

---

## 📚 Learn More

- **Full README**: See `README.md` for detailed documentation
- **Data Exploration**: Open `notebooks/01_data_exploration.ipynb`
- **Model Code**: Check `src/models/` for implementations
- **Training Code**: Check `src/training/train.py`

---

## 🎯 Next Steps After Training

1. **Review Results**
   ```bash
   cat results/00_summary_metrics.txt
   ```

2. **Analyze Visualizations**
   - Open PNGs in `results/visualizations/`
   - Confusion matrices show which emotions are confused

3. **Fine-tune**
   - Adjust `configs/default.yaml`
   - Re-run training with new settings

4. **Interpret Models**
   - Use `src/interpretability/attention_maps.py` for ViT
   - Use `src/interpretability/gradcam.py` for CNN-based models

---

**Happy training! 🚀**

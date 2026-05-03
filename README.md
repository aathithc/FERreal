# FER-2013: CNNs vs Vision Transformers

Comparing CNN and ViT architectures on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) facial expression recognition benchmark.

**Dataset:** 48×48 grayscale images, 7 emotion classes, ~35k images.  
**Classes:** angry, disgust, fear, happy, neutral, sad, surprise.

## Team

| Name | Model |
|------|-------|
| TBD  | SimpleCNN (baseline) |
| TBD  | ResNet-50 |
| TBD  | Vision Transformer (ViT) |
| TBD  | Evaluation & interpretability |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd fer-project
pip install -r requirements.txt
```

### 2. Configure Kaggle API key

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Move the downloaded file:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Download the dataset

```bash
bash scripts/download_data.sh
```

This downloads and unpacks FER-2013 to `data/fer2013/` with the layout:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    └── ...
```

---

## Training

```bash
# Train the ResNet model with default config
python -m src.training.train --config configs/default.yaml --model resnet

# Train ViT (needs 224×224 input — override image_size and channels)
python -m src.training.train --config configs/default.yaml --model vit
```

Each model saves its best checkpoint to `checkpoints/<model_name>/best.pt`.

### Key config options (`configs/default.yaml`)

| Key | Default | Notes |
|-----|---------|-------|
| `image_size` | 48 | Set to 224 for ResNet/ViT |
| `channels` | 1 | Set to 3 for pretrained ImageNet models |
| `use_weighted_sampler` | true | Handles class imbalance via sampling |
| `use_class_weights` | true | Weighted CrossEntropyLoss |
| `scheduler` | cosine | Options: cosine, step, onecycle |
| `early_stopping_patience` | 10 | Epochs without val improvement before stopping |

---

## Evaluation

```python
from src.evaluation.evaluate import evaluate_model, print_results

results = evaluate_model(model, val_loader, device, class_names)
print_results(results)
# Returns: accuracy, per-class precision/recall/F1, confusion matrix
```

---

## Project Structure

```
fer-project/
├── configs/default.yaml        # Shared training config
├── src/
│   ├── data/
│   │   ├── dataset.py          # FER2013Dataset (PyTorch Dataset)
│   │   └── transforms.py       # Train / eval augmentation pipelines
│   ├── models/
│   │   ├── __init__.py         # MODEL_REGISTRY — register your model here
│   │   ├── simple_cnn.py       # Stub: custom CNN baseline
│   │   ├── resnet.py           # Stub: ResNet-50 wrapper
│   │   └── vit.py              # Stub: ViT wrapper
│   ├── training/
│   │   ├── train.py            # Generic training loop (CLI entry point)
│   │   └── utils.py            # set_seed, checkpointing, class weights
│   ├── evaluation/
│   │   └── evaluate.py         # Metrics: accuracy, per-class F1, confusion matrix
│   └── interpretability/
│       ├── gradcam.py          # Stub: Grad-CAM for CNNs
│       └── attention_maps.py   # Stub: attention rollout for ViT
├── notebooks/
│   └── 01_data_exploration.ipynb
└── scripts/
    └── download_data.sh
```

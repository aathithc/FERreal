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

## Setup (do this once, each teammate individually)

### 1. Clone and install dependencies

```bash
git clone https://github.com/aathithc/FERreal.git
cd FERreal
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment to avoid dependency conflicts:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

---

### 2. Get your Kaggle API key

Everyone needs their own free Kaggle account and API key — it only takes 2 minutes.

1. Create a free account at [kaggle.com](https://www.kaggle.com) if you don't have one
2. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → scroll to **API** section → click **Create New Token**
3. This downloads a file called `kaggle.json` to your computer
4. Move it to the right place:

**Mac/Linux:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json   # keeps the file private
```

**Windows (PowerShell):**
```powershell
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```

> The `kaggle.json` file contains your personal API credentials — never commit it to git (it's already in `.gitignore`).

---

### 3. Download the dataset

```bash
bash scripts/download_data.sh
```

This downloads and unpacks FER-2013 to `data/fer2013/` (~60 MB). The `data/` folder is gitignored — everyone downloads their own local copy.

Expected layout after download:
```
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (same structure)
```

**Troubleshooting:**
- `kaggle: command not found` — run `pip install kaggle` first
- `401 Unauthorized` — your `kaggle.json` is missing or in the wrong place; re-check step 2
- `403 Forbidden` — you need to accept the dataset rules: go to the [dataset page](https://www.kaggle.com/datasets/msambare/fer2013) and click **Download** once in the browser to accept terms

---

## Training

Run from the repo root:

```bash
# SimpleCNN (grayscale 48x48 — default config)
python -m src.training.train --config configs/default.yaml --model simple_cnn

# ResNet-50 (needs 224x224 RGB — edit image_size and channels in config first)
python -m src.training.train --config configs/default.yaml --model resnet

# ViT (same 224x224 RGB requirement as ResNet)
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
FERreal/
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

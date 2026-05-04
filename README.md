# FER-2013: CNNs vs Vision Transformers

Comparing CNN and ViT architectures on the FER-2013 facial expression recognition benchmark.

## Dataset

**Source:** [https://www.kaggle.com/datasets/deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013)  
**Kaggle slug:** `deadskull7/fer2013` — CSV version with proper 3-way split. Do **not** substitute `msambare/fer2013` (that version merges PublicTest+PrivateTest into one folder, making held-out evaluation impossible).

- 48×48 grayscale face images
- 7 emotion classes: angry, disgust, fear, happy, sad, surprise, neutral
- Splits: train=28,709 · val (PublicTest)=3,589 · test (PrivateTest)=3,589
- Class-imbalanced: disgust is only ~1.5% of the training set (handled automatically via weighted sampling)

## Team

| Name | Role |
|------|------|
| Aathith Chandra  | SimpleCNN baseline, repo setup |
| Pranav Nair      | Interpretability (GradCAM, attention rollout) |
| Aditya Modi      | Dataset pipeline, evaluation |
| Nishanth Thummala | ResNet-50 / ViT, results analysis |

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

Everyone needs their own free Kaggle account and API key — takes about 2 minutes.

1. Create a free account at [kaggle.com](https://www.kaggle.com) if you don't have one
2. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → scroll to **API** → click **Create New Token**
3. Kaggle will show you a token string (starts with something like `abc123...`)
4. Save it with these commands — paste your actual token where shown:

**Mac/Linux:**
```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
echo "your_token_here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

**Windows (PowerShell):**
```powershell
mkdir $env:USERPROFILE\.kaggle -Force
"your_token_here" | Out-File $env:USERPROFILE\.kaggle\access_token -Encoding ascii
```

> Never share your token or commit it to git — it's a personal credential tied to your account.

---

### 3. Install the Kaggle CLI and download the dataset

```bash
pip install kaggle
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
- `401 Unauthorized` — token is missing or wrong; redo step 2
- `permission denied` on `~/.kaggle/access_token` — run `chmod 700 ~/.kaggle` first, then retry
- `403 Forbidden` — go to the [dataset page](https://www.kaggle.com/datasets/deadskull7/fer2013) and click **Download** once in the browser to accept the license terms

---

## Reproducing the report

> **GPU required.** Approximate runtimes on a T4: SimpleCNN ~10 min · ResNet-50 ~45 min · ViT ~90 min.

### Option A — run everything at once

```bash
bash scripts/train_all.sh
```

This trains all three models, writes `results/{model}/metrics.json` and `results/{model}/training_log.csv`, then generates all figures to `results/figures/`.

### Option B — step by step

```bash
# 1. Download dataset (once per machine)
bash scripts/download_data.sh

# 2. Train each model (image_size and channels are set automatically)
python -m src.training.train --config configs/default.yaml --model simple_cnn --image-size 48  --channels 1
python -m src.training.train --config configs/default.yaml --model resnet     --image-size 224 --channels 3
python -m src.training.train --config configs/default.yaml --model vit        --image-size 224 --channels 3

# 3. Evaluate on held-out PrivateTest split → results/{model}/metrics.json
python scripts/finalize_results.py --model simple_cnn --checkpoint checkpoints/simple_cnn/best.pt
python scripts/finalize_results.py --model resnet     --checkpoint checkpoints/resnet/best.pt
python scripts/finalize_results.py --model vit        --checkpoint checkpoints/vit/best.pt

# 4. Generate report figures → results/figures/*.pdf
python scripts/generate_figures.py
python scripts/generate_saliency_grid.py
```

Each model saves its best checkpoint to `checkpoints/<model_name>/best.pt`.
Per-epoch metrics are logged to `results/<model_name>/training_log.csv` automatically during training.

## Training

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
│   │   ├── simple_cnn.py       # Custom CNN baseline (4 conv blocks)
│   │   ├── resnet.py           # ResNet-50 fine-tuned from ImageNet
│   │   └── vit.py              # ViT-B/16 fine-tuned via timm
│   ├── training/
│   │   ├── train.py            # Training loop — logs to results/{model}/training_log.csv
│   │   └── utils.py            # set_seed, checkpointing, class weights
│   ├── evaluation/
│   │   └── evaluate.py         # Metrics: accuracy, per-class F1, confusion matrix
│   └── interpretability/
│       ├── gradcam.py          # Grad-CAM for CNN models
│       └── attention_maps.py   # Attention rollout for ViT
├── notebooks/
│   └── 01_data_exploration.ipynb
└── scripts/
    ├── download_data.sh        # Download FER-2013 CSV from Kaggle
    ├── train_all.sh            # Train all 3 models end-to-end
    ├── finalize_results.py     # Eval on PrivateTest → results/{model}/metrics.json
    ├── generate_figures.py     # Training curves, confusion matrices, F1 bar chart
    └── generate_saliency_grid.py  # GradCAM / attention maps + failure cases
```

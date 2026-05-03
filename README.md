# FER-2013: CNNs vs Vision Transformers

Comparing CNN and ViT architectures on the FER-2013 facial expression recognition benchmark.

## Dataset

**Source:** [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)  
**Kaggle slug:** `msambare/fer2013` (this is what the download script uses — do not substitute another FER-2013 upload)

- 48×48 grayscale face images
- 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise
- ~35,000 images with a predefined train/test split
- Class-imbalanced: disgust is only ~1.5% of the training set (handled automatically via weighted sampling)

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
- `403 Forbidden` — go to the [dataset page](https://www.kaggle.com/datasets/msambare/fer2013) and click **Download** once in the browser to accept the license terms

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

## Using Claude Code to implement your model

Each teammate can use Claude Code to implement their stub file. Install it once:

```bash
npm install -g @anthropic-ai/claude-code
```

Then authenticate — no manual API key needed if you have a Claude.ai account:

```bash
claude   # opens browser login on first run
```

If your team is using an Anthropic API key directly instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # add to ~/.zshrc or ~/.bashrc to persist
claude
```

Once authenticated, open Claude Code inside the repo:

```bash
cd FERreal
claude
```

Then just describe what you need, e.g.:
> "Implement the ResNetFER class in src/models/resnet.py — load pretrained ResNet-50 from torchvision, replace the fc layer with num_classes outputs, and add an option to freeze early layers"

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

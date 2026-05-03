import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, path: str | Path) -> None:
    """Save a training checkpoint.

    Args:
        state: Dict containing at minimum 'epoch', 'model_state_dict',
               'optimizer_state_dict', and 'best_val_acc'.
        path:  File path (will be created including parent dirs).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint into model (and optionally optimizer).

    Returns the full checkpoint dict so callers can read epoch / best_val_acc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def compute_class_weights(
    dataset, device: torch.device | None = None
) -> torch.Tensor:
    """Compute inverse-frequency class weights for WeightedCrossEntropyLoss.

    Weights are normalised so they sum to num_classes, keeping the loss scale
    comparable to unweighted cross-entropy.

    Args:
        dataset: Any dataset with a ``get_class_counts()`` method (FER2013Dataset).

    Returns:
        FloatTensor of shape (num_classes,).
    """
    counts = np.array(dataset.get_class_counts(), dtype=np.float64)
    weights = 1.0 / np.where(counts > 0, counts, 1.0)
    weights = weights / weights.sum() * len(counts)
    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor

#!/usr/bin/env python3
"""
Quick utility script to:
1. Verify all dependencies are installed
2. Check GPU availability
3. Validate dataset
4. Load and test a single model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Verify all required packages are installed."""
    print("Checking dependencies...")
    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'timm',
        'sklearn',
        'yaml',
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas',
        'tqdm',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed\n")
    return True


def check_gpu():
    """Check GPU availability."""
    print("Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ✗ No GPU detected (training will use CPU - much slower)")
        print()
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}\n")
        return False
    
    return True


def check_dataset():
    """Verify FER-2013 dataset exists."""
    print("Checking dataset...")
    data_root = Path("data/fer2013")
    
    if not data_root.exists():
        print(f"  ✗ Dataset not found at {data_root}")
        print(f"  Run: bash scripts/download_data.sh")
        return False
    
    # Check for expected splits
    splits = ['train', 'test']
    for split in splits:
        split_path = data_root / split
        if split_path.exists():
            num_samples = len(list(split_path.glob('*/*')))
            print(f"  ✓ {split}: {num_samples} images")
        else:
            print(f"  ✗ {split} split not found at {split_path}")
            return False
    
    print()
    return True


def check_configs():
    """Verify configuration files exist."""
    print("Checking configuration...")
    config_file = Path("configs/default.yaml")
    
    if config_file.exists():
        print(f"  ✓ Config found: {config_file}")
    else:
        print(f"  ✗ Config not found at {config_file}")
        return False
    
    print()
    return True


def check_models():
    """Verify models can be loaded."""
    print("Checking models...")
    try:
        from src.models import MODEL_REGISTRY
        
        for model_name in MODEL_REGISTRY.keys():
            try:
                model = MODEL_REGISTRY[model_name](num_classes=7)
                print(f"  ✓ {model_name}")
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
                return False
        
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error loading models: {e}\n")
        return False


def test_data_loading():
    """Test loading a batch of data."""
    print("Testing data loading...")
    try:
        import torch
        from src.data.dataset import FER2013Dataset
        from src.data.transforms import get_eval_transforms
        from torch.utils.data import DataLoader
        
        # Try loading a small batch
        transforms = get_eval_transforms(48, 1)
        dataset = FER2013Dataset("data/fer2013", split="train", transform=transforms)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        
        batch = next(iter(loader))
        images, labels = batch
        
        print(f"  ✓ Loaded batch: images {images.shape}, labels {labels.shape}")
        print(f"  ✓ Dataset classes: {dataset.classes}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error loading data: {e}\n")
        return False


def main():
    """Run all checks."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                         ENVIRONMENT VERIFICATION                           ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    checks = [
        ("Dependencies", check_imports),
        ("GPU", check_gpu),
        ("Configuration", check_configs),
        ("Models", check_models),
        ("Dataset", check_dataset),
        ("Data Loading", test_data_loading),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n❌ {name} check failed: {e}\n")
            results[name] = False
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ All checks passed! Ready to train.")
        print("\nNext step: python scripts/run_full_pipeline.py")
    else:
        print("⚠️  Some checks failed. Please fix issues above before training.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

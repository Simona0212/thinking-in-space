#!/usr/bin/env python3
"""
Quick test script to verify VSI-Bench evaluation setup
"""

import sys
import os

def check_imports():
    """Check if all required packages are installed"""
    print("Checking required packages...")

    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'av': 'PyAV',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements_eval.txt")
        return False

    print("\nAll required packages installed!")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")

    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"  ✓ CUDA available")
            print(f"  ✓ {num_gpus} GPU(s) detected")
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                print(f"    - GPU {i}: {name}")
            return True
        else:
            print("  ✗ CUDA not available")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def check_dataset_access():
    """Check if dataset can be accessed"""
    print("\nChecking dataset access...")

    try:
        from datasets import load_dataset

        # Try to load dataset info
        print("  Attempting to load VSI-Bench metadata...")
        dataset = load_dataset("nyu-visionx/VSI-Bench", split="test", streaming=True)

        # Get first sample
        first_sample = next(iter(dataset))
        print(f"  ✓ Dataset accessible")
        print(f"  ✓ Sample keys: {list(first_sample.keys())}")
        return True

    except Exception as e:
        print(f"  ✗ Cannot access dataset: {e}")
        print("\n  Possible solutions:")
        print("    1. Set HF_TOKEN environment variable")
        print("    2. Run: huggingface-cli login")
        print("    3. Request access to nyu-visionx/VSI-Bench on HuggingFace")
        return False


def check_video_path():
    """Check if video cache directory exists"""
    print("\nChecking video cache directory...")

    dataset_path = "/cephfs/shared/vsi-bench"

    if os.path.exists(dataset_path):
        print(f"  ✓ Dataset path exists: {dataset_path}")

        # Check for video files
        video_count = 0
        for root, dirs, files in os.walk(dataset_path):
            video_count += len([f for f in files if f.endswith('.mp4')])
            if video_count > 0:
                break

        if video_count > 0:
            print(f"  ✓ Found video files")
        else:
            print(f"  ⚠ No .mp4 files found (may need to download)")

        return True
    else:
        print(f"  ⚠ Dataset path not found: {dataset_path}")
        print("    Videos will be downloaded on first run")
        return False


def test_model_loading():
    """Test loading a small model"""
    print("\nTesting model loading (optional)...")
    print("  Skipping model loading test (use --test-model to enable)")
    print("  This would download and load a model, which takes time")
    return True


def main():
    print("="*60)
    print("VSI-Bench Evaluation Setup Verification")
    print("="*60)
    print()

    checks = [
        ("Package imports", check_imports),
        ("CUDA availability", check_cuda),
        ("Dataset access", check_dataset_access),
        ("Video cache", check_video_path),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\nError during {name}: {e}")
            results[name] = False
        print()

    print("="*60)
    print("Summary")
    print("="*60)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print()

    if all(results.values()):
        print("✓ All checks passed! Ready to run evaluation.")
        return 0
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("  1. Install dependencies: pip install -r requirements_eval.txt")
        print("  2. Login to HuggingFace: huggingface-cli login")
        print("  3. Ensure CUDA is properly installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

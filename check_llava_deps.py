#!/usr/bin/env python3
"""
Check if all required dependencies for LLaVA-OneVision are installed
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name} is NOT installed: {e}")
        return False

def main():
    print("Checking LLaVA-OneVision dependencies...\n")

    required = {
        'torch': 'torch',
        'transformers': 'transformers',
        'PIL': 'Pillow',
        'av': 'av',
        'einops': 'einops',
        'timm': 'timm',
        'numpy': 'numpy',
    }

    optional = {
        'flash_attn': 'flash-attn (optional, can use eager attention)',
    }

    print("Required dependencies:")
    missing_required = []
    for module, package in required.items():
        if not check_import(module, package):
            missing_required.append(package)

    print("\nOptional dependencies:")
    for module, package in optional.items():
        check_import(module, package)

    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print(f"\nInstall with: pip install {' '.join(missing_required)}")
        return 1
    else:
        print("\n✓ All required dependencies are installed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

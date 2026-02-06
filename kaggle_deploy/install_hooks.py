#!/usr/bin/env python3
"""
RGT Install Hooks for Kaggle
============================

Unzips rgt_package.zip and adds to sys.path.
Run this cell first in your Kaggle notebook.

Usage (in Kaggle Notebook):
    exec(open("/kaggle/input/rgt-package/install_hooks.py").read())
"""

import os
import sys
import zipfile
from pathlib import Path


def install_rgt():
    """Install RGT package from zip."""
    # Kaggle paths
    INPUT_DIR = Path("/kaggle/input/rgt-package")
    WORKING_DIR = Path("/kaggle/working")
    PACKAGE_ZIP = INPUT_DIR / "rgt_package.zip"
    EXTRACT_DIR = WORKING_DIR / "rgt_extracted"
    
    # Local fallback (for testing)
    if not INPUT_DIR.exists():
        INPUT_DIR = Path("kaggle_deploy")
        WORKING_DIR = Path(".")
        PACKAGE_ZIP = INPUT_DIR / "rgt_package.zip"
        EXTRACT_DIR = WORKING_DIR / "rgt_extracted"
    
    print("=" * 60)
    print("RGT Package Installation")
    print("=" * 60)
    
    # Check if package exists
    if not PACKAGE_ZIP.exists():
        raise FileNotFoundError(f"Package not found: {PACKAGE_ZIP}")
    
    # Extract if needed
    if not EXTRACT_DIR.exists():
        print(f"[Install] Extracting to {EXTRACT_DIR}...")
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(PACKAGE_ZIP, 'r') as zf:
            zf.extractall(EXTRACT_DIR)
        
        print(f"  ✓ Extracted {len(list(EXTRACT_DIR.rglob('*.py')))} Python files")
    else:
        print(f"[Install] Using existing extraction: {EXTRACT_DIR}")
    
    # Add to sys.path
    src_path = str(EXTRACT_DIR / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"  ✓ Added to sys.path: {src_path}")
    
    # Verify import
    try:
        from rgt import HessianEngine, VectorAligner, PoissonSolver, SurfaceExtractor
        print("  ✓ RGT modules imported successfully")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        raise
    
    print("=" * 60)
    print("✓ RGT Package Ready")
    print("=" * 60)


# Auto-run on exec()
if __name__ == "__main__" or "__file__" not in dir():
    install_rgt()

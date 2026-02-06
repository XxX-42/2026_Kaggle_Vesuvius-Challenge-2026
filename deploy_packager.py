#!/usr/bin/env python3
"""
RGT Kaggle Deployment Packager
==============================

Bundles src/rgt/ modules into a Kaggle-compatible package.

Usage:
    python deploy_packager.py

Output:
    kaggle_deploy/
    ├── rgt_package.zip      # Bundled source code
    ├── install_hooks.py     # Bootstrap script
    └── inference_template.py # Notebook template

Author: DevOps Architect
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

SRC_DIR = Path("src/rgt")
OUTPUT_DIR = Path("kaggle_deploy")
PACKAGE_NAME = "rgt_package.zip"

# Files to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
    ".pyc",
    "test_*.py",
    "*.egg-info",
]


# =============================================================================
# Packager Functions
# =============================================================================

def should_include(filepath: Path) -> bool:
    """Check if file should be included in package."""
    name = filepath.name
    
    for pattern in EXCLUDE_PATTERNS:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return False
        elif pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return False
        elif pattern in str(filepath):
            return False
    
    return True


def create_package() -> Path:
    """Create rgt_package.zip from source files."""
    print("[Packager] Creating RGT package...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = OUTPUT_DIR / PACKAGE_NAME
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add src/__init__.py
        src_init = Path("src/__init__.py")
        if src_init.exists():
            zf.write(src_init, "src/__init__.py")
            print(f"  Added: src/__init__.py")
        
        # Add all files from src/rgt/
        for filepath in SRC_DIR.rglob("*"):
            if filepath.is_file() and should_include(filepath):
                arcname = f"src/rgt/{filepath.relative_to(SRC_DIR)}"
                zf.write(filepath, arcname)
                print(f"  Added: {arcname}")
    
    print(f"  ✓ Package created: {zip_path}")
    return zip_path


def create_install_hooks() -> Path:
    """Generate install_hooks.py for Kaggle bootstrap."""
    print("[Packager] Creating install hooks...")
    
    hooks_content = '''#!/usr/bin/env python3
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
'''
    
    hooks_path = OUTPUT_DIR / "install_hooks.py"
    hooks_path.write_text(hooks_content, encoding='utf-8')
    
    print(f"  ✓ Install hooks created: {hooks_path}")
    return hooks_path


def create_inference_template() -> Path:
    """Generate Kaggle inference notebook template."""
    print("[Packager] Creating inference template...")
    
    template_content = f'''#!/usr/bin/env python3
"""
RGT Kaggle Inference Template
=============================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This template provides the structure for running the RGT pipeline
on Kaggle competition data.

Notebook Structure:
    Cell 1: Environment Setup
    Cell 2: Hardware Check
    Cell 3: Pipeline Configuration
    Cell 4: Main Processing Loop
"""

# =============================================================================
# CELL 1: Environment Setup
# =============================================================================
# %%

import os
import gc
import sys
from pathlib import Path

# Install RGT package
exec(open("/kaggle/input/rgt-package/install_hooks.py").read())

# Imports
import numpy as np
import cupy as cp

from rgt import (
    ContextManager,
    HessianEngine,
    VectorAligner,
    PoissonSolver,
    SurfaceExtractor,
)


# =============================================================================
# CELL 2: Hardware Check
# =============================================================================
# %%

# Initialize context and detect environment
ctx = ContextManager.get_instance()
print(ctx)

# GPU memory info
if ctx.gpu_available:
    free, total = cp.cuda.Device(0).mem_info
    print(f"\\nGPU Memory: {{free/1e9:.2f}} GB free / {{total/1e9:.2f}} GB total")
else:
    print("\\n⚠ No GPU available - running in CPU mode")


# =============================================================================
# CELL 3: Pipeline Configuration
# =============================================================================
# %%

# Kaggle paths
INPUT_DIR = Path("/kaggle/input/vesuvius-challenge-data")  # Adjust to actual dataset
OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing parameters
CONFIG = {{
    "chunk_size": ctx.chunk_size if hasattr(ctx, 'chunk_size') else 128,
    "overlap": 16,
    "sigma": 1.5,
    "alignment_threshold": 0.5,
    "poisson_tol": 1e-5,
    "poisson_maxiter": 1000,
    "mesh_step_size": 1,
}}

print("Pipeline Configuration:")
for k, v in CONFIG.items():
    print(f"  {{k}}: {{v}}")


# =============================================================================
# CELL 4: Main Processing Loop
# =============================================================================
# %%

def process_volume(volume_path: Path, output_name: str) -> None:
    """
    Process a single volume through the RGT pipeline.
    
    Pipeline:
        1. Load volume
        2. Hessian feature extraction
        3. Vector field alignment
        4. Poisson solve
        5. Mesh extraction
        6. Save output
    """
    print(f"\\n{{'='*60}}")
    print(f"Processing: {{volume_path.name}}")
    print(f"{{'='*60}}")
    
    # Step 1: Load volume (placeholder - adjust for actual data format)
    print("\\n[1/6] Loading volume...")
    # volume = load_zarr_volume(volume_path)  # Implement based on data format
    # For demo, create synthetic data
    volume = cp.random.randn(64, 64, 64, dtype=cp.float32)
    
    # Step 2: Hessian
    print("\\n[2/6] Computing Hessian features...")
    engine = HessianEngine(slab_size=16, verbose=True)
    hessian = engine.compute_hessian(volume, sigma=CONFIG["sigma"])
    eigen = engine.solve_eigen_system(hessian.as_list())
    
    # Force GC
    hessian.free()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Step 3: Alignment
    print("\\n[3/6] Aligning vector field...")
    aligner = VectorAligner(threshold=CONFIG["alignment_threshold"], verbose=True)
    aligned = aligner.align_vectors(eigen.normal_vectors, eigen.confidence)
    
    # Force GC
    del eigen.normal_vectors
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Step 4: Poisson
    print("\\n[4/6] Solving Poisson equation...")
    weights = eigen.confidence / (eigen.confidence.max() + 1e-10)
    
    solver = PoissonSolver(
        tol=CONFIG["poisson_tol"],
        maxiter=CONFIG["poisson_maxiter"],
        verbose=True
    )
    A, b, _ = solver.assemble_system(aligned, weights)
    result = solver.solve(A, b, volume.shape)
    phi = result.phi
    
    # Force GC
    del A, b, aligned, weights, eigen
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Step 5: Mesh extraction
    print("\\n[5/6] Extracting mesh...")
    extractor = SurfaceExtractor(step_size=CONFIG["mesh_step_size"], verbose=True)
    mesh = extractor.extract_layer(phi, iso_level=float(phi.mean()))
    audit = extractor.audit_curvature(mesh)
    
    # Step 6: Save
    print("\\n[6/6] Saving output...")
    mesh_path = OUTPUT_DIR / f"{{output_name}}.obj"
    extractor.save_mesh(mesh, str(mesh_path))
    
    # Final GC
    del volume, phi, mesh
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    print(f"\\n✓ Complete: {{mesh_path}}")
    print(f"  Curvature defect ratio: {{audit.defect_ratio:.1%}}")


# Main execution
if __name__ == "__main__":
    # Process all volumes
    # volumes = list(INPUT_DIR.glob("*.zarr"))  # Adjust pattern
    # for i, vol_path in enumerate(volumes):
    #     process_volume(vol_path, f"rgt_mesh_{{i:03d}}")
    
    # Demo with synthetic data
    process_volume(Path("demo"), "rgt_demo_mesh")
    
    print("\\n" + "="*60)
    print("✓ ALL PROCESSING COMPLETE")
    print("="*60)
'''
    
    template_path = OUTPUT_DIR / "inference_template.py"
    template_path.write_text(template_content, encoding='utf-8')
    
    print(f"  ✓ Inference template created: {template_path}")
    return template_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the full packaging process."""
    print("=" * 60)
    print("RGT Kaggle Deployment Packager")
    print("=" * 60)
    print()
    
    # Create package
    zip_path = create_package()
    print()
    
    # Create install hooks
    hooks_path = create_install_hooks()
    print()
    
    # Create inference template
    template_path = create_inference_template()
    print()
    
    # Summary
    print("=" * 60)
    print("✓ Deployment Package Ready")
    print("=" * 60)
    print(f"  Package: {zip_path}")
    print(f"  Hooks:   {hooks_path}")
    print(f"  Template: {template_path}")
    print()
    print("To deploy to Kaggle:")
    print("  1. Upload kaggle_deploy/ as a Kaggle Dataset")
    print("  2. Create a new Notebook and add the dataset")
    print("  3. Copy contents from inference_template.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

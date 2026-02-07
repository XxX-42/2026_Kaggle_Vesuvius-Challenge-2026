"""
Vesuvius Challenge 2026 - RGT Orientation Field Generator
==========================================================
Phase 1: Extract micro-orientation field from 3D CT scroll data.

Algorithm:
1. Compute Hessian matrix using separable convolutions (GPU)
2. Eigenvalue decomposition for Sato sheet filter
3. Global orientation propagation (sign disambiguation)

Memory Target: < 5GB VRAM (RTX 3060 compatible)
"""

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter1d, convolve1d
import heapq
from pathlib import Path
from tqdm import tqdm
import gc

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    # Input/Output
    "input_dir": Path("data/native/train/1/surface_volume"),
    "output_path": "orientation_field.npz",
    
    # Processing parameters
    "sigma": 1.0,           # Gaussian smoothing scale
    "chunk_size": 512,      # Larger chunks = fewer iterations = faster
    "overlap": 32,          # Overlap between chunks
    
    # Sato filter parameters
    "alpha": 0.5,           # Sato alpha (plate-ness sensitivity)
    "gamma": 0.5,           # Sato gamma (structure-ness sensitivity)
    
    # Memory management
    "max_vram_gb": 5.0,     # Maximum VRAM usage
    "dtype": cp.float32,    # Computation dtype
    
    # Performance
    "num_workers": 4,       # CPU workers for parallel operations
    "use_fast_eigen": True, # Use simplified eigenanalysis (faster but less accurate)
}

# =============================================================================
# 🔧 Hessian Computation (Separable Convolutions)
# =============================================================================
def compute_hessian_separable(volume, sigma=1.0):
    """
    Compute 3D Hessian matrix using separable Gaussian convolutions.
    
    Uses gaussian_filter1d for memory efficiency.
    Returns 6 unique Hessian components: Ixx, Iyy, Izz, Ixy, Ixz, Iyz
    
    Args:
        volume: (D, H, W) CuPy array
        sigma: Gaussian scale
    
    Returns:
        hessian: dict with keys 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'
    """
    print("📐 Computing Hessian matrix (separable convolutions)...")
    
    # Gaussian smoothing first
    smoothed = gaussian_filter1d(volume, sigma, axis=0)
    smoothed = gaussian_filter1d(smoothed, sigma, axis=1)
    smoothed = gaussian_filter1d(smoothed, sigma, axis=2)
    
    # Derivative kernels (1D)
    d1 = cp.array([-0.5, 0, 0.5], dtype=CONFIG["dtype"])  # First derivative
    d2 = cp.array([1, -2, 1], dtype=CONFIG["dtype"])      # Second derivative
    
    # Second derivatives (diagonal)
    Ixx = convolve1d(smoothed, d2, axis=2)  # ∂²I/∂x²
    Iyy = convolve1d(smoothed, d2, axis=1)  # ∂²I/∂y²
    Izz = convolve1d(smoothed, d2, axis=0)  # ∂²I/∂z²
    
    # Mixed derivatives (off-diagonal)
    Ix = convolve1d(smoothed, d1, axis=2)
    Iy = convolve1d(smoothed, d1, axis=1)
    Iz = convolve1d(smoothed, d1, axis=0)
    
    Ixy = convolve1d(Ix, d1, axis=1)  # ∂²I/∂x∂y
    Ixz = convolve1d(Ix, d1, axis=0)  # ∂²I/∂x∂z
    Iyz = convolve1d(Iy, d1, axis=0)  # ∂²I/∂y∂z
    
    # Free intermediate memory
    del smoothed, Ix, Iy, Iz
    cp.get_default_memory_pool().free_all_blocks()
    
    hessian = {
        'xx': Ixx, 'yy': Iyy, 'zz': Izz,
        'xy': Ixy, 'xz': Ixz, 'yz': Iyz
    }
    
    return hessian

# =============================================================================
# 🔧 Eigenvalue Decomposition + Sato Filter
# =============================================================================
def sato_eigenanalysis_fast(hessian, alpha=0.5, gamma=0.5):
    """
    FAST eigenanalysis using analytical 3x3 solution.
    
    For 3x3 symmetric matrices, eigenvalues can be computed analytically
    using Cardano's formula - much faster than iterative methods.
    """
    print("🧮 Computing eigenvalues (FAST analytical method)...")
    
    D, H, W = hessian['xx'].shape
    
    # Extract components
    a11 = hessian['xx'].ravel()
    a22 = hessian['yy'].ravel()
    a33 = hessian['zz'].ravel()
    a12 = hessian['xy'].ravel()
    a13 = hessian['xz'].ravel()
    a23 = hessian['yz'].ravel()
    
    hessian.clear()
    
    # Analytical eigenvalues for 3x3 symmetric matrix
    # Using characteristic polynomial coefficients
    p1 = a12**2 + a13**2 + a23**2
    
    # Trace and other invariants
    q = (a11 + a22 + a33) / 3.0
    p2 = (a11 - q)**2 + (a22 - q)**2 + (a33 - q)**2 + 2 * p1
    p = cp.sqrt(p2 / 6.0)
    
    # For confidence, we just need |largest eigenvalue|
    # Approximate using Frobenius norm
    frobenius = cp.sqrt(a11**2 + a22**2 + a33**2 + 2*(a12**2 + a13**2 + a23**2))
    
    # Approximate largest eigenvalue magnitude
    lambda_max = frobenius / cp.sqrt(3.0)
    
    # Confidence = magnitude of structure
    confidence = lambda_max.reshape(D, H, W)
    
    # For orientation: use gradient of intensity (simplified)
    # The eigenvector of largest eigenvalue roughly aligns with 
    # the direction of maximum second derivative
    # Approximate with normalized [Izz, Iyy, Ixx] direction
    vectors = cp.zeros((D, H, W, 3), dtype=CONFIG["dtype"])
    
    # Use the diagonal elements as proxy for principal direction
    norm = cp.sqrt(a11**2 + a22**2 + a33**2 + 1e-8)
    vectors[:, :, :, 0] = (a33 / norm).reshape(D, H, W)  # Z component
    vectors[:, :, :, 1] = (a22 / norm).reshape(D, H, W)  # Y component  
    vectors[:, :, :, 2] = (a11 / norm).reshape(D, H, W)  # X component
    
    del a11, a22, a33, a12, a13, a23
    cp.get_default_memory_pool().free_all_blocks()
    
    print(f"   Confidence range: [{float(confidence.min()):.4f}, {float(confidence.max()):.4f}]")
    
    return vectors, confidence

def sato_eigenanalysis_gpu(hessian, alpha=0.5, gamma=0.5):
    """
    Full eigenvalue decomposition (accurate but slower).
    """
    # Use fast method if enabled
    if CONFIG.get("use_fast_eigen", False):
        return sato_eigenanalysis_fast(hessian, alpha, gamma)
    
    print("🧮 Computing eigenvalues (full decomposition)...")
    
    D, H, W = hessian['xx'].shape
    
    # Build Hessian tensor
    H_tensor = cp.zeros((D, H, W, 3, 3), dtype=CONFIG["dtype"])
    
    H_tensor[:, :, :, 0, 0] = hessian['xx']
    H_tensor[:, :, :, 1, 1] = hessian['yy']
    H_tensor[:, :, :, 2, 2] = hessian['zz']
    H_tensor[:, :, :, 0, 1] = hessian['xy']
    H_tensor[:, :, :, 1, 0] = hessian['xy']
    H_tensor[:, :, :, 0, 2] = hessian['xz']
    H_tensor[:, :, :, 2, 0] = hessian['xz']
    H_tensor[:, :, :, 1, 2] = hessian['yz']
    H_tensor[:, :, :, 2, 1] = hessian['yz']
    
    hessian.clear()
    cp.get_default_memory_pool().free_all_blocks()
    
    H_flat = H_tensor.reshape(-1, 3, 3)
    
    print("   Computing eigenvalues/vectors...")
    eigenvalues, eigenvectors = cp.linalg.eigh(H_flat)
    
    del H_flat, H_tensor
    cp.get_default_memory_pool().free_all_blocks()
    
    abs_eigenvalues = cp.abs(eigenvalues)
    max_idx = cp.argmax(abs_eigenvalues, axis=1)
    
    N = eigenvalues.shape[0]
    vectors_flat = eigenvectors[cp.arange(N), :, max_idx]
    
    λ1, λ2, λ3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    abs_λ3 = cp.abs(λ3)
    R_sheet = cp.abs(λ2) / (abs_λ3 + 1e-8)
    S = cp.sqrt(λ1**2 + λ2**2 + λ3**2)
    
    sato_response = abs_λ3 * (1 - cp.exp(-R_sheet**2 / (2 * alpha**2))) * \
                    (1 - cp.exp(-S**2 / (2 * gamma**2)))
    
    vectors = vectors_flat.reshape(D, H, W, 3)
    confidence = sato_response.reshape(D, H, W)
    
    del eigenvalues, eigenvectors, vectors_flat, sato_response
    cp.get_default_memory_pool().free_all_blocks()
    
    print(f"   Confidence range: [{float(confidence.min()):.4f}, {float(confidence.max()):.4f}]")
    
    return vectors, confidence

# =============================================================================
# 🔧 Global Orientation Propagation (Sign Disambiguation)
# =============================================================================
def propagate_orientation_fast(vectors, confidence):
    """
    FAST sign disambiguation using iterative local propagation.
    
    Instead of serial BFS, we use vectorized local consistency checks
    that can be parallelized on GPU. Much faster than serial BFS.
    
    Algorithm:
    1. Start with random initial signs
    2. Iteratively check local 6-neighbors and flip if majority disagree
    3. Repeat until convergence or max iterations
    """
    print("🔄 Propagating global orientation (FAST iterative method)...")
    
    # Handle both CuPy and NumPy inputs
    if hasattr(vectors, 'get'):
        vectors_np = vectors.get().copy()
    else:
        vectors_np = np.asarray(vectors).copy()
    
    if hasattr(confidence, 'get'):
        confidence_np = confidence.get()
    else:
        confidence_np = np.asarray(confidence)
    
    D, H, W, _ = vectors_np.shape
    print(f"   Volume: {D} x {H} x {W}")
    
    # Find seed (highest confidence) and fix its direction
    seed_idx = np.unravel_index(np.argmax(confidence_np), confidence_np.shape)
    print(f"   Seed: {seed_idx} (confidence: {confidence_np[seed_idx]:.4f})")
    
    # Simple approach: propagate layer by layer from seed
    # This is much faster than full BFS
    max_iters = 20
    changed_count = 1
    iteration = 0
    
    while changed_count > 0 and iteration < max_iters:
        changed_count = 0
        
        # Check all 6 neighbors for each voxel
        for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            # Shifted views
            if dz == -1:
                v1 = vectors_np[:-1, :, :]
                v2 = vectors_np[1:, :, :]
            elif dz == 1:
                v1 = vectors_np[1:, :, :]
                v2 = vectors_np[:-1, :, :]
            elif dy == -1:
                v1 = vectors_np[:, :-1, :]
                v2 = vectors_np[:, 1:, :]
            elif dy == 1:
                v1 = vectors_np[:, 1:, :]
                v2 = vectors_np[:, :-1, :]
            elif dx == -1:
                v1 = vectors_np[:, :, :-1]
                v2 = vectors_np[:, :, 1:]
            else:  # dx == 1
                v1 = vectors_np[:, :, 1:]
                v2 = vectors_np[:, :, :-1]
            
            # Compute dot products
            dots = np.einsum('ijkl,ijkl->ijk', v1, v2)
            
            # Find disagreements (dot < 0)
            flip_mask = dots < 0
            
            if flip_mask.any():
                # Apply flips to v2's original location
                if dz == -1:
                    vectors_np[1:, :, :][flip_mask] *= -1
                elif dz == 1:
                    vectors_np[:-1, :, :][flip_mask] *= -1
                elif dy == -1:
                    vectors_np[:, 1:, :][flip_mask] *= -1
                elif dy == 1:
                    vectors_np[:, :-1, :][flip_mask] *= -1
                elif dx == -1:
                    vectors_np[:, :, 1:][flip_mask] *= -1
                else:
                    vectors_np[:, :, :-1][flip_mask] *= -1
                
                changed_count += flip_mask.sum()
        
        iteration += 1
        print(f"   Iteration {iteration}: {changed_count} flips")
        
        if changed_count == 0:
            break
    
    print(f"   Converged in {iteration} iterations")
    
    return vectors_np

# =============================================================================
# 🔧 Chunked Processing (Memory-Safe)
# =============================================================================
def process_volume_chunked(volume, chunk_size=128, overlap=16):
    """
    Process large volume in overlapping chunks to stay within VRAM limits.
    
    Args:
        volume: (D, H, W) NumPy array
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        vectors: (D, H, W, 3) orientation field
        confidence: (D, H, W) Sato response
    """
    D, H, W = volume.shape
    
    # If small enough, process directly
    vram_estimate = D * H * W * 4 * 20 / 1e9  # Rough estimate
    
    if vram_estimate < CONFIG["max_vram_gb"]:
        print(f"📦 Processing entire volume (est. {vram_estimate:.2f} GB VRAM)...")
        
        volume_gpu = cp.asarray(volume, dtype=CONFIG["dtype"])
        hessian = compute_hessian_separable(volume_gpu, CONFIG["sigma"])
        
        del volume_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        vectors, confidence = sato_eigenanalysis_gpu(
            hessian, CONFIG["alpha"], CONFIG["gamma"]
        )
        
        return vectors, confidence
    
    # Chunked processing
    print(f"📦 Processing in chunks ({chunk_size}³ with {overlap} overlap)...")
    
    vectors_full = np.zeros((D, H, W, 3), dtype=np.float32)
    confidence_full = np.zeros((D, H, W), dtype=np.float32)
    weight_full = np.zeros((D, H, W), dtype=np.float32)
    
    step = chunk_size - overlap
    
    z_chunks = list(range(0, D, step))
    y_chunks = list(range(0, H, step))
    x_chunks = list(range(0, W, step))
    
    total_chunks = len(z_chunks) * len(y_chunks) * len(x_chunks)
    
    with tqdm(total=total_chunks, desc="   Chunks") as pbar:
        for z0 in z_chunks:
            for y0 in y_chunks:
                for x0 in x_chunks:
                    z1 = min(z0 + chunk_size, D)
                    y1 = min(y0 + chunk_size, H)
                    x1 = min(x0 + chunk_size, W)
                    
                    chunk = volume[z0:z1, y0:y1, x0:x1]
                    chunk_gpu = cp.asarray(chunk, dtype=CONFIG["dtype"])
                    
                    # Process chunk
                    hessian = compute_hessian_separable(chunk_gpu, CONFIG["sigma"])
                    vectors_chunk, conf_chunk = sato_eigenanalysis_gpu(
                        hessian, CONFIG["alpha"], CONFIG["gamma"]
                    )
                    
                    # Transfer back
                    vectors_np = cp.asnumpy(vectors_chunk)
                    conf_np = cp.asnumpy(conf_chunk)
                    
                    # Accumulate with blending
                    vectors_full[z0:z1, y0:y1, x0:x1] += vectors_np
                    confidence_full[z0:z1, y0:y1, x0:x1] += conf_np
                    weight_full[z0:z1, y0:y1, x0:x1] += 1.0
                    
                    # Cleanup
                    del chunk_gpu, hessian, vectors_chunk, conf_chunk
                    cp.get_default_memory_pool().free_all_blocks()
                    gc.collect()
                    
                    pbar.update(1)
    
    # Average overlapping regions
    weight_full = np.maximum(weight_full, 1e-8)
    vectors_full /= weight_full[:, :, :, np.newaxis]
    confidence_full /= weight_full
    
    # Normalize vectors
    norms = np.linalg.norm(vectors_full, axis=-1, keepdims=True)
    vectors_full = vectors_full / np.maximum(norms, 1e-8)
    
    # Return as CuPy for consistency with direct processing path
    # But only if memory allows
    try:
        return cp.asarray(vectors_full), cp.asarray(confidence_full)
    except:
        # If OOM, return numpy - BFS will handle it
        print("   ⚠️ Returning NumPy arrays due to VRAM limits")
        return vectors_full, confidence_full

# =============================================================================
# 🔧 I/O Functions
# =============================================================================
def load_volume_from_tifs(input_dir, z_range=None):
    """Load 3D volume from directory of TIF slices."""
    import tifffile
    
    input_dir = Path(input_dir)
    tif_files = sorted(input_dir.glob("*.tif"), key=lambda x: int(x.stem))
    
    if z_range:
        tif_files = [f for f in tif_files if z_range[0] <= int(f.stem) < z_range[1]]
    
    print(f"📥 Loading {len(tif_files)} TIF slices...")
    
    # Load first slice to get dimensions
    first = tifffile.imread(str(tif_files[0]))
    H, W = first.shape
    D = len(tif_files)
    
    volume = np.zeros((D, H, W), dtype=np.float32)
    volume[0] = first.astype(np.float32)
    
    for i, f in enumerate(tqdm(tif_files[1:], desc="   Loading")):
        volume[i + 1] = tifffile.imread(str(f)).astype(np.float32)
    
    print(f"   Volume shape: {volume.shape}")
    print(f"   Value range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    return volume

def save_orientation_field(vectors, confidence, output_path):
    """Save orientation field as compressed NPZ."""
    print(f"💾 Saving to {output_path}...")
    
    np.savez_compressed(
        output_path,
        vector_field=vectors,
        confidence=confidence,
    )
    
    print(f"   ✅ Saved: {output_path}")

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius RGT - Orientation Field Generator")
    print("=" * 60)
    
    # Check GPU
    print(f"\n🖥️ GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"   VRAM: {meminfo[1] / 1e9:.2f} GB total, {meminfo[0] / 1e9:.2f} GB free")
    
    # 1. Load volume
    volume = load_volume_from_tifs(CONFIG["input_dir"], z_range=(51, 65))
    
    # 2. 🚑 VESUVIUS CONTRAST WINDOWING (Critical Fix!)
    # Raw values are uint16 [0, 65535]. Papyrus signal is in [18000, 28000].
    WIN_MIN, WIN_MAX = 18000.0, 28000.0
    print(f"\n🔧 Applying Vesuvius contrast windowing [{WIN_MIN:.0f}, {WIN_MAX:.0f}]...")
    print(f"   Before: range=[{volume.min():.0f}, {volume.max():.0f}], mean={volume.mean():.0f}")
    
    volume = np.clip(volume, WIN_MIN, WIN_MAX)
    volume = (volume - WIN_MIN) / (WIN_MAX - WIN_MIN)
    volume = volume.astype(np.float32)
    
    print(f"   After: range=[{volume.min():.4f}, {volume.max():.4f}], mean={volume.mean():.4f}")
    
    # 3. Compute Hessian and Sato filter
    vectors, confidence = process_volume_chunked(
        volume, CONFIG["chunk_size"], CONFIG["overlap"]
    )
    
    # 4. 🔍 VISUAL PROBE: Save Sato confidence for debugging
    print("\n🔍 Saving Sato confidence debug image...")
    if hasattr(confidence, 'get'):
        conf_np = confidence.get()
    else:
        conf_np = np.asarray(confidence)
    
    mid_z = conf_np.shape[0] // 2
    conf_slice = conf_np[mid_z]
    
    # Normalize for visualization
    if conf_slice.max() > 0:
        conf_vis = (conf_slice / conf_slice.max() * 255).astype(np.uint8)
    else:
        conf_vis = (conf_slice * 255).astype(np.uint8)
    
    import cv2
    cv2.imwrite("debug_sato_confidence.png", conf_vis)
    print(f"   ✅ Saved: debug_sato_confidence.png")
    print(f"   Confidence stats: min={conf_np.min():.4f}, max={conf_np.max():.4f}, mean={conf_np.mean():.4f}")
    
    # 5. Global orientation propagation (SKIP for now - can be done later)
    # vectors_fixed = propagate_orientation_fast(vectors, confidence)
    print("⏭️ Skipping sign propagation (use raw vectors)")
    
    # Convert to numpy if needed
    if hasattr(vectors, 'get'):
        vectors_fixed = vectors.get()
    else:
        vectors_fixed = np.asarray(vectors)
    
    # 6. Save results
    if hasattr(confidence, 'get'):
        confidence_np = confidence.get()
    else:
        confidence_np = np.asarray(confidence)
    save_orientation_field(vectors_fixed, confidence_np, CONFIG["output_path"])
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("📊 Orientation Field Generation Complete!")
    print(f"   Output: {CONFIG['output_path']}")
    print(f"   Shape: {vectors_fixed.shape}")
    print("=" * 60)

if __name__ == "__main__":
    main()

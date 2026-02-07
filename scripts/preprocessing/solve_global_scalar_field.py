"""
Vesuvius Challenge 2026 - RGT Scalar Field Solver (GPU ONLY)
=============================================================
Phase 2: Solve weighted Poisson equation for global scalar field.

Equation: ∇·(w∇φ) = ∇·(wv)

Method: Matrix-free CG with Jacobi Preconditioner (PURE CUPY)
Target: RTX 3060 6GB VRAM
"""

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator, cg
from cupyx.scipy.ndimage import zoom as gpu_zoom
from pathlib import Path
import gc

# =============================================================================
# ⚙️ Configuration
# =============================================================================
CONFIG = {
    "input_path": "orientation_field.npz",
    "output_path": "scalar_field.npz",
    "debug_slice_path": "debug_scalar_slice.png",
    
    "tol": 1e-4,
    "maxiter": 500,
    "print_interval": 25,
    
    "epsilon": 1e-5,
    "downsample_factor": 4,
}

# =============================================================================
# 🔧 GPU Finite Difference Operators
# =============================================================================
def gradient_x_gpu(phi, w=None):
    """Forward difference in X (GPU)"""
    grad = cp.zeros_like(phi)
    grad[..., :-1] = phi[..., 1:] - phi[..., :-1]
    if w is not None:
        grad *= w
    return grad

def gradient_y_gpu(phi, w=None):
    """Forward difference in Y (GPU)"""
    grad = cp.zeros_like(phi)
    grad[:, :-1, :] = phi[:, 1:, :] - phi[:, :-1, :]
    if w is not None:
        grad *= w
    return grad

def gradient_z_gpu(phi, w=None):
    """Forward difference in Z (GPU)"""
    grad = cp.zeros_like(phi)
    grad[:-1, :, :] = phi[1:, :, :] - phi[:-1, :, :]
    if w is not None:
        grad *= w
    return grad

def divergence_gpu(fx, fy, fz):
    """Compute divergence (GPU, backward differences)"""
    div = cp.zeros_like(fx)
    
    div[..., 1:] += fx[..., 1:] - fx[..., :-1]
    div[..., 0] += fx[..., 0]
    
    div[:, 1:, :] += fy[:, 1:, :] - fy[:, :-1, :]
    div[:, 0, :] += fy[:, 0, :]
    
    div[1:, :, :] += fz[1:, :, :] - fz[:-1, :, :]
    div[0, :, :] += fz[0, :, :]
    
    return div

# =============================================================================
# 🔧 GPU Matrix-Free Linear Operator
# =============================================================================
class GPUWeightedLaplacian:
    """Pure CuPy matrix-free weighted Laplacian."""
    
    def __init__(self, shape, weights_gpu, epsilon=1e-5):
        self.shape = shape
        self.n = int(cp.prod(cp.array(shape)))
        self.dtype = cp.float32
        self.w = weights_gpu  # Already on GPU
        self.epsilon = epsilon
        
        # Jacobi diagonal
        self.diag = cp.maximum(6.0 * self.w + epsilon, 1e-8)
        
    def matvec(self, phi_flat):
        """A @ φ = -∇·(w∇φ) + ε*φ (GPU)"""
        phi = phi_flat.reshape(self.shape)
        
        gx = gradient_x_gpu(phi, self.w)
        gy = gradient_y_gpu(phi, self.w)
        gz = gradient_z_gpu(phi, self.w)
        
        result = -divergence_gpu(gx, gy, gz)
        result += self.epsilon * phi
        
        del gx, gy, gz
        
        return result.ravel()
    
    def precond(self, r_flat):
        """Jacobi preconditioner (GPU)"""
        return (r_flat.reshape(self.shape) / self.diag).ravel()
    
    def as_operator(self):
        return LinearOperator(
            shape=(self.n, self.n),
            matvec=self.matvec,
            dtype=self.dtype
        )
    
    def as_precond(self):
        return LinearOperator(
            shape=(self.n, self.n),
            matvec=self.precond,
            dtype=self.dtype
        )

# =============================================================================
# 🔧 CPU Downsampling (before GPU transfer)
# =============================================================================
def downsample_cpu(arr, factor, is_vector=False):
    """Downsample on CPU using scipy zoom."""
    from scipy.ndimage import zoom as cpu_zoom
    
    if is_vector:
        D, H, W, C = arr.shape
        result = np.zeros((D, H // factor, W // factor, C), dtype=np.float32)
        for c in range(C):
            result[:, :, :, c] = cpu_zoom(arr[:, :, :, c], (1, 1/factor, 1/factor), order=1)
        # Re-normalize
        norms = np.linalg.norm(result, axis=-1, keepdims=True)
        result = result / np.maximum(norms, 1e-8)
        return result
    else:
        return cpu_zoom(arr, (1, 1/factor, 1/factor), order=1).astype(np.float32)

# =============================================================================
# 🔧 CG Callback
# =============================================================================
class CGCallback:
    def __init__(self, interval=25):
        self.i = 0
        self.interval = interval
        
    def __call__(self, xk):
        self.i += 1
        if self.i % self.interval == 0:
            print(f"   Iteration {self.i}...", flush=True)

# =============================================================================
# 🚀 Main
# =============================================================================
def main():
    print("=" * 60)
    print("🔬 Vesuvius RGT - Scalar Field Solver (GPU ONLY)")
    print("=" * 60)
    
    # GPU check
    print(f"\n🖥️ GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"   VRAM: {meminfo[1]/1e9:.2f} GB total, {meminfo[0]/1e9:.2f} GB free")
    
    # 1. Load on CPU
    print(f"\n📥 Loading {CONFIG['input_path']}...")
    data = np.load(CONFIG['input_path'])
    vectors = data['vector_field']
    confidence = data['confidence']
    
    print(f"   Original: {vectors.shape}, conf: {confidence.shape}")
    original_shape = confidence.shape
    
    # 2. Downsample on CPU (more memory efficient)
    factor = CONFIG["downsample_factor"]
    print(f"\n📉 Downsampling by {factor}x on CPU...")
    
    vectors_ds = downsample_cpu(vectors, factor, is_vector=True)
    weights_ds = downsample_cpu(confidence, factor)
    
    del vectors, confidence, data
    gc.collect()
    
    D, H, W = weights_ds.shape
    n = D * H * W
    print(f"   Downsampled: ({D}, {H}, {W}) = {n:,} voxels")
    print(f"   Est. GPU memory: {n * 4 * 6 / 1e9:.2f} GB")
    
    # 3. Transfer to GPU
    print("\n🔄 Transferring to GPU...")
    
    vectors_gpu = cp.asarray(vectors_ds, dtype=cp.float32)
    weights_gpu = cp.asarray(weights_ds, dtype=cp.float32)
    weights_gpu = cp.maximum(weights_gpu, 1e-8)
    
    del vectors_ds, weights_ds
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"   VRAM after transfer: {meminfo[0]/1e9:.2f} GB free")
    
    # 4. Compute RHS on GPU
    print("\n📐 Computing RHS: b = ∇·(w·v) on GPU...")
    
    vx = vectors_gpu[:, :, :, 2]
    vy = vectors_gpu[:, :, :, 1]
    vz = vectors_gpu[:, :, :, 0]
    
    w_vx = weights_gpu * vx
    w_vy = weights_gpu * vy
    w_vz = weights_gpu * vz
    
    b = divergence_gpu(w_vx, w_vy, w_vz)
    b_flat = b.ravel()
    
    print(f"   RHS range: [{float(b.min()):.6f}, {float(b.max()):.6f}]")
    
    del vectors_gpu, vx, vy, vz, w_vx, w_vy, w_vz, b
    cp.get_default_memory_pool().free_all_blocks()
    
    meminfo = cp.cuda.runtime.memGetInfo()
    print(f"   VRAM after RHS: {meminfo[0]/1e9:.2f} GB free")
    
    # 5. Build operator
    print("\n🔧 Building GPU LinearOperator...")
    op = GPUWeightedLaplacian((D, H, W), weights_gpu, CONFIG["epsilon"])
    A = op.as_operator()
    M = op.as_precond()
    
    # 6. Initial guess
    x0 = cp.zeros(n, dtype=cp.float32)
    
    # 7. Solve CG on GPU
    print(f"\n🧮 Solving Poisson (CG on GPU)...")
    print(f"   tol={CONFIG['tol']}, maxiter={CONFIG['maxiter']}")
    
    callback = CGCallback(CONFIG["print_interval"])
    
    phi_flat, info = cg(A, b_flat, x0=x0, tol=CONFIG["tol"], 
                        maxiter=CONFIG["maxiter"], M=M, callback=callback)
    
    if info == 0:
        print(f"   ✅ Converged in {callback.i} iterations!")
    else:
        print(f"   ⚠️ info={info}, iterations={callback.i}")
    
    # 8. Reshape and transfer to CPU
    phi_ds = phi_flat.reshape((D, H, W))
    phi_ds_np = cp.asnumpy(phi_ds)
    
    print(f"\n📊 Solution stats:")
    print(f"   Range: [{phi_ds_np.min():.4f}, {phi_ds_np.max():.4f}]")
    print(f"   Mean: {phi_ds_np.mean():.4f}")
    
    # 9. Upsample to original resolution
    print(f"\n📈 Upsampling to original resolution...")
    from scipy.ndimage import zoom as cpu_zoom
    phi_full = cpu_zoom(phi_ds_np, (1, factor, factor), order=1)
    phi_full = phi_full[:original_shape[0], :original_shape[1], :original_shape[2]]
    print(f"   Full shape: {phi_full.shape}")
    
    # 10. Save
    print(f"\n💾 Saving {CONFIG['output_path']}...")
    np.savez_compressed(CONFIG['output_path'], phi=phi_full)
    print(f"   ✅ Saved!")
    
    # 11. Debug viz
    import cv2
    mid_z = phi_full.shape[0] // 2
    s = phi_full[mid_z]
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 1e-8:
        vis = ((s - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        vis = np.zeros_like(s, dtype=np.uint8)
    cv2.imwrite(CONFIG['debug_slice_path'], vis)
    print(f"   ✅ Debug: {CONFIG['debug_slice_path']}")
    
    print("\n" + "=" * 60)
    print("📊 Scalar Field Solver Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

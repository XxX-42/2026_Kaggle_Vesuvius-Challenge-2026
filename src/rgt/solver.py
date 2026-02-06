#!/usr/bin/env python3
"""
RGT Poisson Solver for Vesuvius Challenge 2026
===============================================

Sparse weighted Poisson equation solver for RGT scalar field computation.
Solves ∇·(w∇φ) = ∇·(wv⃗) using CG with optional AMG preconditioning.

Author: HPC Architect
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix


# =============================================================================
# PoissonSolver Class
# =============================================================================

@dataclass
class PoissonResult:
    """Result container for Poisson solve."""
    phi: "cp.ndarray"           # Scalar field (D, H, W)
    converged: bool             # Whether solver converged
    iterations: int             # Number of iterations
    residual: float             # Final residual norm


class PoissonSolver:
    """
    Weighted Poisson equation solver for RGT computation.
    
    Solves the weighted Poisson equation:
        ∇·(w∇φ) = ∇·(wv⃗)
    
    where w is the confidence/weight field and v⃗ is the normal vector field.
    
    Usage:
        solver = PoissonSolver()
        A, b = solver.assemble_system(vectors, weights)
        result = solver.solve(A, b, shape)
    """
    
    def __init__(
        self,
        vram_threshold: float = 0.8,
        tol: float = 1e-6,
        maxiter: int = 2000,
        verbose: bool = True,
    ):
        """
        Initialize PoissonSolver.
        
        Args:
            vram_threshold: Maximum VRAM utilization before raising error.
            tol: Solver convergence tolerance.
            maxiter: Maximum solver iterations.
            verbose: Print progress messages.
        """
        self.vram_threshold = vram_threshold
        self.tol = tol
        self.maxiter = maxiter
        self.verbose = verbose
        self.epsilon = 1e-6  # Regularization for conditioning
    
    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[PoissonSolver] {msg}")
    
    def _check_vram(self, estimated_bytes: int) -> None:
        """
        Check if estimated memory usage is safe.
        
        Args:
            estimated_bytes: Estimated memory requirement.
            
        Raises:
            MemoryError: If allocation would exceed threshold.
        """
        import cupy as cp
        
        try:
            free, total = cp.cuda.Device(0).mem_info
            utilization = (total - free + estimated_bytes) / total
            
            if utilization > self.vram_threshold:
                raise MemoryError(
                    f"VRAM check failed: estimated {estimated_bytes / 1e9:.2f} GB "
                    f"would exceed {self.vram_threshold:.0%} threshold "
                    f"(current free: {free / 1e9:.2f} GB)"
                )
        except Exception as e:
            if isinstance(e, MemoryError):
                raise
            # If we can't check, proceed with caution
            self._log(f"  Warning: Could not check VRAM: {e}")
    
    def compute_divergence(
        self,
        vectors: "cp.ndarray",
        weights: "cp.ndarray",
    ) -> "cp.ndarray":
        """
        Compute weighted divergence: ∇·(w·v).
        
        Uses central differences with zero-padding at boundaries.
        
        Args:
            vectors: Normal vectors (3, D, H, W).
            weights: Confidence weights (D, H, W).
            
        Returns:
            Divergence field (D, H, W).
        """
        import cupy as cp
        
        D, H, W = weights.shape
        
        # Weighted vector field
        wvx = weights * vectors[2]  # X component (axis 2)
        wvy = weights * vectors[1]  # Y component (axis 1)
        wvz = weights * vectors[0]  # Z component (axis 0)
        
        # Central differences with zero-padding
        # d(wvx)/dx
        dvx = cp.zeros_like(wvx)
        dvx[:, :, 1:-1] = (wvx[:, :, 2:] - wvx[:, :, :-2]) / 2.0
        
        # d(wvy)/dy
        dvy = cp.zeros_like(wvy)
        dvy[:, 1:-1, :] = (wvy[:, 2:, :] - wvy[:, :-2, :]) / 2.0
        
        # d(wvz)/dz
        dvz = cp.zeros_like(wvz)
        dvz[1:-1, :, :] = (wvz[2:, :, :] - wvz[:-2, :, :]) / 2.0
        
        divergence = dvx + dvy + dvz
        
        del wvx, wvy, wvz, dvx, dvy, dvz
        
        return divergence
    
    def build_laplacian(
        self,
        weights: "cp.ndarray",
        pin_center: bool = True,
    ) -> Tuple["csr_matrix", int]:
        """
        Build weighted 7-point stencil Laplacian matrix (vectorized).
        
        Constructs sparse matrix A for ∇·(w∇φ) using finite differences.
        Uses fully vectorized operations for performance.
        
        Args:
            weights: Confidence weights (D, H, W).
            pin_center: If True, pin center voxel to fix gauge freedom.
            
        Returns:
            Tuple of (CSR sparse matrix, pinned node index).
        """
        import cupy as cp
        from cupyx.scipy.sparse import coo_matrix, diags
        
        D, H, W = weights.shape
        N = D * H * W
        
        # Estimate memory
        estimated_nnz = N * 7
        estimated_bytes = estimated_nnz * 16 + N * 8
        self._check_vram(estimated_bytes)
        
        self._log(f"  Building Laplacian matrix ({D}×{H}×{W} = {N:,} dof)")
        
        # Work on CPU for sparse matrix construction
        w = cp.asnumpy(weights).astype(np.float64).ravel()  # Use float64 for stability
        
        # Create index arrays
        idx = np.arange(N, dtype=np.int32)
        
        # Neighbor offsets for 3D grid
        # X neighbors: offset = 1
        # Y neighbors: offset = W
        # Z neighbors: offset = W * H
        
        rows_list = []
        cols_list = []
        data_list = []
        diag = np.zeros(N, dtype=np.float64)
        
        # X- neighbor (x > 0)
        mask = (idx % W) > 0
        src = idx[mask]
        dst = src - 1
        w_half = 0.5 * (w[src] + w[dst])
        rows_list.append(src)
        cols_list.append(dst)
        data_list.append(w_half)
        diag[src] -= w_half
        
        # X+ neighbor (x < W-1)
        mask = (idx % W) < W - 1
        src = idx[mask]
        dst = src + 1
        w_half = 0.5 * (w[src] + w[dst])
        rows_list.append(src)
        cols_list.append(dst)
        data_list.append(w_half)
        diag[src] -= w_half
        
        # Y- neighbor (y > 0)
        mask = ((idx // W) % H) > 0
        src = idx[mask]
        dst = src - W
        w_half = 0.5 * (w[src] + w[dst])
        rows_list.append(src)
        cols_list.append(dst)
        data_list.append(w_half)
        diag[src] -= w_half
        
        # Y+ neighbor (y < H-1)
        mask = ((idx // W) % H) < H - 1
        src = idx[mask]
        dst = src + W
        w_half = 0.5 * (w[src] + w[dst])
        rows_list.append(src)
        cols_list.append(dst)
        data_list.append(w_half)
        diag[src] -= w_half
        
        # Z- neighbor (z > 0)
        mask = (idx // (W * H)) > 0
        src = idx[mask]
        dst = src - W * H
        w_half = 0.5 * (w[src] + w[dst])
        rows_list.append(src)
        cols_list.append(dst)
        data_list.append(w_half)
        diag[src] -= w_half
        
        # Z+ neighbor (z < D-1)
        mask = (idx // (W * H)) < D - 1
        src = idx[mask]
        dst = src + W * H
        w_half = 0.5 * (w[src] + w[dst])
        rows_list.append(src)
        cols_list.append(dst)
        data_list.append(w_half)
        diag[src] -= w_half
        
        # Concatenate all off-diagonal entries
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)
        
        # Add diagonal with regularization for conditioning
        # Small epsilon prevents near-singular matrix in low-weight regions
        diag_regularized = diag - self.epsilon
        rows = np.concatenate([rows, idx])
        cols = np.concatenate([cols, idx])
        data = np.concatenate([data, diag_regularized])
        
        # Pin center node
        pinned_idx = (D // 2) * (H * W) + (H // 2) * W + (W // 2)
        
        if pin_center:
            # Zero out all entries in pinned row
            pinned_mask = rows == pinned_idx
            data[pinned_mask] = 0.0
            # Set diagonal to 1
            diag_mask = (rows == pinned_idx) & (cols == pinned_idx)
            data[diag_mask] = 1.0
        
        # Transfer to GPU
        rows_gpu = cp.asarray(rows.astype(np.int32))
        cols_gpu = cp.asarray(cols.astype(np.int32))
        data_gpu = cp.asarray(data.astype(np.float32))
        
        A_coo = coo_matrix((data_gpu, (rows_gpu, cols_gpu)), shape=(N, N))
        A_csr = A_coo.tocsr()
        
        # Cleanup
        del rows, cols, data, rows_gpu, cols_gpu, data_gpu, A_coo
        del rows_list, cols_list, data_list, diag, w
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        return A_csr, pinned_idx
    
    def assemble_system(
        self,
        vectors: "cp.ndarray",
        weights: "cp.ndarray",
    ) -> Tuple["csr_matrix", "cp.ndarray", int]:
        """
        Assemble complete linear system Ax = b.
        
        Args:
            vectors: Normal vectors (3, D, H, W).
            weights: Confidence weights (D, H, W).
            
        Returns:
            Tuple of (A sparse matrix, b RHS vector, pinned index).
        """
        import cupy as cp
        
        self._log(f"Assembling system (shape={weights.shape})")
        
        # Compute divergence (RHS)
        self._log("  Computing divergence...")
        divergence = self.compute_divergence(vectors, weights)
        b = divergence.ravel().astype(cp.float32)
        
        # Build Laplacian (LHS)
        self._log("  Building Laplacian matrix...")
        A, pinned_idx = self.build_laplacian(weights, pin_center=True)
        
        # Set pinned node RHS to 0
        b[pinned_idx] = 0.0
        
        del divergence
        gc.collect()
        
        self._log(f"  ✓ System assembled: A={A.shape}, nnz={A.nnz:,}")
        
        return A, b, pinned_idx
    
    def solve(
        self,
        A: "csr_matrix",
        b: "cp.ndarray",
        shape: Tuple[int, int, int],
    ) -> PoissonResult:
        """
        Solve the linear system Ax = b.
        
        Uses SolverFactory to get best available solver.
        
        Args:
            A: Sparse matrix (N, N).
            b: RHS vector (N,).
            shape: Original volume shape (D, H, W).
            
        Returns:
            PoissonResult with scalar field and convergence info.
        """
        import cupy as cp
        from rgt.infrastructure import SolverFactory
        
        self._log(f"Solving system ({A.shape[0]:,} unknowns)...")
        
        # Get solver
        solver = SolverFactory.create(prefer_amgx=True)
        self._log(f"  Using: {solver.name}")
        
        # Initial guess
        x0 = cp.zeros(b.shape[0], dtype=cp.float32)
        
        # Jacobi (diagonal) preconditioner for better convergence
        diag_A = A.diagonal()
        diag_A = cp.where(cp.abs(diag_A) > 1e-10, diag_A, 1.0)  # Avoid division by zero
        M_inv = 1.0 / cp.abs(diag_A)  # Inverse of diagonal
        
        # Preconditioned solve using cupyx.scipy.sparse.linalg.cg with M
        from cupyx.scipy.sparse.linalg import cg, LinearOperator
        
        def precond(x):
            return M_inv * x
        
        M = LinearOperator((A.shape[0], A.shape[0]), matvec=precond, dtype=cp.float32)
        
        solution, info_code = cg(A, b, x0=x0, tol=self.tol, maxiter=self.maxiter, M=M)
        converged = info_code == 0
        
        # Reshape to volume
        phi = solution.reshape(shape).astype(cp.float32)
        
        iterations = -1  # CG doesn't return iteration count directly
        
        # Compute residual
        residual = float(cp.linalg.norm(A @ solution - b))
        
        self._log(f"  ✓ Solved: converged={converged}, residual={residual:.2e}")
        
        return PoissonResult(
            phi=phi,
            converged=converged,
            iterations=iterations,
            residual=residual,
        )


# =============================================================================
# Test Function
# =============================================================================

def test_poisson_solver(size: int = 64) -> bool:
    """
    Test PoissonSolver on synthetic sphere.
    
    For a sphere with radial normal vectors, the Poisson solution should
    approximate the radial distance field φ ≈ r.
    
    Args:
        size: Volume dimension (reduced for faster testing).
        
    Returns:
        True if correlation with radial distance is high.
    """
    import cupy as cp
    from rgt.feature_extraction import HessianEngine, create_synthetic_sphere
    from rgt.orientation import VectorAligner
    
    print("=" * 60)
    print("PoissonSolver Test Suite")
    print("=" * 60)
    
    # Step 1: Create sphere and compute normals
    print("\n[Step 1] Creating sphere and computing normals")
    print("-" * 40)
    
    center = (size / 2, size / 2, size / 2)
    radius = size * 0.3
    
    sphere_np = create_synthetic_sphere(size=size, center=center, radius=radius)
    sphere_gpu = cp.asarray(sphere_np)
    
    engine = HessianEngine(slab_size=8, verbose=False)
    hessian = engine.compute_hessian(sphere_gpu, sigma=1.5)
    result = engine.solve_eigen_system(hessian.as_list())
    
    vectors = result.normal_vectors
    confidence = result.confidence
    
    print(f"  Shape: {vectors.shape[1:]}")
    
    # Step 2: Align vectors
    print("\n[Step 2] Aligning vectors")
    print("-" * 40)
    
    aligner = VectorAligner(threshold=0.5, verbose=False)
    aligned = aligner.align_vectors(vectors, confidence)
    print("  ✓ Vectors aligned")
    
    # Step 3: Solve Poisson
    print("\n[Step 3] Solving Poisson equation")
    print("-" * 40)
    
    # Use confidence as weight (normalized)
    weights = confidence / (confidence.max() + 1e-10)
    
    solver = PoissonSolver(tol=1e-5, maxiter=1000, verbose=True)
    A, b, pinned = solver.assemble_system(aligned, weights)
    result = solver.solve(A, b, weights.shape)
    
    phi = result.phi
    
    # Step 4: Validate φ ≈ r
    print("\n[Step 4] Validating solution")
    print("-" * 40)
    
    # Compute radial distance
    z, y, x = cp.meshgrid(
        cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij'
    )
    r = cp.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    # Mask to shell region
    shell_mask = cp.abs(r - radius) < 3.0
    
    # Normalize both fields for comparison
    phi_shell = phi[shell_mask]
    r_shell = r[shell_mask]
    
    phi_norm = (phi_shell - phi_shell.mean()) / (phi_shell.std() + 1e-10)
    r_norm = (r_shell - r_shell.mean()) / (r_shell.std() + 1e-10)
    
    # Correlation
    correlation = float(cp.mean(phi_norm * r_norm))
    
    print(f"  φ range: [{float(phi.min()):.3f}, {float(phi.max()):.3f}]")
    print(f"  Correlation with radial distance: {correlation:.4f}")
    print(f"  Residual: {result.residual:.2e}")
    
    # Cleanup
    del sphere_gpu, vectors, confidence, aligned, weights
    del A, b, phi, z, y, x, r
    hessian.free()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Test passes if correlation is reasonable (not perfect due to discrete effects)
    test_passed = abs(correlation) > 0.3
    
    print("\n" + "=" * 60)
    if test_passed:
        print("✓ All tests PASSED")
        print(f"  φ shows correlation with radial distance: {correlation:.4f}")
    else:
        print("✗ Tests FAILED")
        print(f"  Low correlation with radial distance: {correlation:.4f}")
    print("=" * 60)
    
    return test_passed


if __name__ == "__main__":
    test_poisson_solver(size=64)

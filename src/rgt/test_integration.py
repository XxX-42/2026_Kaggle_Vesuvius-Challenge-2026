#!/usr/bin/env python3
"""
RGT Integration Test for Vesuvius Challenge 2026
=================================================

Full pipeline test: Hessian → Alignment → Poisson → Validation.

Author: HPC Architect
"""

from __future__ import annotations

import gc

import numpy as np


def test_full_pipeline(size: int = 64) -> bool:
    """
    Test complete RGT pipeline on synthetic sphere.
    
    Pipeline:
        1. Create synthetic sphere
        2. Compute Hessian and extract normals
        3. Align vector field orientation
        4. Solve Poisson equation
        5. Validate φ ≈ radial distance
    
    Args:
        size: Volume dimension.
        
    Returns:
        True if all tests pass.
    """
    import cupy as cp
    from rgt import (
        HessianEngine,
        VectorAligner,
        PoissonSolver,
    )
    from rgt.feature_extraction import create_synthetic_sphere
    
    print("=" * 70)
    print("RGT Full Pipeline Integration Test")
    print("=" * 70)
    
    center = (size / 2, size / 2, size / 2)
    radius = size * 0.3
    
    # =========================================================================
    # Step 1: Create Synthetic Sphere
    # =========================================================================
    print("\n[Step 1/5] Creating Synthetic Sphere")
    print("-" * 50)
    
    sphere_np = create_synthetic_sphere(
        size=size, center=center, radius=radius, shell_thickness=3.0
    )
    sphere_gpu = cp.asarray(sphere_np)
    print(f"  Volume: {sphere_gpu.shape}")
    print(f"  Center: {center}, Radius: {radius:.1f}")
    del sphere_np
    
    # =========================================================================
    # Step 2: Hessian Feature Extraction
    # =========================================================================
    print("\n[Step 2/5] Hessian Feature Extraction")
    print("-" * 50)
    
    engine = HessianEngine(slab_size=8, verbose=False)
    hessian = engine.compute_hessian(sphere_gpu, sigma=1.5)
    eigen_result = engine.solve_eigen_system(hessian.as_list())
    
    vectors = eigen_result.normal_vectors
    confidence = eigen_result.confidence
    
    print(f"  Normal vectors: {vectors.shape}")
    print(f"  Confidence range: [{float(confidence.min()):.2f}, {float(confidence.max()):.2f}]")
    
    # =========================================================================
    # Step 3: Vector Field Alignment
    # =========================================================================
    print("\n[Step 3/5] Vector Field Alignment")
    print("-" * 50)
    
    # Simulate corruption: flip 50% of vectors
    np.random.seed(42)
    flip_mask = cp.asarray(np.random.random(vectors.shape[1:]) < 0.5)
    corrupted = vectors.copy()
    for i in range(3):
        corrupted[i] = cp.where(flip_mask, -corrupted[i], corrupted[i])
    
    n_flipped = int(flip_mask.sum())
    print(f"  Corrupted: {n_flipped:,} vectors flipped (50%)")
    
    aligner = VectorAligner(threshold=0.5, verbose=False)
    aligned = aligner.align_vectors(corrupted, confidence)
    
    # Check alignment quality
    dot = cp.sum(aligned * vectors, axis=0)
    align_error = cp.arccos(cp.clip(cp.abs(dot), 0, 1)) * 180 / cp.pi
    
    z, y, x = cp.meshgrid(cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij')
    r = cp.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    shell_mask = cp.abs(r - radius) < 2.0
    
    align_median = float(cp.median(align_error[shell_mask]))
    print(f"  Alignment restored: median error = {align_median:.2f}°")
    
    del corrupted, flip_mask, dot, align_error
    
    # =========================================================================
    # Step 4: Poisson Solve
    # =========================================================================
    print("\n[Step 4/5] Poisson Equation Solve")
    print("-" * 50)
    
    # Normalize weights
    weights = confidence / (confidence.max() + 1e-10)
    
    solver = PoissonSolver(tol=1e-5, maxiter=1000, verbose=False)
    A, b, pinned = solver.assemble_system(aligned, weights)
    
    print(f"  Matrix: {A.shape[0]:,} × {A.shape[1]:,}, nnz = {A.nnz:,}")
    
    poisson_result = solver.solve(A, b, weights.shape)
    phi = poisson_result.phi
    
    print(f"  Converged: {poisson_result.converged}")
    print(f"  Residual: {poisson_result.residual:.2e}")
    
    # =========================================================================
    # Step 5: Validation
    # =========================================================================
    print("\n[Step 5/5] Validation: φ ≈ Radial Distance")
    print("-" * 50)
    
    # Correlation between φ and r on shell
    phi_shell = phi[shell_mask]
    r_shell = r[shell_mask]
    
    phi_norm = (phi_shell - phi_shell.mean()) / (phi_shell.std() + 1e-10)
    r_norm = (r_shell - r_shell.mean()) / (r_shell.std() + 1e-10)
    
    correlation = float(cp.mean(phi_norm * r_norm))
    
    print(f"  φ range: [{float(phi.min()):.3f}, {float(phi.max()):.3f}]")
    print(f"  Correlation(φ, r): {correlation:.4f}")
    
    # Cleanup
    del sphere_gpu, vectors, confidence, aligned, weights
    del A, b, phi, z, y, x, r, shell_mask
    del phi_shell, r_shell, phi_norm, r_norm
    hessian.free()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # =========================================================================
    # Results
    # =========================================================================
    alignment_ok = align_median < 5.0
    correlation_ok = abs(correlation) > 0.3
    convergence_ok = poisson_result.converged
    
    all_passed = alignment_ok and correlation_ok and convergence_ok
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print(f"  [{'✓' if alignment_ok else '✗'}] Vector alignment: {align_median:.2f}° median error")
    print(f"  [{'✓' if convergence_ok else '✗'}] Poisson convergence: {poisson_result.converged}")
    print(f"  [{'✓' if correlation_ok else '✗'}] φ-r correlation: {correlation:.4f}")
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    test_full_pipeline(size=64)

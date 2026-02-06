#!/usr/bin/env python3
"""
RGT Full Pipeline Test for Vesuvius Challenge 2026
====================================================

End-to-end test: Hessian → Alignment → Poisson → Mesh → Curvature Audit.

Author: HPC Architect
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np


def test_full_pipeline(size: int = 64) -> bool:
    """
    Test complete RGT pipeline from raw volume to mesh.
    
    Pipeline:
        1. Create synthetic sphere
        2. Compute Hessian and extract normals
        3. Align vector field orientation
        4. Solve Poisson equation
        5. Extract isosurface mesh
        6. Audit Gaussian curvature
    
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
        SurfaceExtractor,
    )
    from rgt.feature_extraction import create_synthetic_sphere
    
    print("=" * 70)
    print("RGT Full Pipeline Test: Hessian → Mesh")
    print("=" * 70)
    
    center = (size / 2, size / 2, size / 2)
    radius = size * 0.3
    theoretical_K = 1.0 / (radius ** 2)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # =========================================================================
    # Step 1: Create Synthetic Sphere
    # =========================================================================
    print("\n[Step 1/6] Creating Synthetic Sphere")
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
    print("\n[Step 2/6] Hessian Feature Extraction")
    print("-" * 50)
    
    engine = HessianEngine(slab_size=8, verbose=False)
    hessian = engine.compute_hessian(sphere_gpu, sigma=1.5)
    eigen_result = engine.solve_eigen_system(hessian.as_list())
    
    vectors = eigen_result.normal_vectors
    confidence = eigen_result.confidence
    
    print(f"  Normal vectors: {vectors.shape}")
    
    # =========================================================================
    # Step 3: Vector Field Alignment
    # =========================================================================
    print("\n[Step 3/6] Vector Field Alignment")
    print("-" * 50)
    
    # Flip 50% to simulate sign ambiguity
    np.random.seed(42)
    flip_mask = cp.asarray(np.random.random(vectors.shape[1:]) < 0.5)
    corrupted = vectors.copy()
    for i in range(3):
        corrupted[i] = cp.where(flip_mask, -corrupted[i], corrupted[i])
    
    aligner = VectorAligner(threshold=0.5, verbose=False)
    aligned = aligner.align_vectors(corrupted, confidence)
    
    print(f"  ✓ Alignment complete")
    del corrupted, flip_mask
    
    # =========================================================================
    # Step 4: Poisson Solve
    # =========================================================================
    print("\n[Step 4/6] Poisson Equation Solve")
    print("-" * 50)
    
    weights = confidence / (confidence.max() + 1e-10)
    
    solver = PoissonSolver(tol=1e-5, maxiter=1000, verbose=False)
    A, b, _ = solver.assemble_system(aligned, weights)
    poisson_result = solver.solve(A, b, weights.shape)
    phi = poisson_result.phi
    
    print(f"  Converged: {poisson_result.converged}")
    print(f"  Residual: {poisson_result.residual:.2e}")
    print(f"  φ range: [{float(phi.min()):.2f}, {float(phi.max()):.2f}]")
    
    del A, b, aligned, weights
    
    # =========================================================================
    # Step 5: Mesh Extraction
    # =========================================================================
    print("\n[Step 5/6] Isosurface Extraction")
    print("-" * 50)
    
    extractor = SurfaceExtractor(curvature_threshold=0.05, verbose=True)
    
    # Extract at mean φ level
    iso_level = float(phi.mean())
    mesh = extractor.extract_layer(phi, iso_level=iso_level)
    
    # =========================================================================
    # Step 6: Curvature Audit
    # =========================================================================
    print("\n[Step 6/6] Curvature Quality Audit")
    print("-" * 50)
    
    audit = extractor.audit_curvature(mesh, theoretical_K=theoretical_K)
    
    # Save mesh
    mesh_path = output_dir / "rgt_sphere_mesh.obj"
    extractor.save_mesh(mesh, str(mesh_path))
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    del sphere_gpu, vectors, confidence, phi
    hessian.free()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # =========================================================================
    # Results
    # =========================================================================
    poisson_ok = poisson_result.converged
    mesh_ok = mesh.n_vertices > 100
    curvature_ok = audit.defect_ratio < 0.5  # Less than 50% defects
    
    all_passed = poisson_ok and mesh_ok and curvature_ok
    
    print("\n" + "=" * 70)
    print("Test Results:")
    print(f"  [{'✓' if poisson_ok else '✗'}] Poisson convergence: {poisson_result.converged}")
    print(f"  [{'✓' if mesh_ok else '✗'}] Mesh extraction: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
    print(f"  [{'✓' if curvature_ok else '✗'}] Curvature audit: {audit.defect_ratio:.1%} defects")
    print(f"      Mean K: {audit.mean_K:.6f} (expected: {theoretical_K:.6f})")
    print("=" * 70)
    
    if all_passed:
        print("✓ ALL PIPELINE TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    
    print(f"\nMesh saved to: {mesh_path.absolute()}")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    test_full_pipeline(size=64)

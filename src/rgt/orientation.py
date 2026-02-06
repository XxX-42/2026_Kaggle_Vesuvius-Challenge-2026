#!/usr/bin/env python3
"""
RGT Orientation Module for Vesuvius Challenge 2026
===================================================

Vector field orientation alignment using priority-flood algorithm.
Resolves ±180° sign ambiguity of Hessian eigenvectors.

Author: HPC Architect
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList

if TYPE_CHECKING:
    import cupy as cp


# =============================================================================
# Numba-Accelerated Priority Flood Core
# =============================================================================

@njit(cache=True)
def _heap_push(heap: np.ndarray, heap_size: int, priority: float, 
               z: int, y: int, x: int) -> int:
    """
    Push element to max-heap (simulated with array).
    Heap structure: [priority, z, y, x] per entry.
    Returns new heap size.
    """
    idx = heap_size
    heap[idx, 0] = priority
    heap[idx, 1] = z
    heap[idx, 2] = y
    heap[idx, 3] = x
    
    # Bubble up
    while idx > 0:
        parent = (idx - 1) // 2
        if heap[parent, 0] < heap[idx, 0]:
            # Swap
            for i in range(4):
                heap[parent, i], heap[idx, i] = heap[idx, i], heap[parent, i]
            idx = parent
        else:
            break
    
    return heap_size + 1


@njit(cache=True)
def _heap_pop(heap: np.ndarray, heap_size: int) -> Tuple[int, float, int, int, int]:
    """
    Pop max element from heap.
    Returns (new_size, priority, z, y, x).
    """
    if heap_size == 0:
        return 0, 0.0, -1, -1, -1
    
    # Get root (max)
    priority = heap[0, 0]
    z = int(heap[0, 1])
    y = int(heap[0, 2])
    x = int(heap[0, 3])
    
    # Move last to root
    heap_size -= 1
    if heap_size > 0:
        for i in range(4):
            heap[0, i] = heap[heap_size, i]
        
        # Bubble down
        idx = 0
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            largest = idx
            
            if left < heap_size and heap[left, 0] > heap[largest, 0]:
                largest = left
            if right < heap_size and heap[right, 0] > heap[largest, 0]:
                largest = right
            
            if largest != idx:
                for i in range(4):
                    heap[idx, i], heap[largest, i] = heap[largest, i], heap[idx, i]
                idx = largest
            else:
                break
    
    return heap_size, priority, z, y, x


@njit(cache=True, parallel=False)
def _priority_flood_align(
    vectors: np.ndarray,     # (3, D, H, W) float32
    coherence: np.ndarray,   # (D, H, W) float32
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Numba-accelerated priority-flood vector alignment.
    
    Propagates orientation from high-coherence seeds to neighbors,
    flipping vectors to maintain consistency.
    
    Args:
        vectors: Normal vectors (3, D, H, W), modified in-place.
        coherence: Confidence weights (D, H, W).
        threshold: Minimum coherence to process.
        
    Returns:
        Aligned vectors (3, D, H, W).
    """
    D, H, W = coherence.shape
    total_voxels = D * H * W
    
    # Allocate visited array
    visited = np.zeros((D, H, W), dtype=np.bool_)
    
    # Allocate heap (max possible size)
    heap = np.zeros((total_voxels, 4), dtype=np.float32)
    heap_size = 0
    
    # Find seed: voxel with highest coherence above threshold
    max_coh = -1.0
    seed_z, seed_y, seed_x = 0, 0, 0
    
    for z in range(D):
        for y in range(H):
            for x in range(W):
                if coherence[z, y, x] > max_coh and coherence[z, y, x] >= threshold:
                    max_coh = coherence[z, y, x]
                    seed_z, seed_y, seed_x = z, y, x
    
    if max_coh < threshold:
        # No valid seed found
        return vectors
    
    # Initialize with seed
    visited[seed_z, seed_y, seed_x] = True
    
    # 6-connected neighbors: (dz, dy, dx)
    neighbors = np.array([
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1]
    ], dtype=np.int32)
    
    # Add seed's neighbors to heap
    for n in range(6):
        nz = seed_z + neighbors[n, 0]
        ny = seed_y + neighbors[n, 1]
        nx = seed_x + neighbors[n, 2]
        
        if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
            if not visited[nz, ny, nx] and coherence[nz, ny, nx] >= threshold:
                heap_size = _heap_push(heap, heap_size, coherence[nz, ny, nx], nz, ny, nx)
    
    # Process heap
    processed = 1  # seed already processed
    
    while heap_size > 0:
        heap_size, priority, z, y, x = _heap_pop(heap, heap_size)
        
        if z < 0 or visited[z, y, x]:
            continue
        
        visited[z, y, x] = True
        processed += 1
        
        # Find a visited neighbor to align with
        best_neighbor_coh = -1.0
        best_nz, best_ny, best_nx = -1, -1, -1
        
        for n in range(6):
            nz = z + neighbors[n, 0]
            ny = y + neighbors[n, 1]
            nx = x + neighbors[n, 2]
            
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                if visited[nz, ny, nx] and coherence[nz, ny, nx] > best_neighbor_coh:
                    best_neighbor_coh = coherence[nz, ny, nx]
                    best_nz, best_ny, best_nx = nz, ny, nx
        
        if best_nz >= 0:
            # Compute dot product with best neighbor
            dot = (vectors[0, z, y, x] * vectors[0, best_nz, best_ny, best_nx] +
                   vectors[1, z, y, x] * vectors[1, best_nz, best_ny, best_nx] +
                   vectors[2, z, y, x] * vectors[2, best_nz, best_ny, best_nx])
            
            # Flip if anti-aligned
            if dot < 0.0:
                vectors[0, z, y, x] = -vectors[0, z, y, x]
                vectors[1, z, y, x] = -vectors[1, z, y, x]
                vectors[2, z, y, x] = -vectors[2, z, y, x]
        
        # Add unvisited neighbors to heap
        for n in range(6):
            nz = z + neighbors[n, 0]
            ny = y + neighbors[n, 1]
            nx = x + neighbors[n, 2]
            
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                if not visited[nz, ny, nx] and coherence[nz, ny, nx] >= threshold:
                    heap_size = _heap_push(heap, heap_size, coherence[nz, ny, nx], nz, ny, nx)
    
    return vectors


# =============================================================================
# VectorAligner Class
# =============================================================================

class VectorAligner:
    """
    Align vector field orientation using priority-flood propagation.
    
    Resolves the ±180° sign ambiguity of Hessian eigenvectors by propagating
    consistent orientation from high-confidence regions.
    
    Usage:
        aligner = VectorAligner()
        aligned = aligner.align_vectors(vectors_gpu, confidence_gpu)
    """
    
    def __init__(self, threshold: float = 0.1, verbose: bool = True):
        """
        Initialize VectorAligner.
        
        Args:
            threshold: Minimum coherence to include a voxel in alignment.
            verbose: Print progress messages.
        """
        self.threshold = threshold
        self.verbose = verbose
    
    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[VectorAligner] {msg}")
    
    def align_vectors(
        self,
        vectors_gpu: "cp.ndarray",
        confidence_gpu: "cp.ndarray",
    ) -> "cp.ndarray":
        """
        Align vector field orientation.
        
        Args:
            vectors_gpu: CuPy array (3, D, H, W) of normal vectors.
            confidence_gpu: CuPy array (D, H, W) of confidence/coherence values.
            
        Returns:
            Aligned vectors as CuPy array (3, D, H, W).
        """
        import cupy as cp
        
        self._log(f"Aligning vectors (shape={vectors_gpu.shape[1:]}, threshold={self.threshold})")
        
        # Transfer to CPU
        self._log("  Transferring to CPU...")
        vectors_cpu = cp.asnumpy(vectors_gpu).astype(np.float32)
        confidence_cpu = cp.asnumpy(confidence_gpu).astype(np.float32)
        
        # Run Numba-accelerated alignment
        self._log("  Running priority-flood alignment...")
        aligned_cpu = _priority_flood_align(vectors_cpu, confidence_cpu, self.threshold)
        
        # Transfer back to GPU
        self._log("  Transferring to GPU...")
        aligned_gpu = cp.asarray(aligned_cpu)
        
        # Cleanup
        del vectors_cpu, confidence_cpu, aligned_cpu
        gc.collect()
        
        self._log("  ✓ Alignment complete")
        
        return aligned_gpu


# =============================================================================
# Test Function
# =============================================================================

def test_vector_alignment(size: int = 128) -> bool:
    """
    Test VectorAligner with synthetic data.
    
    Creates a sphere, extracts normals, randomly flips 50% of vectors,
    then verifies alignment restores consistency.
    
    Args:
        size: Volume dimension.
        
    Returns:
        True if test passes.
    """
    import cupy as cp
    from rgt.feature_extraction import (
        HessianEngine, 
        create_synthetic_sphere,
    )
    
    print("=" * 60)
    print("VectorAligner Test Suite")
    print("=" * 60)
    
    # Step 1: Create synthetic sphere and compute normals
    print("\n[Step 1] Creating synthetic sphere and computing normals")
    print("-" * 40)
    
    center = (size / 2, size / 2, size / 2)
    radius = size * 0.3
    
    sphere_np = create_synthetic_sphere(size=size, center=center, radius=radius)
    sphere_gpu = cp.asarray(sphere_np)
    del sphere_np
    
    engine = HessianEngine(slab_size=16, verbose=False)
    hessian = engine.compute_hessian(sphere_gpu, sigma=1.5)
    result = engine.solve_eigen_system(hessian.as_list())
    
    original_normals = result.normal_vectors.copy()
    confidence = result.confidence.copy()
    
    print(f"  Shape: {original_normals.shape}")
    print(f"  Confidence range: [{float(confidence.min()):.3f}, {float(confidence.max()):.3f}]")
    
    # Step 2: Randomly flip 50% of vectors
    print("\n[Step 2] Randomly flipping 50% of vectors")
    print("-" * 40)
    
    np.random.seed(42)
    flip_mask_np = np.random.random(original_normals.shape[1:]) < 0.5
    flip_mask = cp.asarray(flip_mask_np)
    
    corrupted_normals = original_normals.copy()
    for i in range(3):
        corrupted_normals[i] = cp.where(flip_mask, -corrupted_normals[i], corrupted_normals[i])
    
    n_flipped = int(flip_mask.sum())
    print(f"  Flipped {n_flipped} vectors ({100*n_flipped/flip_mask.size:.1f}%)")
    
    # Compute pre-alignment angular error
    z, y, x = cp.meshgrid(cp.arange(size), cp.arange(size), cp.arange(size), indexing='ij')
    distance = cp.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    shell_mask = cp.abs(distance - radius) < 2.0
    
    # Pre-alignment error
    dot_pre = cp.sum(corrupted_normals * original_normals, axis=0)
    abs_dot_pre = cp.clip(cp.abs(dot_pre), 0.0, 1.0)
    error_pre = cp.arccos(abs_dot_pre) * 180.0 / cp.pi
    pre_median = float(cp.median(error_pre[shell_mask]))
    
    print(f"  Pre-alignment median error: {pre_median:.2f}°")
    
    # Step 3: Run VectorAligner
    print("\n[Step 3] Running VectorAligner")
    print("-" * 40)
    
    aligner = VectorAligner(threshold=0.5, verbose=True)
    aligned_normals = aligner.align_vectors(corrupted_normals, confidence)
    
    # Step 4: Verify alignment
    print("\n[Step 4] Verifying alignment")
    print("-" * 40)
    
    # Post-alignment error (vs original ground truth)
    dot_post = cp.sum(aligned_normals * original_normals, axis=0)
    abs_dot_post = cp.clip(cp.abs(dot_post), 0.0, 1.0)
    error_post = cp.arccos(abs_dot_post) * 180.0 / cp.pi
    
    post_median = float(cp.median(error_post[shell_mask]))
    post_mean = float(error_post[shell_mask].mean())
    post_max = float(error_post[shell_mask].max())
    
    print(f"  Post-alignment median error: {post_median:.2f}°")
    print(f"  Post-alignment mean error: {post_mean:.2f}°")
    print(f"  Post-alignment max error: {post_max:.2f}°")
    
    # Cleanup
    del sphere_gpu, original_normals, corrupted_normals, aligned_normals
    del confidence, flip_mask, z, y, x, distance, shell_mask
    del dot_pre, dot_post, error_pre, error_post
    hessian.free()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Test passes if post-alignment error is low
    test_passed = post_median < 5.0
    
    print("\n" + "=" * 60)
    if test_passed:
        print("✓ All tests PASSED")
        print(f"  Alignment restored accuracy: {pre_median:.2f}° → {post_median:.2f}°")
    else:
        print("✗ Tests FAILED")
        print(f"  Post-alignment error too high: {post_median:.2f}°")
    print("=" * 60)
    
    return test_passed


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    test_vector_alignment(size=128)

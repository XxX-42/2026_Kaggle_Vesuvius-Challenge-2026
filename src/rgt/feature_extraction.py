#!/usr/bin/env python3
"""
RGT Feature Extraction for Vesuvius Challenge 2026
===================================================

Hessian-based micro-feature extraction using separable Gaussian derivatives.
Implements eigenvalue decomposition for normal vector computation on sheet-like
structures (papyrus layers).

Author: HPC Architect
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

import numpy as np

# Lazy imports for GPU libraries
if TYPE_CHECKING:
    import cupy as cp


# =============================================================================
# Hessian Engine: Separable Gaussian Derivative-Based Feature Extraction
# =============================================================================

@dataclass
class HessianResult:
    """Result container for Hessian computation."""
    Ixx: "cp.ndarray"
    Iyy: "cp.ndarray"
    Izz: "cp.ndarray"
    Ixy: "cp.ndarray"
    Ixz: "cp.ndarray"
    Iyz: "cp.ndarray"
    
    def as_list(self) -> List["cp.ndarray"]:
        """Return components as ordered list [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]."""
        return [self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Ixz, self.Iyz]
    
    def free(self) -> None:
        """Explicitly free GPU memory for all components."""
        import cupy as cp
        
        for attr in ['Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz']:
            arr = getattr(self, attr, None)
            if arr is not None:
                del arr
        
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


@dataclass
class EigenResult:
    """Result container for eigenvalue decomposition."""
    normal_vectors: "cp.ndarray"  # Shape: (3, D, H, W)
    confidence: "cp.ndarray"       # Shape: (D, H, W)


class HessianEngine:
    """
    GPU-accelerated Hessian matrix computation using separable Gaussian derivatives.
    
    Implements memory-safe batch processing for eigenvalue decomposition to
    prevent VRAM overflow on consumer GPUs.
    
    Usage:
        engine = HessianEngine()
        hessian = engine.compute_hessian(volume, sigma=1.0)
        result = engine.solve_eigen_system(hessian.as_list())
        # result.normal_vectors contains the extracted surface normals
    """
    
    def __init__(self, slab_size: int = 16, verbose: bool = True):
        """
        Initialize HessianEngine.
        
        Args:
            slab_size: Number of Z-slices to process at once in eigen decomposition.
                       Smaller = less VRAM, larger = faster.
            verbose: Print progress messages.
        """
        self.slab_size = slab_size
        self.verbose = verbose
    
    def _log(self, msg: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[HessianEngine] {msg}")
    
    def compute_hessian(
        self,
        volume: "cp.ndarray",
        sigma: float = 1.0,
    ) -> HessianResult:
        """
        Compute 6 unique Hessian matrix components using separable 1D Gaussian filters.
        
        Uses cupyx.scipy.ndimage.gaussian_filter1d exclusively (no 3D kernels).
        
        Args:
            volume: 3D CuPy array of shape (D, H, W), dtype will be cast to float32.
            sigma: Gaussian scale parameter (standard deviation in voxels).
            
        Returns:
            HessianResult containing Ixx, Iyy, Izz, Ixy, Ixz, Iyz components.
            
        Math (axis convention: 0=Z, 1=Y, 2=X):
            Ixx: order=2 on axis=2, order=0 on axis=1, order=0 on axis=0
            Iyy: order=0 on axis=2, order=2 on axis=1, order=0 on axis=0
            Izz: order=0 on axis=2, order=0 on axis=1, order=2 on axis=0
            Ixy: order=1 on axis=2, order=1 on axis=1, order=0 on axis=0
            Ixz: order=1 on axis=2, order=0 on axis=1, order=1 on axis=0
            Iyz: order=0 on axis=2, order=1 on axis=1, order=1 on axis=0
        """
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter1d
        
        # Force float32 for memory efficiency
        if volume.dtype != cp.float32:
            self._log(f"Casting volume from {volume.dtype} to float32")
            volume = volume.astype(cp.float32)
        
        self._log(f"Computing Hessian components (sigma={sigma}, shape={volume.shape})")
        
        # Helper: apply separable Gaussian derivative
        def separable_derivative(data: cp.ndarray, orders: Tuple[int, int, int]) -> cp.ndarray:
            """
            Apply separable Gaussian derivative.
            orders[0] = Z-axis order, orders[1] = Y-axis order, orders[2] = X-axis order
            """
            result = data
            for axis, order in enumerate(orders):
                result = gaussian_filter1d(result, sigma=sigma, axis=axis, order=order)
            return result.astype(cp.float32)
        
        # Compute all 6 Hessian components
        # Axis mapping: axis=0 is Z, axis=1 is Y, axis=2 is X
        
        # Second-order partial derivatives (diagonal elements)
        Ixx = separable_derivative(volume, (0, 0, 2))  # d²/dx²
        self._log("  ✓ Ixx computed")
        
        Iyy = separable_derivative(volume, (0, 2, 0))  # d²/dy²
        self._log("  ✓ Iyy computed")
        
        Izz = separable_derivative(volume, (2, 0, 0))  # d²/dz²
        self._log("  ✓ Izz computed")
        
        # Mixed partial derivatives (off-diagonal elements)
        Ixy = separable_derivative(volume, (0, 1, 1))  # d²/dxdy
        self._log("  ✓ Ixy computed")
        
        Ixz = separable_derivative(volume, (1, 0, 1))  # d²/dxdz
        self._log("  ✓ Ixz computed")
        
        Iyz = separable_derivative(volume, (1, 1, 0))  # d²/dydz
        self._log("  ✓ Iyz computed")
        
        return HessianResult(Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixy=Ixy, Ixz=Ixz, Iyz=Iyz)
    
    def _slab_generator(
        self,
        depth: int,
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Generate (start, end) indices for slab-by-slab processing.
        
        Args:
            depth: Total depth (Z dimension) of the volume.
            
        Yields:
            Tuples of (start_idx, end_idx) for each slab.
        """
        for start in range(0, depth, self.slab_size):
            end = min(start + self.slab_size, depth)
            yield start, end
    
    def solve_eigen_system(
        self,
        hessian_components: List["cp.ndarray"],
    ) -> EigenResult:
        """
        Perform eigenvalue decomposition on Hessian matrices with batch processing.
        
        Memory-safe implementation that processes volume in Z-slabs to prevent
        VRAM overflow. For each voxel, constructs symmetric 3x3 Hessian matrix
        and computes eigenvalues/eigenvectors.
        
        Args:
            hessian_components: List of 6 CuPy arrays [Ixx, Iyy, Izz, Ixy, Ixz, Iyz].
                                Each array has shape (D, H, W).
                                
        Returns:
            EigenResult containing:
                - normal_vectors: (3, D, H, W) - eigenvector of largest |eigenvalue|
                - confidence: (D, H, W) - ratio of largest to second largest |eigenvalue|
                
        Note:
            For sheet-like structures, the normal direction corresponds to the
            eigenvector of the eigenvalue with largest absolute value (direction
            of maximum intensity variation).
        """
        import cupy as cp
        
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = hessian_components
        D, H, W = Ixx.shape
        
        self._log(f"Solving eigen system (shape={Ixx.shape}, slab_size={self.slab_size})")
        
        # Pre-allocate output arrays (float32)
        normal_vectors = cp.zeros((3, D, H, W), dtype=cp.float32)
        confidence = cp.zeros((D, H, W), dtype=cp.float32)
        
        total_slabs = (D + self.slab_size - 1) // self.slab_size
        
        for slab_idx, (z_start, z_end) in enumerate(self._slab_generator(D)):
            slab_depth = z_end - z_start
            
            self._log(f"  Processing slab {slab_idx + 1}/{total_slabs} (z={z_start}:{z_end})")
            
            # Extract slab data for all components
            slab_Ixx = Ixx[z_start:z_end]
            slab_Iyy = Iyy[z_start:z_end]
            slab_Izz = Izz[z_start:z_end]
            slab_Ixy = Ixy[z_start:z_end]
            slab_Ixz = Ixz[z_start:z_end]
            slab_Iyz = Iyz[z_start:z_end]
            
            # Reshape for batch eigenvalue computation
            # Target shape: (N, 3, 3) where N = slab_depth * H * W
            N = slab_depth * H * W
            
            # Construct symmetric 3x3 Hessian matrices
            # H = [[Ixx, Ixy, Ixz],
            #      [Ixy, Iyy, Iyz],
            #      [Ixz, Iyz, Izz]]
            hessian_batch = cp.zeros((N, 3, 3), dtype=cp.float32)
            
            flat_Ixx = slab_Ixx.ravel()
            flat_Iyy = slab_Iyy.ravel()
            flat_Izz = slab_Izz.ravel()
            flat_Ixy = slab_Ixy.ravel()
            flat_Ixz = slab_Ixz.ravel()
            flat_Iyz = slab_Iyz.ravel()
            
            # Fill symmetric matrix
            hessian_batch[:, 0, 0] = flat_Ixx
            hessian_batch[:, 1, 1] = flat_Iyy
            hessian_batch[:, 2, 2] = flat_Izz
            hessian_batch[:, 0, 1] = flat_Ixy
            hessian_batch[:, 1, 0] = flat_Ixy  # Symmetric
            hessian_batch[:, 0, 2] = flat_Ixz
            hessian_batch[:, 2, 0] = flat_Ixz  # Symmetric
            hessian_batch[:, 1, 2] = flat_Iyz
            hessian_batch[:, 2, 1] = flat_Iyz  # Symmetric
            
            # Clean up temporary flat arrays
            del flat_Ixx, flat_Iyy, flat_Izz, flat_Ixy, flat_Ixz, flat_Iyz
            del slab_Ixx, slab_Iyy, slab_Izz, slab_Ixy, slab_Ixz, slab_Iyz
            
            # Eigenvalue decomposition using eigh (optimized for symmetric matrices)
            # eigenvalues shape: (N, 3), eigenvectors shape: (N, 3, 3)
            # eigh returns eigenvalues in ascending order
            eigenvalues, eigenvectors = cp.linalg.eigh(hessian_batch)
            
            # Clean up Hessian batch
            del hessian_batch
            
            # Find eigenvector corresponding to largest |eigenvalue|
            # eigenvalues are in ascending order, so check first and last
            abs_eigenvalues = cp.abs(eigenvalues)
            
            # Get index of maximum absolute eigenvalue (0, 1, or 2)
            max_idx = cp.argmax(abs_eigenvalues, axis=1)  # Shape: (N,)
            
            # Extract the corresponding eigenvector for each voxel
            # eigenvectors[:, :, i] is the eigenvector for eigenvalue i
            n_range = cp.arange(N)
            selected_eigenvectors = eigenvectors[n_range, :, max_idx]  # Shape: (N, 3)
            
            # Compute confidence: ratio of largest to second largest |eigenvalue|
            sorted_abs = cp.sort(abs_eigenvalues, axis=1)
            largest = sorted_abs[:, 2]
            second_largest = sorted_abs[:, 1]
            # Avoid division by zero
            slab_confidence = largest / (second_largest + 1e-10)
            
            # Clean up
            del eigenvalues, eigenvectors, abs_eigenvalues, sorted_abs
            del largest, second_largest, max_idx, n_range
            
            # Reshape and store results
            # Eigenvector components are in Hessian matrix order: [X, Y, Z]
            # But we store as volume order: [Z, Y, X] to match indexing convention
            normal_vectors[0, z_start:z_end] = selected_eigenvectors[:, 2].reshape(slab_depth, H, W)  # Z
            normal_vectors[1, z_start:z_end] = selected_eigenvectors[:, 1].reshape(slab_depth, H, W)  # Y
            normal_vectors[2, z_start:z_end] = selected_eigenvectors[:, 0].reshape(slab_depth, H, W)  # X
            confidence[z_start:z_end] = slab_confidence.reshape(slab_depth, H, W)
            
            # Clean up slab results
            del selected_eigenvectors, slab_confidence
            
            # Force garbage collection
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        
        self._log("  ✓ Eigen system solved")
        
        return EigenResult(normal_vectors=normal_vectors, confidence=confidence)


# =============================================================================
# Synthetic Test Suite
# =============================================================================

def create_synthetic_sphere(
    size: int = 128,
    center: Optional[Tuple[float, float, float]] = None,
    radius: float = 40.0,
    shell_thickness: float = 5.0,
) -> np.ndarray:
    """
    Create a synthetic hollow sphere for testing normal vector extraction.
    
    The sphere has a bright shell (simulating a sheet-like structure).
    Expected normal vectors should point toward the sphere center.
    
    Args:
        size: Volume dimension (creates size^3 cube).
        center: Sphere center (z, y, x). Default: center of volume.
        radius: Sphere radius in voxels.
        shell_thickness: Thickness of the bright shell.
        
    Returns:
        3D numpy array (size, size, size) with float32 dtype.
    """
    if center is None:
        center = (size / 2, size / 2, size / 2)
    
    z, y, x = np.meshgrid(
        np.arange(size),
        np.arange(size),
        np.arange(size),
        indexing='ij'
    )
    
    # Distance from center
    distance = np.sqrt(
        (z - center[0])**2 + 
        (y - center[1])**2 + 
        (x - center[2])**2
    )
    
    # Create shell: bright at radius, fading towards inside/outside
    # Gaussian profile centered at radius
    shell = np.exp(-((distance - radius) ** 2) / (2 * shell_thickness**2))
    
    return shell.astype(np.float32)


def create_synthetic_plane(
    size: int = 128,
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    offset: float = 64.0,
    thickness: float = 5.0,
) -> np.ndarray:
    """
    Create a synthetic plane for testing normal vector extraction.
    
    The plane has a bright sheet perpendicular to the given normal direction.
    
    Args:
        size: Volume dimension (creates size^3 cube).
        normal: Plane normal direction (z, y, x), will be normalized.
        offset: Distance from origin along normal.
        thickness: Plane thickness (Gaussian sigma).
        
    Returns:
        3D numpy array (size, size, size) with float32 dtype.
    """
    # Normalize normal vector
    normal = np.array(normal, dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    
    z, y, x = np.meshgrid(
        np.arange(size),
        np.arange(size),
        np.arange(size),
        indexing='ij'
    )
    
    # Signed distance from plane
    distance = z * normal[0] + y * normal[1] + x * normal[2] - offset
    
    # Gaussian profile
    plane = np.exp(-(distance**2) / (2 * thickness**2))
    
    return plane.astype(np.float32)


def test_hessian_synthetic(size: int = 128) -> bool:
    """
    Test HessianEngine on synthetic data.
    
    Creates a synthetic sphere and verifies that extracted normal vectors
    point toward the sphere center.
    
    Args:
        size: Volume dimension for synthetic sphere (default: 128 for VRAM safety).
        
    Returns:
        True if all tests pass.
    """
    import cupy as cp
    
    print("=" * 60)
    print("HessianEngine Synthetic Test Suite")
    print("=" * 60)
    
    # Test 1: Create synthetic sphere
    print("\n[Test 1] Creating synthetic sphere")
    print("-" * 40)
    center = (size / 2, size / 2, size / 2)
    radius = size * 0.3  # 30% of volume size
    
    sphere_np = create_synthetic_sphere(
        size=size,
        center=center,
        radius=radius,
        shell_thickness=3.0,
    )
    print(f"  Sphere shape: {sphere_np.shape}")
    print(f"  Sphere center: {center}")
    print(f"  Sphere radius: {radius}")
    print(f"  Memory: {sphere_np.nbytes / 1024**2:.2f} MB")
    
    # Transfer to GPU
    sphere_gpu = cp.asarray(sphere_np)
    del sphere_np
    gc.collect()
    
    # Test 2: Compute Hessian
    print("\n[Test 2] Computing Hessian components")
    print("-" * 40)
    
    engine = HessianEngine(slab_size=16, verbose=True)
    hessian = engine.compute_hessian(sphere_gpu, sigma=1.5)
    
    components = hessian.as_list()
    print(f"  Generated {len(components)} Hessian components")
    
    # Test 3: Eigenvalue decomposition
    print("\n[Test 3] Solving eigen system")
    print("-" * 40)
    
    result = engine.solve_eigen_system(components)
    
    print(f"  Normal vectors shape: {result.normal_vectors.shape}")
    print(f"  Confidence shape: {result.confidence.shape}")
    
    # Test 4: Validate normal vectors point toward center
    print("\n[Test 4] Validating normal vector directions")
    print("-" * 40)
    
    # Sample points on the sphere shell
    z, y, x = cp.meshgrid(
        cp.arange(size),
        cp.arange(size),
        cp.arange(size),
        indexing='ij'
    )
    
    # Distance from center
    distance = cp.sqrt(
        (z - center[0])**2 + 
        (y - center[1])**2 + 
        (x - center[2])**2
    )
    
    # Select voxels on the sphere shell (within 2 voxels of radius)
    shell_mask = cp.abs(distance - radius) < 2.0
    n_shell_voxels = int(shell_mask.sum())
    
    if n_shell_voxels > 0:
        # Ground truth: unit vectors pointing toward center
        gt_normals = cp.zeros((3, size, size, size), dtype=cp.float32)
        
        # Avoid division by zero
        safe_distance = distance + 1e-10
        
        gt_normals[0] = (center[0] - z) / safe_distance  # Z component
        gt_normals[1] = (center[1] - y) / safe_distance  # Y component
        gt_normals[2] = (center[2] - x) / safe_distance  # X component
        
        del z, y, x, safe_distance
        
        # Compute dot product with predicted normals (considering sign ambiguity)
        dot_product = (
            result.normal_vectors[0] * gt_normals[0] +
            result.normal_vectors[1] * gt_normals[1] +
            result.normal_vectors[2] * gt_normals[2]
        )
        
        # Take absolute value (eigenvectors can be +/- flipped)
        abs_dot = cp.abs(dot_product)
        
        # Angular error: arccos(|dot|) in degrees
        # Clamp to valid range for arccos
        abs_dot_clamped = cp.clip(abs_dot, 0.0, 1.0)
        angular_error = cp.arccos(abs_dot_clamped) * 180.0 / cp.pi
        
        # Get mean angular error on shell voxels
        shell_errors = angular_error[shell_mask]
        mean_error = float(shell_errors.mean())
        median_error = float(cp.median(shell_errors))
        max_error = float(shell_errors.max())
        
        print(f"  Shell voxels analyzed: {n_shell_voxels}")
        print(f"  Mean angular error: {mean_error:.2f}°")
        print(f"  Median angular error: {median_error:.2f}°")
        print(f"  Max angular error: {max_error:.2f}°")
        
        # Cleanup GPU memory
        del gt_normals, dot_product, abs_dot, abs_dot_clamped, angular_error
        del shell_mask, shell_errors, distance
        
        # Test passes if median error < 15 degrees
        test_passed = median_error < 15.0
    else:
        print("  WARNING: No shell voxels found for validation")
        test_passed = False
    
    # Cleanup
    hessian.free()
    del result.normal_vectors, result.confidence
    del sphere_gpu
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Summary
    print("\n" + "=" * 60)
    if test_passed:
        print("✓ All tests PASSED")
        print(f"  Normal vector accuracy: {median_error:.2f}° median error")
    else:
        print("✗ Tests FAILED")
        print(f"  Normal vector accuracy insufficient or validation failed")
    print("=" * 60)
    
    return test_passed


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    test_hessian_synthetic(size=128)

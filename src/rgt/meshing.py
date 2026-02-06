#!/usr/bin/env python3
"""
RGT Meshing Module for Vesuvius Challenge 2026
===============================================

Isosurface extraction and quality auditing via Gaussian curvature.

Author: HPC Architect
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import cupy as cp


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MeshData:
    """Container for mesh geometry."""
    vertices: np.ndarray       # (N, 3) float32
    faces: np.ndarray          # (M, 3) int32
    normals: Optional[np.ndarray] = None  # (N, 3) float32
    
    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]
    
    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]


@dataclass 
class CurvatureAudit:
    """Result of curvature quality audit."""
    mean_K: float              # Mean Gaussian curvature
    std_K: float               # Std dev of Gaussian curvature
    defect_ratio: float        # Fraction with |K| > threshold
    theoretical_K: Optional[float] = None  # Expected K (for spheres)
    vertex_curvatures: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# SurfaceExtractor Class
# =============================================================================

class SurfaceExtractor:
    """
    Extract isosurfaces from RGT scalar field and audit quality.
    
    Uses marching cubes for surface extraction and discrete Gaussian
    curvature for quality assessment.
    
    Usage:
        extractor = SurfaceExtractor()
        mesh = extractor.extract_layer(phi, iso_level=0.5)
        audit = extractor.audit_curvature(mesh)
        extractor.save_mesh(mesh, "output.obj")
    """
    
    def __init__(
        self,
        step_size: int = 1,
        gradient_direction: str = "descent",
        curvature_threshold: float = 0.01,
        verbose: bool = True,
    ):
        """
        Initialize SurfaceExtractor.
        
        Args:
            step_size: Marching cubes step size (1 = full resolution).
            gradient_direction: 'descent' or 'ascent' for surface normals.
            curvature_threshold: Threshold for defect detection.
            verbose: Print progress messages.
        """
        self.step_size = step_size
        self.gradient_direction = gradient_direction
        self.curvature_threshold = curvature_threshold
        self.verbose = verbose
    
    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[SurfaceExtractor] {msg}")
    
    def extract_layer(
        self,
        phi: "cp.ndarray",
        iso_level: Optional[float] = None,
    ) -> MeshData:
        """
        Extract isosurface from scalar field.
        
        Args:
            phi: Scalar field (D, H, W) on GPU.
            iso_level: Iso-value to extract. If None, uses mean.
            
        Returns:
            MeshData with vertices and faces.
        """
        import cupy as cp
        from skimage.measure import marching_cubes
        
        self._log(f"Extracting isosurface (shape={phi.shape})")
        
        # Transfer to CPU
        phi_cpu = cp.asnumpy(phi).astype(np.float32)
        
        # Determine iso-level
        if iso_level is None:
            iso_level = float(phi_cpu.mean())
        
        self._log(f"  Iso-level: {iso_level:.4f}")
        
        # Extract isosurface
        try:
            verts, faces, normals, _ = marching_cubes(
                phi_cpu,
                level=iso_level,
                step_size=self.step_size,
                gradient_direction=self.gradient_direction,
            )
        except ValueError as e:
            self._log(f"  Warning: Marching cubes failed: {e}")
            # Return empty mesh
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=np.zeros((0, 3), dtype=np.float32),
            )
        
        self._log(f"  ✓ Extracted: {len(verts):,} vertices, {len(faces):,} faces")
        
        return MeshData(
            vertices=verts.astype(np.float32),
            faces=faces.astype(np.int32),
            normals=normals.astype(np.float32),
        )
    
    def compute_vertex_curvatures(
        self,
        mesh: MeshData,
    ) -> np.ndarray:
        """
        Compute discrete Gaussian curvature at each vertex.
        
        Uses the angle deficit formula:
            K_i = (2π - Σ θ_ij) / A_i
        
        where θ_ij are the angles at vertex i in incident triangles,
        and A_i is the barycentric area around vertex i.
        
        Args:
            mesh: Mesh data.
            
        Returns:
            Gaussian curvature at each vertex.
        """
        verts = mesh.vertices
        faces = mesh.faces
        n_verts = mesh.n_vertices
        
        # Initialize angle sum and area per vertex
        angle_sum = np.zeros(n_verts, dtype=np.float64)
        area = np.zeros(n_verts, dtype=np.float64)
        
        # Process each face
        for f in faces:
            v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
            
            # Edge vectors
            e01 = v1 - v0
            e02 = v2 - v0
            e12 = v2 - v1
            e10 = -e01
            e20 = -e02
            e21 = -e12
            
            # Lengths
            l01 = np.linalg.norm(e01) + 1e-10
            l02 = np.linalg.norm(e02) + 1e-10
            l12 = np.linalg.norm(e12) + 1e-10
            
            # Angles at each vertex of this triangle
            cos_a0 = np.clip(np.dot(e01, e02) / (l01 * l02), -1.0, 1.0)
            cos_a1 = np.clip(np.dot(e10, e12) / (l01 * l12), -1.0, 1.0)
            cos_a2 = np.clip(np.dot(e20, e21) / (l02 * l12), -1.0, 1.0)
            
            a0 = np.arccos(cos_a0)
            a1 = np.arccos(cos_a1)
            a2 = np.arccos(cos_a2)
            
            # Accumulate angles
            angle_sum[f[0]] += a0
            angle_sum[f[1]] += a1
            angle_sum[f[2]] += a2
            
            # Triangle area (for barycentric distribution)
            tri_area = 0.5 * np.linalg.norm(np.cross(e01, e02))
            area[f[0]] += tri_area / 3.0
            area[f[1]] += tri_area / 3.0
            area[f[2]] += tri_area / 3.0
        
        # Gaussian curvature via angle deficit
        angle_deficit = 2.0 * np.pi - angle_sum
        
        # Avoid division by zero
        area = np.maximum(area, 1e-10)
        
        K = angle_deficit / area
        
        return K.astype(np.float32)
    
    def audit_curvature(
        self,
        mesh: MeshData,
        theoretical_K: Optional[float] = None,
    ) -> CurvatureAudit:
        """
        Audit mesh quality via Gaussian curvature statistics.
        
        For developable surfaces (paper sheets), K should be ~0.
        For spheres of radius R, K should be ~1/R².
        
        Args:
            mesh: Mesh data.
            theoretical_K: Expected curvature (for validation).
            
        Returns:
            CurvatureAudit with statistics.
        """
        self._log("Computing Gaussian curvature...")
        
        if mesh.n_vertices == 0:
            self._log("  Warning: Empty mesh")
            return CurvatureAudit(
                mean_K=0.0, std_K=0.0, defect_ratio=1.0,
                theoretical_K=theoretical_K,
            )
        
        K = self.compute_vertex_curvatures(mesh)
        
        mean_K = float(np.mean(K))
        std_K = float(np.std(K))
        
        # Defect ratio: vertices with |K| above threshold
        if theoretical_K is not None:
            # Compare to expected value
            defects = np.abs(K - theoretical_K) > self.curvature_threshold
        else:
            # For developable surfaces, K should be ~0
            defects = np.abs(K) > self.curvature_threshold
        
        defect_ratio = float(np.mean(defects))
        
        self._log(f"  Mean K: {mean_K:.6f}")
        self._log(f"  Std K: {std_K:.6f}")
        self._log(f"  Defect ratio: {defect_ratio:.2%}")
        
        if theoretical_K is not None:
            error = abs(mean_K - theoretical_K)
            self._log(f"  Theoretical K: {theoretical_K:.6f} (error: {error:.6f})")
        
        return CurvatureAudit(
            mean_K=mean_K,
            std_K=std_K,
            defect_ratio=defect_ratio,
            theoretical_K=theoretical_K,
            vertex_curvatures=K,
        )
    
    def save_mesh(
        self,
        mesh: MeshData,
        filepath: str,
        include_normals: bool = True,
    ) -> None:
        """
        Save mesh to Wavefront OBJ format.
        
        Args:
            mesh: Mesh data.
            filepath: Output path (should end in .obj).
            include_normals: Include vertex normals in output.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self._log(f"Saving mesh to {filepath}")
        
        with open(filepath, 'w') as f:
            f.write(f"# RGT Mesh Export\n")
            f.write(f"# Vertices: {mesh.n_vertices}\n")
            f.write(f"# Faces: {mesh.n_faces}\n\n")
            
            # Vertices
            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Normals (optional)
            if include_normals and mesh.normals is not None:
                f.write("\n")
                for n in mesh.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
            # Faces (1-indexed in OBJ format)
            f.write("\n")
            for face in mesh.faces:
                if include_normals and mesh.normals is not None:
                    f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                else:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        self._log(f"  ✓ Saved {filepath}")


# =============================================================================
# Test Function
# =============================================================================

def test_surface_extractor(size: int = 64) -> bool:
    """
    Test SurfaceExtractor on synthetic sphere.
    
    For a sphere of radius R, Gaussian curvature K = 1/R² everywhere.
    
    Args:
        size: Volume dimension.
        
    Returns:
        True if curvature matches theoretical value.
    """
    import cupy as cp
    from rgt.feature_extraction import create_synthetic_sphere
    
    print("=" * 60)
    print("SurfaceExtractor Test Suite")
    print("=" * 60)
    
    center = (size / 2, size / 2, size / 2)
    radius = size * 0.3
    theoretical_K = 1.0 / (radius ** 2)
    
    print(f"\n[Setup]")
    print(f"  Radius: {radius:.2f}")
    print(f"  Theoretical K: {theoretical_K:.6f}")
    
    # Create sphere as scalar field (distance from center)
    z, y, x = np.meshgrid(
        np.arange(size), np.arange(size), np.arange(size), indexing='ij'
    )
    phi_np = np.sqrt(
        (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    ).astype(np.float32)
    phi = cp.asarray(phi_np)
    
    print(f"\n[Step 1] Extracting isosurface")
    print("-" * 40)
    
    extractor = SurfaceExtractor(curvature_threshold=0.05)
    mesh = extractor.extract_layer(phi, iso_level=radius)
    
    print(f"\n[Step 2] Auditing curvature")
    print("-" * 40)
    
    audit = extractor.audit_curvature(mesh, theoretical_K=theoretical_K)
    
    print(f"\n[Step 3] Saving mesh")
    print("-" * 40)
    
    extractor.save_mesh(mesh, "output/test_sphere.obj")
    
    # Cleanup
    del phi, phi_np
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    # Test: mean curvature should be close to theoretical
    curvature_error = abs(audit.mean_K - theoretical_K) / theoretical_K
    test_passed = curvature_error < 0.5  # Within 50% (discrete approximation)
    
    print("\n" + "=" * 60)
    if test_passed:
        print("✓ All tests PASSED")
        print(f"  Mean K: {audit.mean_K:.6f} (expected: {theoretical_K:.6f})")
        print(f"  Relative error: {curvature_error:.1%}")
    else:
        print("✗ Tests FAILED")
        print(f"  Mean K: {audit.mean_K:.6f} (expected: {theoretical_K:.6f})")
        print(f"  Relative error: {curvature_error:.1%}")
    print("=" * 60)
    
    return test_passed


if __name__ == "__main__":
    test_surface_extractor(size=64)

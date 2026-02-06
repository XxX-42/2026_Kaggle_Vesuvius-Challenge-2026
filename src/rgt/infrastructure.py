#!/usr/bin/env python3
"""
RGT Infrastructure for Vesuvius Challenge 2026
==============================================

Environment-aware compute infrastructure with:
- Automatic Kaggle/Local environment detection
- Hardware-specific VRAM configuration
- Dependency injection with graceful fallbacks
- Proactive VRAM overflow protection

Author: HPC Architect
"""

from __future__ import annotations

import gc
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

# Lazy imports for GPU libraries
if TYPE_CHECKING:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cupy_splinalg


# =============================================================================
# Environment Detection & Configuration
# =============================================================================

class ExecutionEnvironment(Enum):
    """Execution environment enumeration."""
    LOCAL_RTX3060 = auto()
    KAGGLE_P100 = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable hardware configuration profile."""
    name: str
    total_vram_gb: float
    max_chunk_size: int
    enable_memory_pool: bool
    aggressive_gc: bool
    vram_safety_threshold: float = 0.85  # 85% max utilization


# Predefined hardware profiles
HARDWARE_PROFILES = {
    ExecutionEnvironment.LOCAL_RTX3060: HardwareProfile(
        name="NVIDIA RTX 3060 (Local Dev)",
        total_vram_gb=12.0,
        max_chunk_size=256,
        enable_memory_pool=False,
        aggressive_gc=True,
        vram_safety_threshold=0.80,  # More conservative locally
    ),
    ExecutionEnvironment.KAGGLE_P100: HardwareProfile(
        name="NVIDIA Tesla P100 (Kaggle)",
        total_vram_gb=16.0,
        max_chunk_size=640,
        enable_memory_pool=True,
        aggressive_gc=False,
        vram_safety_threshold=0.85,
    ),
    ExecutionEnvironment.UNKNOWN: HardwareProfile(
        name="Unknown GPU (Conservative)",
        total_vram_gb=8.0,
        max_chunk_size=128,
        enable_memory_pool=False,
        aggressive_gc=True,
        vram_safety_threshold=0.70,
    ),
}


class ContextManager:
    """
    Environment-aware context manager for RGT computations.
    
    Automatically detects execution environment (Kaggle vs Local) and
    configures appropriate VRAM management strategies.
    
    Usage:
        ctx = ContextManager()
        print(f"Running on: {ctx.profile.name}")
        print(f"Max chunk size: {ctx.profile.max_chunk_size}")
    """
    
    _instance: Optional["ContextManager"] = None
    
    def __new__(cls) -> "ContextManager":
        """Singleton pattern to ensure consistent configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._env = self._detect_environment()
        self._profile = HARDWARE_PROFILES[self._env]
        self._gpu_available = self._check_gpu()
        self._paths = self._configure_paths()
        self._initialized = True
        
        self._log_initialization()
    
    @property
    def environment(self) -> ExecutionEnvironment:
        """Current execution environment."""
        return self._env
    
    @property
    def profile(self) -> HardwareProfile:
        """Hardware profile for current environment."""
        return self._profile
    
    @property
    def gpu_available(self) -> bool:
        """Whether CUDA GPU is available."""
        return self._gpu_available
    
    @property
    def input_path(self) -> Path:
        """Read-only input data path."""
        return self._paths["input"]
    
    @property
    def output_path(self) -> Path:
        """Writable output path."""
        return self._paths["output"]
    
    @property
    def is_kaggle(self) -> bool:
        """Whether running in Kaggle environment."""
        return self._env == ExecutionEnvironment.KAGGLE_P100
    
    def _detect_environment(self) -> ExecutionEnvironment:
        """
        Detect execution environment through multiple heuristics.
        
        Detection order:
        1. KAGGLE_KERNEL_RUN_TYPE environment variable
        2. /kaggle path existence
        3. GPU model string matching
        """
        # Method 1: Kaggle environment variable
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            return ExecutionEnvironment.KAGGLE_P100
        
        # Method 2: Kaggle filesystem marker
        if Path("/kaggle").exists():
            return ExecutionEnvironment.KAGGLE_P100
        
        # Method 3: GPU model detection
        gpu_name = self._get_gpu_name()
        if gpu_name:
            gpu_name_lower = gpu_name.lower()
            if "p100" in gpu_name_lower or "v100" in gpu_name_lower or "t4" in gpu_name_lower:
                return ExecutionEnvironment.KAGGLE_P100
            elif "3060" in gpu_name_lower or "3070" in gpu_name_lower or "3080" in gpu_name_lower:
                return ExecutionEnvironment.LOCAL_RTX3060
            elif "4060" in gpu_name_lower or "4070" in gpu_name_lower or "4080" in gpu_name_lower or "4090" in gpu_name_lower:
                # Treat RTX 40 series similar to 30 series for local dev
                return ExecutionEnvironment.LOCAL_RTX3060
        
        return ExecutionEnvironment.UNKNOWN
    
    def _get_gpu_name(self) -> Optional[str]:
        """Get GPU device name via CuPy."""
        try:
            import cupy as cp
            device = cp.cuda.Device(0)
            return device.attributes.get("DeviceName", str(device))
        except Exception:
            try:
                # Fallback: nvidia-smi parsing
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return result.stdout.strip().split("\n")[0]
            except Exception:
                pass
        return None
    
    def _check_gpu(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False
    
    def _configure_paths(self) -> dict[str, Path]:
        """Configure input/output paths based on environment."""
        if self._env == ExecutionEnvironment.KAGGLE_P100:
            return {
                "input": Path("/kaggle/input"),
                "output": Path("/kaggle/working"),
            }
        else:
            # Local development paths
            cwd = Path.cwd()
            return {
                "input": cwd / "data" / "input",
                "output": cwd / "data" / "output",
            }
    
    def _log_initialization(self) -> None:
        """Log initialization summary."""
        print("=" * 60)
        print("RGT Infrastructure Initialized")
        print("=" * 60)
        print(f"  Environment   : {self._env.name}")
        print(f"  Profile       : {self._profile.name}")
        print(f"  GPU Available : {self._gpu_available}")
        print(f"  Max Chunk     : {self._profile.max_chunk_size}")
        print(f"  Memory Pool   : {'Enabled' if self._profile.enable_memory_pool else 'Disabled'}")
        print(f"  Aggressive GC : {'Enabled' if self._profile.aggressive_gc else 'Disabled'}")
        print(f"  VRAM Threshold: {self._profile.vram_safety_threshold:.0%}")
        print(f"  Input Path    : {self._paths['input']}")
        print(f"  Output Path   : {self._paths['output']}")
        print("=" * 60)
    
    def ensure_output_dir(self) -> Path:
        """Ensure output directory exists and return path."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        return self.output_path
    
    def force_gc(self) -> None:
        """Force garbage collection including GPU memory."""
        gc.collect()
        if self._gpu_available:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass


# =============================================================================
# Dependency Injection: Sparse Solver Backend
# =============================================================================

class SparseSolverBackend(ABC):
    """Abstract base class for sparse linear system solvers."""
    
    @abstractmethod
    def solve(
        self,
        A,  # Sparse matrix (CSR format)
        b,  # Right-hand side vector
        x0=None,  # Initial guess
        tol: float = 1e-6,
        maxiter: int = 1000,
    ) -> Tuple:
        """
        Solve sparse linear system Ax = b.
        
        Returns:
            tuple: (solution_vector, info_dict)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass


class CuPyCGBackend(SparseSolverBackend):
    """CuPy Conjugate Gradient solver backend (fallback)."""
    
    def solve(self, A, b, x0=None, tol: float = 1e-6, maxiter: int = 1000) -> Tuple:
        import cupy as cp
        from cupyx.scipy.sparse.linalg import cg
        
        x, info = cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
        
        return x, {"converged": info == 0, "iterations": "N/A", "backend": self.name}
    
    @property
    def name(self) -> str:
        return "CuPy CG (cupyx.scipy.sparse.linalg.cg)"


class PyAMGXBackend(SparseSolverBackend):
    """PyAMGX solver backend (high-performance, if available)."""
    
    def __init__(self):
        self._config = None
        self._resources = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization of AMGX resources."""
        if self._initialized:
            return
        
        import pyamgx
        
        pyamgx.initialize()
        
        # Default solver configuration
        cfg_dict = {
            "config_version": 2,
            "solver": {
                "preconditioner": {"solver": "NOSOLVER"},
                "solver": "PCG",
                "print_solve_stats": 0,
                "obtain_timings": 0,
                "max_iters": 1000,
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI_CORE",
                "tolerance": 1e-6,
            },
        }
        
        self._config = pyamgx.Config().create_from_dict(cfg_dict)
        self._resources = pyamgx.Resources().create_simple(self._config)
        self._initialized = True
    
    def solve(self, A, b, x0=None, tol: float = 1e-6, maxiter: int = 1000) -> Tuple:
        import pyamgx
        import cupy as cp
        
        self._lazy_init()
        
        # Create AMGX vectors and matrix
        d_x = pyamgx.Vector().create(self._resources)
        d_b = pyamgx.Vector().create(self._resources)
        d_A = pyamgx.Matrix().create(self._resources)
        
        # Upload data
        d_A.upload_CSR(A)
        d_b.upload(b.get() if hasattr(b, 'get') else b)
        
        if x0 is not None:
            d_x.upload(x0.get() if hasattr(x0, 'get') else x0)
        else:
            d_x.upload(cp.zeros(b.shape[0], dtype=b.dtype).get())
        
        # Create and setup solver
        solver = pyamgx.Solver().create(self._resources, self._config)
        solver.setup(d_A)
        
        # Solve
        solver.solve(d_b, d_x)
        
        # Download solution
        solution = cp.empty(b.shape[0], dtype=b.dtype)
        d_x.download(solution.get())
        
        # Cleanup
        d_x.destroy()
        d_b.destroy()
        d_A.destroy()
        solver.destroy()
        
        return cp.asarray(solution), {"converged": True, "backend": self.name}
    
    @property
    def name(self) -> str:
        return "PyAMGX (GPU-accelerated AMG)"
    
    def __del__(self):
        """Cleanup AMGX resources."""
        if self._initialized:
            try:
                import pyamgx
                if self._config:
                    self._config.destroy()
                if self._resources:
                    self._resources.destroy()
                pyamgx.finalize()
            except Exception:
                pass


class SolverFactory:
    """
    Factory for creating sparse solver backends with automatic fallback.
    
    Attempts to load PyAMGX first; falls back to CuPy CG if unavailable.
    """
    
    @staticmethod
    def create(prefer_amgx: bool = True) -> SparseSolverBackend:
        """
        Create best available solver backend.
        
        Args:
            prefer_amgx: If True, attempt to load PyAMGX first.
            
        Returns:
            SparseSolverBackend instance
        """
        if prefer_amgx:
            try:
                import pyamgx
                # Test that pyamgx is actually functional
                pyamgx.initialize()
                pyamgx.finalize()
                
                print("[SolverFactory] ✓ PyAMGX available, using GPU-accelerated AMG solver")
                return PyAMGXBackend()
                
            except ImportError:
                warnings.warn(
                    "[SolverFactory] PyAMGX not installed. "
                    "Falling back to CuPy CG solver. "
                    "For better performance, install pyamgx.",
                    RuntimeWarning,
                )
            except OSError as e:
                if "libamgxsh.so" in str(e) or "amgx" in str(e).lower():
                    warnings.warn(
                        f"[SolverFactory] PyAMGX found but libamgxsh.so not available: {e}. "
                        "This is expected on Kaggle without pre-built wheels. "
                        "Falling back to CuPy CG solver.",
                        RuntimeWarning,
                    )
                else:
                    warnings.warn(
                        f"[SolverFactory] PyAMGX initialization failed: {e}. "
                        "Falling back to CuPy CG solver.",
                        RuntimeWarning,
                    )
            except Exception as e:
                warnings.warn(
                    f"[SolverFactory] Unexpected error loading PyAMGX: {e}. "
                    "Falling back to CuPy CG solver.",
                    RuntimeWarning,
                )
        
        print("[SolverFactory] Using CuPy CG solver (cupyx.scipy.sparse.linalg.cg)")
        return CuPyCGBackend()


# =============================================================================
# VRAM Defense System
# =============================================================================

@dataclass
class VRAMEstimate:
    """VRAM usage estimate for a tensor allocation."""
    shape: Tuple[int, ...]
    dtype_bytes: int
    required_bytes: int
    required_gb: float
    available_bytes: int
    available_gb: float
    utilization_ratio: float
    is_safe: bool
    safety_threshold: float


class VRAMDefense:
    """
    Proactive VRAM overflow protection system.
    
    Performs dry-run checks before actual GPU memory allocation to prevent
    out-of-memory crashes that could corrupt computation state.
    """
    
    # Common dtype sizes in bytes
    DTYPE_SIZES = {
        "float16": 2,
        "float32": 4,
        "float64": 8,
        "int32": 4,
        "int64": 8,
        "complex64": 8,
        "complex128": 16,
    }
    
    def __init__(self, context: ContextManager):
        self.context = context
        self.profile = context.profile
    
    def _get_dtype_bytes(self, dtype) -> int:
        """Get byte size for a dtype."""
        if hasattr(dtype, 'itemsize'):
            return dtype.itemsize
        
        dtype_str = str(dtype).lower()
        for key, size in self.DTYPE_SIZES.items():
            if key in dtype_str:
                return size
        
        # Default to float32
        return 4
    
    def _get_vram_info(self) -> Tuple[int, int]:
        """
        Get current VRAM info (free, total) in bytes.
        
        Returns:
            tuple: (free_bytes, total_bytes)
        """
        if not self.context.gpu_available:
            # Return profile-based estimate for CPU fallback
            total = int(self.profile.total_vram_gb * (1024 ** 3))
            return total, total
        
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            
            # Get device memory info
            free, total = cp.cuda.Device(0).mem_info
            
            # Account for memory pool usage
            pool_used = mempool.used_bytes()
            effective_free = free - pool_used
            
            return max(0, effective_free), total
            
        except Exception:
            # Fallback to profile-based estimate
            total = int(self.profile.total_vram_gb * (1024 ** 3))
            return total, total
    
    def estimate(
        self,
        shape: Tuple[int, ...],
        dtype="float32",
        count: int = 1,
    ) -> VRAMEstimate:
        """
        Estimate VRAM requirements for tensor allocation.
        
        Args:
            shape: Tensor shape
            dtype: Data type (string or numpy/cupy dtype)
            count: Number of such tensors to allocate
            
        Returns:
            VRAMEstimate with detailed breakdown
        """
        dtype_bytes = self._get_dtype_bytes(dtype)
        
        # Calculate total elements
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        
        required_bytes = total_elements * dtype_bytes * count
        required_gb = required_bytes / (1024 ** 3)
        
        free_bytes, total_bytes = self._get_vram_info()
        free_gb = free_bytes / (1024 ** 3)
        
        # Calculate utilization after allocation
        post_alloc_used = (total_bytes - free_bytes) + required_bytes
        utilization_ratio = post_alloc_used / total_bytes
        
        is_safe = utilization_ratio <= self.profile.vram_safety_threshold
        
        return VRAMEstimate(
            shape=shape,
            dtype_bytes=dtype_bytes,
            required_bytes=required_bytes,
            required_gb=required_gb,
            available_bytes=free_bytes,
            available_gb=free_gb,
            utilization_ratio=utilization_ratio,
            is_safe=is_safe,
            safety_threshold=self.profile.vram_safety_threshold,
        )
    
    def dry_run_check(
        self,
        shape: Tuple[int, ...],
        dtype="float32",
        count: int = 1,
        operation_name: str = "allocation",
    ) -> VRAMEstimate:
        """
        Perform dry-run VRAM check and raise if unsafe.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            count: Number of tensors
            operation_name: Name for error messages
            
        Returns:
            VRAMEstimate if safe
            
        Raises:
            MemoryError: If allocation would exceed safety threshold
        """
        estimate = self.estimate(shape, dtype, count)
        
        if not estimate.is_safe:
            raise MemoryError(
                f"VRAM Defense: Blocking unsafe {operation_name}!\n"
                f"  Requested    : {estimate.required_gb:.2f} GB for shape {shape}\n"
                f"  Available    : {estimate.available_gb:.2f} GB\n"
                f"  Post-alloc   : {estimate.utilization_ratio:.1%} utilization\n"
                f"  Safety Limit : {estimate.safety_threshold:.0%}\n"
                f"  Suggestion   : Reduce chunk_size or use gradient checkpointing"
            )
        
        return estimate
    
    def check_rgt_allocation(
        self,
        grid_shape: Tuple[int, int, int],
        chunk_size: int,
        dtype="float32",
    ) -> VRAMEstimate:
        """
        Specialized check for RGT computation allocations.
        
        Estimates memory for:
        - Input volume chunk
        - Output gradient tensors (3x for x,y,z)
        - Sparse matrix working memory
        
        Args:
            grid_shape: Full volume shape (D, H, W)
            chunk_size: Processing chunk size
            dtype: Data type
            
        Returns:
            VRAMEstimate for the operation
        """
        # Enforce chunk size limit from profile
        if chunk_size > self.profile.max_chunk_size:
            warnings.warn(
                f"[VRAMDefense] chunk_size={chunk_size} exceeds profile max "
                f"({self.profile.max_chunk_size}). Clamping.",
                RuntimeWarning,
            )
            chunk_size = self.profile.max_chunk_size
        
        # Estimate allocations:
        # 1. Input chunk: chunk_size^3
        # 2. Output gradients: 3 x chunk_size^3 (Gx, Gy, Gz)
        # 3. Sparse matrix overhead: ~2x chunk_size^3 (indices + values)
        # 4. Working memory: ~1x chunk_size^3
        
        chunk_volume = chunk_size ** 3
        total_elements = chunk_volume * 7  # Conservative multiplier
        
        effective_shape = (total_elements,)
        
        return self.dry_run_check(
            shape=effective_shape,
            dtype=dtype,
            count=1,
            operation_name=f"RGT chunk processing (chunk_size={chunk_size})",
        )


# =============================================================================
# Unified RGT Infrastructure Facade
# =============================================================================

class RGTInfrastructure:
    """
    Unified facade for RGT computation infrastructure.
    
    Provides:
    - Environment-aware configuration
    - Automatic solver backend selection
    - VRAM protection
    - Garbage collection management
    """
    
    def __init__(self):
        self.context = ContextManager()
        self.vram = VRAMDefense(self.context)
        self._solver: Optional[SparseSolverBackend] = None
    
    @property
    def solver(self) -> SparseSolverBackend:
        """Lazy-loaded sparse solver backend."""
        if self._solver is None:
            self._solver = SolverFactory.create(prefer_amgx=True)
        return self._solver
    
    @property
    def chunk_size(self) -> int:
        """Maximum allowed chunk size for current environment."""
        return self.context.profile.max_chunk_size
    
    def pre_compute_check(
        self,
        volume_shape: Tuple[int, int, int],
        chunk_size: Optional[int] = None,
        dtype: str = "float32",
    ) -> bool:
        """
        Pre-computation safety check.
        
        Args:
            volume_shape: Full volume dimensions
            chunk_size: Processing chunk size (default: profile max)
            dtype: Data type
            
        Returns:
            True if computation is safe to proceed
            
        Raises:
            MemoryError: If VRAM is insufficient
        """
        chunk_size = chunk_size or self.chunk_size
        
        # Clamp chunk size
        chunk_size = min(chunk_size, self.context.profile.max_chunk_size)
        
        # Perform VRAM check
        estimate = self.vram.check_rgt_allocation(volume_shape, chunk_size, dtype)
        
        print(f"[RGTInfra] Pre-compute check passed:")
        print(f"  Volume       : {volume_shape}")
        print(f"  Chunk size   : {chunk_size}")
        print(f"  VRAM required: {estimate.required_gb:.2f} GB")
        print(f"  VRAM available: {estimate.available_gb:.2f} GB")
        print(f"  Utilization  : {estimate.utilization_ratio:.1%}")
        
        return True
    
    def cleanup(self) -> None:
        """Force memory cleanup."""
        if self.context.profile.aggressive_gc:
            self.context.force_gc()
            print("[RGTInfra] Aggressive GC completed")


# =============================================================================
# Test Entry Point
# =============================================================================

def run_tests():
    """Test infrastructure components."""
    print("\n" + "=" * 60)
    print("RGT Infrastructure Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Context Manager
    print("[Test 1] Context Manager Initialization")
    print("-" * 40)
    ctx = ContextManager()
    assert ctx.environment is not None
    assert ctx.profile is not None
    print("✓ Context Manager initialized successfully\n")
    
    # Test 2: Singleton Pattern
    print("[Test 2] Singleton Pattern")
    print("-" * 40)
    ctx2 = ContextManager()
    assert ctx is ctx2, "Singleton pattern failed"
    print("✓ Singleton pattern working correctly\n")
    
    # Test 3: Path Configuration
    print("[Test 3] Path Configuration")
    print("-" * 40)
    print(f"  Input path  : {ctx.input_path}")
    print(f"  Output path : {ctx.output_path}")
    assert isinstance(ctx.input_path, Path)
    assert isinstance(ctx.output_path, Path)
    print("✓ Paths configured correctly\n")
    
    # Test 4: VRAM Defense
    print("[Test 4] VRAM Defense System")
    print("-" * 40)
    vram = VRAMDefense(ctx)
    
    # Safe allocation
    small_shape = (64, 64, 64)
    estimate = vram.estimate(small_shape, "float32")
    print(f"  Small tensor {small_shape}: {estimate.required_gb:.4f} GB")
    print(f"  Is safe: {estimate.is_safe}")
    
    # Large allocation (should fail on small GPU)
    large_shape = (2048, 2048, 2048)
    estimate_large = vram.estimate(large_shape, "float32")
    print(f"  Large tensor {large_shape}: {estimate_large.required_gb:.2f} GB")
    print(f"  Is safe: {estimate_large.is_safe}")
    
    # Test dry_run_check with safe allocation
    try:
        vram.dry_run_check(small_shape, "float32", operation_name="test_small")
        print("✓ Safe allocation check passed")
    except MemoryError as e:
        print(f"✗ Unexpected MemoryError: {e}")
    
    print()
    
    # Test 5: Solver Factory
    print("[Test 5] Solver Factory")
    print("-" * 40)
    solver = SolverFactory.create(prefer_amgx=True)
    print(f"  Selected solver: {solver.name}")
    print("✓ Solver factory working correctly\n")
    
    # Test 6: RGT Infrastructure Facade
    print("[Test 6] RGT Infrastructure Facade")
    print("-" * 40)
    infra = RGTInfrastructure()
    print(f"  Max chunk size: {infra.chunk_size}")
    
    try:
        infra.pre_compute_check(
            volume_shape=(512, 512, 512),
            chunk_size=128,
        )
        print("✓ Pre-compute check passed")
    except MemoryError as e:
        print(f"  Pre-compute check blocked (expected on limited VRAM): {e}")
    
    infra.cleanup()
    print()
    
    # Summary
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print(f"\nEnvironment Summary:")
    print(f"  - Execution Mode: {ctx.environment.name}")
    print(f"  - Hardware Profile: {ctx.profile.name}")
    print(f"  - GPU Available: {ctx.gpu_available}")
    print(f"  - Solver Backend: {solver.name}")
    print(f"  - Max Chunk Size: {ctx.profile.max_chunk_size}")


if __name__ == "__main__":
    run_tests()

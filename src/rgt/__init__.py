# RGT Package for Vesuvius Challenge 2026
"""
rgt: Relative Geological Time computation infrastructure.

Modules:
    - infrastructure: Environment detection, VRAM management, solver backends
    - feature_extraction: Hessian-based micro-feature extraction
"""

from rgt.infrastructure import (
    ContextManager,
    VRAMDefense,
    RGTInfrastructure,
    SolverFactory,
)
from rgt.feature_extraction import (
    HessianEngine,
    HessianResult,
    EigenResult,
)

__all__ = [
    "ContextManager",
    "VRAMDefense", 
    "RGTInfrastructure",
    "SolverFactory",
    "HessianEngine",
    "HessianResult",
    "EigenResult",
]

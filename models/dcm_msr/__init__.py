"""
DCM-MSR: Dynamic Coarse Masking - Multi-Scale Routing

A hybrid quantum-classical attention scheme that:
1. Separates tokens into windows
2. Combines keys into an ensemble density matrix
3. Performs fidelity (SWAP) test for coarse attention scoring
4. Routes top-k windows for fine-grained token-to-token attention
"""

from .quantum_utils import (
    create_pure_state_density_matrix,
    create_ensemble_density_matrix,
    swap_test_fidelity,
)
from .attention import DCMMSRAttention, DCMMSRSelfAttention
from .windowing import create_windows, merge_windows

__version__ = "0.1.0"

__all__ = [
    "DCMMSRAttention",
    "DCMMSRSelfAttention",
    "create_pure_state_density_matrix",
    "create_ensemble_density_matrix",
    "swap_test_fidelity",
    "create_windows",
    "merge_windows",
]

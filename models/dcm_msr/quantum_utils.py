"""
Quantum utilities for DCM-MSR attention.

Implements density matrix operations and SWAP test for fidelity computation.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def create_pure_state_density_matrix(state: torch.Tensor) -> torch.Tensor:
    """
    Create a pure state density matrix from a state vector.
    
    For a pure state |ψ⟩, the density matrix is ρ = |ψ⟩⟨ψ|
    
    Args:
        state: State vector of shape (..., d) where d is the state dimension.
               The state is assumed to be normalized.
    
    Returns:
        Density matrix of shape (..., d, d)
    """
    # Normalize the state
    state = F.normalize(state, p=2, dim=-1)
    
    # ρ = |ψ⟩⟨ψ| = outer product
    # state: (..., d) -> (..., d, 1) @ (..., 1, d) -> (..., d, d)
    return torch.einsum('...i,...j->...ij', state, state.conj())


def create_ensemble_density_matrix(
    states: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Create an ensemble (mixed state) density matrix from multiple states.
    
    For an ensemble of pure states {|ψ_i⟩} with probabilities {p_i},
    the density matrix is ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|
    
    Args:
        states: State vectors of shape (..., n, d) where n is the number of
                states and d is the state dimension.
        weights: Optional weights of shape (..., n). If None, uniform weights
                 are used (1/n for each state).
    
    Returns:
        Ensemble density matrix of shape (..., d, d)
    """
    n = states.shape[-2]
    
    # Normalize each state
    states = F.normalize(states, p=2, dim=-1)
    
    if weights is None:
        # Uniform mixture: ρ = (1/n) Σ_i |ψ_i⟩⟨ψ_i|
        weights = torch.ones(states.shape[:-1], device=states.device, dtype=states.dtype) / n
    else:
        # Normalize weights to sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Create individual density matrices and weight them
    # states: (..., n, d)
    # density_matrices: (..., n, d, d)
    density_matrices = torch.einsum('...ni,...nj->...nij', states, states.conj())
    
    # Weight and sum: ρ = Σ_i w_i ρ_i
    # weights: (..., n)
    # result: (..., d, d)
    return torch.einsum('...n,...nij->...ij', weights, density_matrices)


def swap_test_fidelity(
    rho: torch.Tensor,
    sigma: torch.Tensor,
    classical_approximation: bool = True
) -> torch.Tensor:
    """
    Compute the fidelity between two density matrices using SWAP test.
    
    The SWAP test measures the overlap between two quantum states.
    For a pure state |ψ⟩ (with density matrix ρ = |ψ⟩⟨ψ|) and a mixed state σ,
    the fidelity is F(ρ, σ) = ⟨ψ|σ|ψ⟩ = Tr(ρσ).
    
    In general, the fidelity is F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²
    
    For computational efficiency, we use the classical approximation:
    F ≈ Tr(ρσ) which is exact when at least one state is pure.
    
    The SWAP test probability of measuring |0⟩ is P(0) = (1 + Tr(ρσ))/2
    So Tr(ρσ) = 2*P(0) - 1
    
    Args:
        rho: First density matrix of shape (..., d, d)
        sigma: Second density matrix of shape (..., d, d)
        classical_approximation: If True, use Tr(ρσ) approximation.
                                 If False, compute full quantum fidelity.
    
    Returns:
        Fidelity values of shape (...)
    """
    if classical_approximation:
        # Tr(ρσ) = sum of element-wise products
        # This is equivalent to the SWAP test outcome for pure states
        fidelity = torch.einsum('...ij,...ji->...', rho, sigma).real
        # Clamp to valid range [0, 1]
        return torch.clamp(fidelity, 0.0, 1.0)
    else:
        # Full quantum fidelity: F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²
        # This requires matrix square root, which is computationally expensive
        # We use eigendecomposition: √A = V √Λ V†
        # Note: Requires PyTorch 1.9+ for torch.linalg.eigh and torch.linalg.eigvalsh
        
        # Compute √ρ via eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=0)  # Ensure non-negative
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        sqrt_rho = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.conj().transpose(-1, -2)
        
        # Compute √ρ σ √ρ
        inner = sqrt_rho @ sigma @ sqrt_rho
        
        # Compute eigenvalues of inner product
        inner_eigenvalues = torch.linalg.eigvalsh(inner)
        inner_eigenvalues = torch.clamp(inner_eigenvalues, min=0)
        
        # Fidelity = [Tr(√inner)]² = [sum(√λ_i)]²
        fidelity = torch.sum(torch.sqrt(inner_eigenvalues), dim=-1) ** 2
        
        return torch.clamp(fidelity, 0.0, 1.0)


def compute_purity(rho: torch.Tensor) -> torch.Tensor:
    """
    Compute the purity of a density matrix.
    
    Purity is defined as Tr(ρ²) and measures how "pure" (vs "mixed") a state is.
    - Pure state: purity = 1
    - Maximally mixed state: purity = 1/d (where d is dimension)
    
    Args:
        rho: Density matrix of shape (..., d, d)
    
    Returns:
        Purity values of shape (...)
    """
    # Compute Tr(ρ²) = Tr(ρ @ ρ)
    purity = torch.einsum('...ij,...ji->...', rho, rho).real
    return torch.clamp(purity, 0.0, 1.0)


def compute_fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute the fidelity between two density matrices.
    
    Uses trace fidelity approximation: F(ρ, σ) = Tr(ρσ)
    which is exact when at least one state is pure.
    
    Args:
        rho: First density matrix of shape (..., d, d)
        sigma: Second density matrix of shape (..., d, d)
    
    Returns:
        Fidelity values of shape (...)
    """
    # Tr(ρσ) = sum of element-wise products
    fidelity = torch.einsum('...ij,...ji->...', rho, sigma).real
    return torch.clamp(fidelity, 0.0, 1.0)


def batch_swap_test_scores(
    query_dm: torch.Tensor,
    key_ensemble_dms: torch.Tensor,
) -> torch.Tensor:
    """
    Compute SWAP test fidelity scores between a query and multiple key ensembles.
    
    This is the core operation for coarse attention: comparing a query's
    pure state against ensemble density matrices from different windows.
    
    Args:
        query_dm: Query density matrix of shape (batch, heads, d, d)
        key_ensemble_dms: Key ensemble density matrices of shape 
                          (batch, heads, num_windows, d, d)
    
    Returns:
        Fidelity scores of shape (batch, heads, num_windows)
    """
    # query_dm: (B, H, d, d) -> (B, H, 1, d, d) for broadcasting
    query_dm = query_dm.unsqueeze(-3)
    
    # Compute Tr(query_dm @ key_ensemble_dms) for each window
    # Using einsum for batched trace of matrix products
    scores = torch.einsum('...ij,...ji->...', query_dm, key_ensemble_dms).real
    
    return torch.clamp(scores, 0.0, 1.0)

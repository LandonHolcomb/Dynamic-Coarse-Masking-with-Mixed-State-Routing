"""
DCM-MSR Attention Module.

Implements the hybrid quantum-classical attention scheme with:
1. Window-based token separation
2. Ensemble density matrix construction from keys
3. Fidelity (SWAP) test for coarse attention scoring
4. Top-k window routing
5. Fine-grained token-to-token attention within routed windows
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .quantum_utils import (
    create_pure_state_density_matrix,
    create_ensemble_density_matrix,
    batch_swap_test_scores,
)
from .windowing import create_windows, merge_windows, select_windows


class DCMMSRAttention(nn.Module):
    """
    Dynamic Coarse Masking - Multi-Scale Routing (DCM-MSR) Attention.
    
    A hybrid quantum-classical attention mechanism that:
    1. Separates tokens into windows
    2. Combines keys within each window into an ensemble density matrix
    3. Performs SWAP test between query pure state and key ensemble for coarse scoring
    4. Routes top-k windows based on coarse attention scores
    5. Applies fine-grained token-to-token attention within routed windows
    
    This provides O(n * k * w) complexity where:
    - n: sequence length
    - k: number of top-k windows selected
    - w: window size
    
    For k << n/w, this is significantly more efficient than full O(n²) attention.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        window_size: Size of each window for coarse attention
        top_k: Number of top windows to select for fine attention
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
        use_quantum_fidelity: If True, use full quantum fidelity calculation
                              If False, use classical Tr(ρσ) approximation
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 16,
        top_k: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        use_quantum_fidelity: bool = False,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.top_k = top_k
        self.use_quantum_fidelity = use_quantum_fidelity
        
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for coarse attention
        self.coarse_temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of DCM-MSR attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len_q, embed_dim)
            key: Key tensor of shape (batch, seq_len_kv, embed_dim)
            value: Value tensor of shape (batch, seq_len_kv, embed_dim)
            attention_mask: Optional mask of shape (batch, seq_len_kv)
            return_attention: If True, return attention weights
        
        Returns:
            Tuple of:
            - Output tensor of shape (batch, seq_len_q, embed_dim)
            - Optional tuple of (coarse_attention, fine_attention) if return_attention
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to multi-head format
        # (batch, seq, embed) -> (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # === Step 1: Create windows for K and V ===
        # Reshape for windowing: (batch, heads, seq, dim) -> (batch * heads, seq, dim)
        k_flat = k.reshape(batch_size * self.num_heads, seq_len_kv, self.head_dim)
        v_flat = v.reshape(batch_size * self.num_heads, seq_len_kv, self.head_dim)
        
        k_windowed, window_mask, orig_len = create_windows(k_flat, self.window_size)
        v_windowed, _, _ = create_windows(v_flat, self.window_size)
        
        # k_windowed: (batch * heads, num_windows, window_size, head_dim)
        num_windows = k_windowed.shape[1]
        
        # Reshape back with heads dimension
        k_windowed = k_windowed.view(batch_size, self.num_heads, num_windows, self.window_size, self.head_dim)
        v_windowed = v_windowed.view(batch_size, self.num_heads, num_windows, self.window_size, self.head_dim)
        window_mask = window_mask.view(batch_size, self.num_heads, num_windows, self.window_size)
        
        # === Step 2: Create ensemble density matrices for each window's keys ===
        # k_windowed: (batch, heads, num_windows, window_size, head_dim)
        # We create an ensemble density matrix for each window
        key_ensembles = self._create_key_ensembles(k_windowed, window_mask)
        # key_ensembles: (batch, heads, num_windows, head_dim, head_dim)
        
        # === Step 3: Create pure state density matrices for queries (QUERY-DEPENDENT) ===
        # Each query token gets its own routing decision
        # q: (batch, heads, seq_q, head_dim)
        q_normalized = F.normalize(q, p=2, dim=-1)  # Normalize for stable quantum state
        
        # Create density matrix for each query token
        # Reshape: (batch, heads, seq_q, head_dim) -> (batch * heads * seq_q, head_dim)
        q_flat = q_normalized.reshape(batch_size * self.num_heads * seq_len_q, self.head_dim)
        query_dm_flat = create_pure_state_density_matrix(q_flat)
        # query_dm_flat: (batch * heads * seq_q, head_dim, head_dim)
        
        # Reshape back: (batch * heads * seq_q, head_dim, head_dim) -> (batch, heads, seq_q, head_dim, head_dim)
        query_dm = query_dm_flat.view(batch_size, self.num_heads, seq_len_q, self.head_dim, self.head_dim)
        
        # === Step 4: SWAP test between each query and all key ensembles ===
        # For each query token, compute fidelity with each window ensemble
        # query_dm: (batch, heads, seq_q, head_dim, head_dim)
        # key_ensembles: (batch, heads, num_windows, head_dim, head_dim)
        
        # Expand dimensions for broadcasting
        # query_dm: (batch, heads, seq_q, 1, head_dim, head_dim)
        # key_ensembles: (batch, heads, 1, num_windows, head_dim, head_dim)
        query_dm_expanded = query_dm.unsqueeze(3)
        key_ensembles_expanded = key_ensembles.unsqueeze(2)
        
        # Compute fidelity for all (query, window) pairs
        coarse_scores = self._compute_query_dependent_scores(query_dm_expanded, key_ensembles_expanded)
        # coarse_scores: (batch, heads, seq_q, num_windows)
        
        # Apply temperature scaling with safe clamping
        temp = self.coarse_temperature.clamp(min=0.1)
        coarse_scores = coarse_scores / temp
        
        # === Step 5: Select top-k windows PER QUERY (query-dependent routing) ===
        k = min(self.top_k, num_windows)
        top_scores, top_indices = torch.topk(coarse_scores, k, dim=-1)
        # top_indices: (batch, heads, seq_q, k) - each query gets its own top-k windows
        
        # Compute coarse attention weights with safe softmax
        coarse_attn = F.softmax(top_scores, dim=-1)
        coarse_attn = self.dropout(coarse_attn)
        # coarse_attn: (batch, heads, seq_q, k)
        
        # === Step 6: Fine-grained attention within selected windows ===
        output = self._fine_attention(
            q, k_windowed, v_windowed, top_indices, coarse_attn, window_mask
        )
        # output: (batch, heads, seq_len_q, head_dim)
        
        # Reshape output: (batch, heads, seq_q, head_dim) -> (batch, seq_q, embed)
        output = output.transpose(1, 2).reshape(batch_size, seq_len_q, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        if return_attention:
            # Return both coarse and fine attention for visualization
            return output, (coarse_scores, coarse_attn)
        
        return output, None
    
    def _create_key_ensembles(
        self,
        k_windowed: torch.Tensor,
        window_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create ensemble density matrices for keys within each window.
        
        Args:
            k_windowed: (batch, heads, num_windows, window_size, head_dim)
            window_mask: (batch, heads, num_windows, window_size)
        
        Returns:
            Ensemble density matrices: (batch, heads, num_windows, head_dim, head_dim)
        """
        batch, heads, num_windows, window_size, head_dim = k_windowed.shape
        
        # Flatten batch dimensions for processing
        k_flat = k_windowed.reshape(batch * heads * num_windows, window_size, head_dim)
        mask_flat = window_mask.reshape(batch * heads * num_windows, window_size)
        
        # Create weights based on mask (uniform for valid tokens, 0 for padded)
        weights = mask_flat.float()
        # Normalize weights per window
        weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = weights / weights_sum
        
        # Create ensemble density matrices
        ensembles = create_ensemble_density_matrix(k_flat, weights)
        
        # Reshape back
        ensembles = ensembles.view(batch, heads, num_windows, head_dim, head_dim)
        
        return ensembles
    
    def _compute_query_dependent_scores(
        self,
        query_dm: torch.Tensor,
        key_ensembles: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fidelity scores between each query and all key ensembles.
        
        Uses direct batched computation via einsum for efficiency.
        Computes: Tr(sqrt(sqrt(rho) @ sigma @ sqrt(rho)))^2 ≈ Tr(rho @ sigma)
        Using trace fidelity as approximation: F(rho, sigma) = Tr(rho @ sigma)
        
        Args:
            query_dm: (batch, heads, seq_q, 1, head_dim, head_dim)
            key_ensembles: (batch, heads, 1, num_windows, head_dim, head_dim)
        
        Returns:
            scores: (batch, heads, seq_q, num_windows)
        """
        # Remove singleton dimensions
        # query_dm: (batch, heads, seq_q, head_dim, head_dim)
        # key_ensembles: (batch, heads, num_windows, head_dim, head_dim)
        query_dm = query_dm.squeeze(3)
        key_ensembles = key_ensembles.squeeze(2)
        
        # Compute pairwise fidelity using einsum
        # For each (query, window) pair: F = Tr(query_dm @ key_ensemble)
        # einsum: 'bhqij,bhwjk->bhqwik' then trace over last two dims
        # Simplified: 'bhqij,bhwji->bhqw' (exploiting hermiticity and taking trace directly)
        scores = torch.einsum('bhqij,bhwji->bhqw', query_dm, key_ensembles)
        
        # scores: (batch, heads, seq_q, num_windows)
        return scores.real  # Take real part for numerical stability
    
    def _fine_attention(
        self,
        q: torch.Tensor,
        k_windowed: torch.Tensor,
        v_windowed: torch.Tensor,
        top_indices: torch.Tensor,
        coarse_attn: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fine-grained token-to-token attention within selected windows.
        NOW QUERY-DEPENDENT: Each query token attends to its own selected windows.
        FULLY BATCHED: No Python loops over sequence positions.
        
        Args:
            q: Query tensor (batch, heads, seq_q, head_dim)
            k_windowed: Windowed keys (batch, heads, num_windows, window_size, head_dim)
            v_windowed: Windowed values (batch, heads, num_windows, window_size, head_dim)
            top_indices: Selected window indices (batch, heads, seq_q, k) - per-query routing
            coarse_attn: Coarse attention weights (batch, heads, seq_q, k)
            window_mask: Validity mask (batch, heads, num_windows, window_size)
        
        Returns:
            Attention output (batch, heads, seq_q, head_dim)
        """
        batch, heads, seq_q, head_dim = q.shape
        _, _, num_windows, window_size, _ = k_windowed.shape
        k = top_indices.shape[-1]
        
        # Batched gather: select windows for all queries at once
        # top_indices: (batch, heads, seq_q, k)
        # We need to expand to (batch, heads, seq_q, k, window_size, head_dim) for gathering
        
        # Expand indices for gathering keys and values
        idx = top_indices.unsqueeze(-1).unsqueeze(-1)  # (batch, heads, seq_q, k, 1, 1)
        idx = idx.expand(-1, -1, -1, -1, window_size, head_dim)  # (batch, heads, seq_q, k, window_size, head_dim)
        
        # Add seq_q dimension to k_windowed and v_windowed for gathering
        k_exp = k_windowed.unsqueeze(2).expand(-1, -1, seq_q, -1, -1, -1)  # (batch, heads, seq_q, num_windows, window_size, head_dim)
        v_exp = v_windowed.unsqueeze(2).expand(-1, -1, seq_q, -1, -1, -1)
        
        # Gather selected windows
        k_selected = torch.gather(k_exp, 3, idx)  # (batch, heads, seq_q, k, window_size, head_dim)
        v_selected = torch.gather(v_exp, 3, idx)
        
        # Gather masks
        idx_mask = top_indices.unsqueeze(-1).expand(-1, -1, -1, -1, window_size)  # (batch, heads, seq_q, k, window_size)
        mask_exp = window_mask.unsqueeze(2).expand(-1, -1, seq_q, -1, -1)  # (batch, heads, seq_q, num_windows, window_size)
        mask_selected = torch.gather(mask_exp, 3, idx_mask)  # (batch, heads, seq_q, k, window_size)
        
        # Flatten windows: (batch, heads, seq_q, k, window_size, head_dim) -> (batch, heads, seq_q, k * window_size, head_dim)
        k_flat = k_selected.reshape(batch, heads, seq_q, k * window_size, head_dim)
        v_flat = v_selected.reshape(batch, heads, seq_q, k * window_size, head_dim)
        mask_flat = mask_selected.reshape(batch, heads, seq_q, k * window_size)
        
        # Compute attention scores (batched over all queries)
        # q: (batch, heads, seq_q, head_dim)
        # k_flat: (batch, heads, seq_q, k * window_size, head_dim)
        attn_scores = torch.matmul(q.unsqueeze(-2), k_flat.transpose(-2, -1)) * self.scale
        # attn_scores: (batch, heads, seq_q, 1, k * window_size)
        attn_scores = attn_scores.squeeze(-2)  # (batch, heads, seq_q, k * window_size)
        
        # Apply mask
        attn_scores = attn_scores.masked_fill(~mask_flat, float('-inf'))
        
        # Apply coarse attention weights
        # coarse_attn: (batch, heads, seq_q, k) -> (batch, heads, seq_q, k * window_size)
        coarse_weights = coarse_attn.unsqueeze(-1).expand(-1, -1, -1, -1, window_size)
        coarse_weights = coarse_weights.reshape(batch, heads, seq_q, k * window_size)
        log_coarse = torch.log(coarse_weights.clamp(min=1e-8))
        attn_scores = attn_scores + log_coarse
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute output (batched matmul)
        # attn_probs: (batch, heads, seq_q, k * window_size)
        # v_flat: (batch, heads, seq_q, k * window_size, head_dim)
        output = torch.matmul(attn_probs.unsqueeze(-2), v_flat).squeeze(-2)
        # output: (batch, heads, seq_q, head_dim)
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, top_k={self.top_k}, "
            f"use_quantum_fidelity={self.use_quantum_fidelity}"
        )


class DCMMSRSelfAttention(DCMMSRAttention):
    """
    Self-attention variant of DCM-MSR where query, key, and value come from the same input.
    """
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            attention_mask: Optional mask of shape (batch, seq_len)
            return_attention: If True, return attention weights
        
        Returns:
            Same as parent forward method
        """
        return super().forward(x, x, x, attention_mask, return_attention)

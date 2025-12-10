"""
Window operations for DCM-MSR attention.

Handles partitioning tokens into windows and merging results.
"""

import torch
from typing import Tuple


def create_windows(
    tokens: torch.Tensor,
    window_size: int,
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Partition tokens into non-overlapping windows.
    
    Args:
        tokens: Input tokens of shape (batch, seq_len, dim)
        window_size: Size of each window
        pad_value: Value to use for padding if seq_len is not divisible by window_size
    
    Returns:
        Tuple of:
        - Windowed tokens of shape (batch, num_windows, window_size, dim)
        - Padding mask of shape (batch, num_windows, window_size) where True indicates valid tokens
        - Original sequence length before padding
    """
    batch_size, seq_len, dim = tokens.shape
    original_seq_len = seq_len
    
    # Calculate padding needed
    remainder = seq_len % window_size
    if remainder != 0:
        padding_needed = window_size - remainder
        # Pad the sequence
        padding = torch.full(
            (batch_size, padding_needed, dim),
            pad_value,
            device=tokens.device,
            dtype=tokens.dtype
        )
        tokens = torch.cat([tokens, padding], dim=1)
        seq_len = tokens.shape[1]
    
    num_windows = seq_len // window_size
    
    # Reshape to windows: (batch, seq_len, dim) -> (batch, num_windows, window_size, dim)
    windowed = tokens.reshape(batch_size, num_windows, window_size, dim)
    
    # Create padding mask
    # True for valid tokens, False for padded tokens
    mask = torch.ones(batch_size, seq_len, device=tokens.device, dtype=torch.bool)
    if remainder != 0:
        mask[:, original_seq_len:] = False
    mask = mask.reshape(batch_size, num_windows, window_size)
    
    return windowed, mask, original_seq_len


def merge_windows(
    windowed: torch.Tensor,
    original_seq_len: int
) -> torch.Tensor:
    """
    Merge windowed tokens back to sequence format.
    
    Args:
        windowed: Windowed tokens of shape (batch, num_windows, window_size, dim)
        original_seq_len: Original sequence length before windowing
    
    Returns:
        Merged tokens of shape (batch, original_seq_len, dim)
    """
    batch_size, num_windows, window_size, dim = windowed.shape
    
    # Reshape: (batch, num_windows, window_size, dim) -> (batch, seq_len, dim)
    merged = windowed.reshape(batch_size, num_windows * window_size, dim)
    
    # Remove padding
    merged = merged[:, :original_seq_len, :]
    
    return merged


def select_windows(
    windowed: torch.Tensor,
    indices: torch.Tensor
) -> torch.Tensor:
    """
    Select specific windows based on indices (for top-k routing).
    
    Args:
        windowed: Windowed tokens of shape (batch, num_windows, window_size, dim)
        indices: Window indices to select of shape (batch, k) or (batch, heads, k)
    
    Returns:
        Selected windows of shape (batch, k, window_size, dim) or
        (batch, heads, k, window_size, dim)
    """
    batch_size, num_windows, window_size, dim = windowed.shape
    
    if indices.dim() == 2:
        # indices: (batch, k)
        k = indices.shape[1]
        # Expand indices for gather: (batch, k) -> (batch, k, window_size, dim)
        indices_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, window_size, dim)
        return torch.gather(windowed, dim=1, index=indices_expanded)
    else:
        # indices: (batch, heads, k)
        heads = indices.shape[1]
        k = indices.shape[2]
        # Expand windowed: (batch, num_windows, window_size, dim) -> (batch, heads, num_windows, window_size, dim)
        windowed_expanded = windowed.unsqueeze(1).expand(-1, heads, -1, -1, -1)
        # Expand indices: (batch, heads, k) -> (batch, heads, k, window_size, dim)
        indices_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, window_size, dim)
        return torch.gather(windowed_expanded, dim=2, index=indices_expanded)


def create_window_position_encoding(
    num_windows: int,
    window_size: int,
    dim: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create position encodings that are aware of window structure.
    
    Args:
        num_windows: Number of windows
        window_size: Size of each window
        dim: Dimension of embeddings
        device: Device to create tensor on
    
    Returns:
        Position encodings of shape (num_windows * window_size, dim)
    """
    seq_len = num_windows * window_size
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    
    # Create frequency bands
    dim_indices = torch.arange(dim, device=device)
    freq = 1.0 / (10000 ** (2 * (dim_indices // 2) / dim))
    
    # Compute position encodings
    pe = position * freq
    
    # Apply sin to even indices and cos to odd indices
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    
    return pe

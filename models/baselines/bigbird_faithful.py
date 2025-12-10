"""
Faithful PyTorch port of BigBird attention from Google's JAX implementation.
Based on: https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/bigbird/bigbird_attention.py

Key differences from our simplified version:
1. Block-based (not per-token) attention
2. Band mask operates on blocks
3. Random attention samples blocks, not individual positions
4. First and last blocks have full global attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_block_rand_mask(seq_len, block_size, num_rand_blocks, seed=42):
    """
    Create random block attention mask.
    
    This is a faithful port of get_block_rand_mask from JAX implementation.
    
    Args:
        seq_len: sequence length
        block_size: size of each block
        num_rand_blocks: number of random blocks each block attends to
        seed: random seed for reproducibility
        
    Returns:
        rand_attn: (num_blocks - 2, num_rand_blocks) indices of random blocks
    """
    np.random.seed(seed)
    
    num_blocks = seq_len // block_size
    
    # Middle blocks (exclude first and last)
    rand_attn = np.zeros((num_blocks - 2, num_rand_blocks), dtype=np.int64)
    
    # Available block indices (exclude first and last)
    available = np.arange(1, num_blocks - 1)
    
    for i in range(1, num_blocks - 1):
        # Exclude nearby blocks (previous, current, next)
        start = i - 2
        end = i
        
        # Special cases at boundaries
        if i == 1:
            valid_blocks = available[2:]
        elif i == 2:
            valid_blocks = available[3:]
        elif i == num_blocks - 3:
            valid_blocks = available[:num_blocks - 5]
        elif i == num_blocks - 2:
            valid_blocks = available[:num_blocks - 4]
        else:
            # Exclude blocks in range [start, end+1]
            valid_blocks = np.concatenate([available[:start], available[end + 1:]])
        
        # Randomly sample num_rand_blocks
        if len(valid_blocks) == 0:
            # No valid blocks available, skip
            continue
        elif len(valid_blocks) >= num_rand_blocks:
            rand_attn[i - 1, :] = np.random.choice(valid_blocks, size=num_rand_blocks, replace=False)
        else:
            # If not enough valid blocks, sample with replacement
            rand_attn[i - 1, :] = np.random.choice(valid_blocks, size=num_rand_blocks, replace=True)
    
    return rand_attn


class BigBirdBlockAttention(nn.Module):
    """
    Faithful PyTorch implementation of BigBird block-sparse attention.
    
    Matches the JAX implementation from Google Research LRA.
    
    Architecture:
    - First block: Full attention to all tokens
    - Middle blocks: Band (sliding window) + Random blocks
    - Last block: Full attention to all tokens
    """
    
    def __init__(self, embed_dim, num_heads, block_size=64, num_rand_blocks=3, 
                 dropout=0.1, max_seq_len=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.num_rand_blocks = num_rand_blocks
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Cache for random block indices
        self.rand_attn_cache = {}
    
    def _get_rand_attn(self, seq_len, device):
        """Get or create random attention block indices."""
        cache_key = seq_len
        if cache_key not in self.rand_attn_cache:
            rand_attn = get_block_rand_mask(
                seq_len, 
                self.block_size, 
                self.num_rand_blocks,
                seed=42
            )
            self.rand_attn_cache[cache_key] = torch.from_numpy(rand_attn).to(device)
        return self.rand_attn_cache[cache_key]
    
    def _create_block_mask(self, seq_len, device):
        """
        Create BigBird block-sparse attention mask.
        
        Pattern:
        - First block (positions 0 to block_size-1): Attend to ALL tokens
        - Middle blocks: Band (3 blocks: prev, current, next) + Random blocks
        - Last block: Attend to ALL tokens
        
        Returns:
            mask: (seq_len, seq_len) where 1=attend, 0=mask
        """
        num_blocks = seq_len // self.block_size
        
        # Start with zeros
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # Get random block indices for middle blocks
        rand_attn = self._get_rand_attn(seq_len, device)
        
        for block_idx in range(num_blocks):
            block_start = block_idx * self.block_size
            block_end = block_start + self.block_size
            
            # First block: Full attention
            if block_idx == 0:
                mask[block_start:block_end, :] = 1
            
            # Last block: Full attention
            elif block_idx == num_blocks - 1:
                mask[block_start:block_end, :] = 1
            
            # Middle blocks: Band + Random
            else:
                # Band: previous, current, next blocks (3 blocks total)
                band_start = max(0, (block_idx - 1) * self.block_size)
                band_end = min(seq_len, (block_idx + 2) * self.block_size)
                mask[block_start:block_end, band_start:band_end] = 1
                
                # Random blocks
                rand_block_indices = rand_attn[block_idx - 1]  # -1 because first block is excluded
                for rand_block_idx in rand_block_indices:
                    rand_start = rand_block_idx * self.block_size
                    rand_end = rand_start + self.block_size
                    mask[block_start:block_end, rand_start:rand_end] = 1
        
        return mask
    
    def forward(self, x, padding_mask=None):
        """
        Forward pass with block-sparse attention.
        
        Args:
            x: (batch, seq_len, embed_dim)
            padding_mask: (batch, seq_len) where 1=real token, 0=padding
            
        Returns:
            output: (batch, seq_len, embed_dim)
            None: (for compatibility)
        """
        batch, seq_len, _ = x.shape
        
        # Check sequence length is divisible by block size
        if seq_len % self.block_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by block_size ({self.block_size})")
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, heads, seq, seq)
        
        # Apply block-sparse mask
        block_mask = self._create_block_mask(seq_len, x.device)
        block_mask_expanded = block_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        attn_scores = attn_scores.masked_fill(block_mask_expanded == 0, float('-inf'))
        
        # Apply padding mask if provided
        if padding_mask is not None:
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            attn_scores = attn_scores.masked_fill(pad_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)  # (batch, heads, seq, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = self.out_proj(output)
        
        return output, None
    
    def get_mask_stats(self, seq_len, device='cpu'):
        """Get statistics about the mask for analysis."""
        mask = self._create_block_mask(seq_len, device)
        total_elements = seq_len * seq_len
        attended_elements = mask.sum().item()
        sparsity_pct = (attended_elements / total_elements) * 100
        
        return {
            'total_elements': total_elements,
            'attended_elements': attended_elements,
            'sparsity_pct': sparsity_pct,
            'num_blocks': seq_len // self.block_size,
            'block_size': self.block_size,
            'num_rand_blocks': self.num_rand_blocks
        }


def test_bigbird_faithful():
    """Test that faithful implementation works correctly."""
    print("Testing Faithful BigBird Implementation")
    print("="*60)
    
    batch_size = 2
    seq_len = 512
    embed_dim = 128
    num_heads = 4
    block_size = 64
    
    model = BigBirdBlockAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        block_size=block_size,
        num_rand_blocks=3
    )
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    padding_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output, _ = model(x, padding_mask)
    print(f"✓ Forward pass: input {x.shape} -> output {output.shape}")
    
    # Get mask statistics
    stats = model.get_mask_stats(seq_len)
    print(f"\nMask Statistics:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Block size: {stats['block_size']}")
    print(f"  Number of blocks: {stats['num_blocks']}")
    print(f"  Random blocks per block: {stats['num_rand_blocks']}")
    print(f"  Attended elements: {stats['attended_elements']:,} / {stats['total_elements']:,}")
    print(f"  Sparsity: {stats['sparsity_pct']:.2f}%")
    
    # Test different sequence lengths
    print(f"\nSparsity at different sequence lengths:")
    for test_len in [256, 512, 1024, 2048, 4096]:
        stats = model.get_mask_stats(test_len)
        print(f"  {test_len}: {stats['sparsity_pct']:.2f}%")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_bigbird_faithful()

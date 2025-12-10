"""
BiFormer Attention Adapter for 1D Sequences

Adapts BiFormer's Bi-Level Routing Attention for LRA sequence tasks.
Original: 2D images (NCHW) -> 1D sequences (N, L, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BiLevelRoutingAttention1D(nn.Module):
    """
    Bi-Level Routing Attention adapted for 1D sequences.
    
    Key changes from original BiFormer:
    1. Input: (N, L, C) instead of (N, C, H, W)
    2. Regions: Consecutive chunks instead of 2D windows
    3. All operations in sequence dimension instead of spatial
    """
    
    def __init__(self, embed_dim, num_heads=8, n_regions=7, topk=4, 
                 qk_scale=None, side_conv=5, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # Regional routing params
        self.n_regions = n_regions  # Number of regions to split sequence into
        self.topk = topk  # Top-k regions to attend to
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Local Context Enhancement (LCE) - depthwise conv
        self.lepe = nn.Conv1d(embed_dim, embed_dim, kernel_size=side_conv,
                              padding=side_conv//2, groups=embed_dim) if side_conv > 0 else None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (N, L, C) tensor
            mask: (N, L) boolean mask (True = valid token)
        Returns:
            output: (N, L, C) tensor
            attn_mask: (N, num_heads, L, L) sparse attention mask for visualization
        """
        N, L, C = x.shape
        
        # STEP 1: Linear projection to QKV
        qkv = self.qkv(x)  # (N, L, 3*C)
        qkv = qkv.reshape(N, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (N, H, L, D)
        
        # STEP 2: Region-to-region routing
        region_size = max(1, L // self.n_regions)
        
        # Reshape into regions: (N, H, n_regions, region_size, D)
        pad_len = (region_size * self.n_regions) - L
        if pad_len > 0:
            q_pad = F.pad(q, (0, 0, 0, pad_len))
            k_pad = F.pad(k, (0, 0, 0, pad_len))
        else:
            q_pad = q
            k_pad = k
        
        L_padded = q_pad.size(2)
        actual_n_regions = L_padded // region_size
        
        # Average pool each region (coarse representation)
        q_regions = q_pad.reshape(N, self.num_heads, actual_n_regions, region_size, self.head_dim)
        k_regions = k_pad.reshape(N, self.num_heads, actual_n_regions, region_size, self.head_dim)
        
        # Region representatives (mean pooling, detached for non-parametric routing)
        q_r = q_regions.mean(dim=3).detach()  # (N, H, n_regions, D)
        k_r = k_regions.mean(dim=3).detach()  # (N, H, n_regions, D)
        
        # Compute region-to-region affinity
        attn_r = torch.einsum('nhrd,nhsd->nhrs', q_r, k_r)  # (N, H, n_regions, n_regions)
        attn_r = attn_r * self.scale
        
        # Top-k region selection per query region
        _, idx_topk = torch.topk(attn_r, k=min(self.topk, actual_n_regions), dim=-1)  # (N, H, n_regions, k)
        
        # STEP 3: Token-to-token attention within selected regions
        # For each query region, attend to tokens in its top-k key regions
        output_list = []
        attn_mask_sparse = torch.zeros(N, self.num_heads, L, L, device=x.device)
        
        for i in range(actual_n_regions):
            # Get query tokens from region i
            start_q = i * region_size
            end_q = min((i + 1) * region_size, L_padded)
            q_local = q_pad[:, :, start_q:end_q]  # (N, H, region_size, D)
            
            # Get key/value tokens from top-k regions
            k_local_list = []
            v_local_list = []
            
            for b in range(N):
                for h in range(self.num_heads):
                    topk_regions = idx_topk[b, h, i]  # (k,)
                    
                    # Gather k/v from selected regions
                    for r_idx in topk_regions:
                        start_k = r_idx * region_size
                        end_k = min((r_idx + 1) * region_size, L_padded)
                        k_local_list.append(k_pad[b:b+1, h:h+1, start_k:end_k])
                        v_local_list.append(v[b:b+1, h:h+1, start_k:end_k] if end_k <= L else 
                                          F.pad(v[b:b+1, h:h+1, start_k:L], (0, 0, 0, end_k-L)))
            
            # Concatenate selected keys/values
            # For simplicity, use full attention within selected regions
            # (More efficient impl would use gather operations)
            
        # Simplified: Use full attention with top-k masking
        attn_full = torch.einsum('nhqd,nhkd->nhqk', q, k) * self.scale  # (N, H, L, L)
        
        # Create routing mask based on top-k regions
        routing_mask = torch.zeros(N, self.num_heads, L, L, device=x.device, dtype=torch.bool)
        for i in range(actual_n_regions):
            start_q = i * region_size
            end_q = min((i + 1) * region_size, L)
            
            for b in range(N):
                for h in range(self.num_heads):
                    topk_regions = idx_topk[b, h, i]
                    for r_idx in topk_regions:
                        start_k = r_idx * region_size
                        end_k = min((r_idx + 1) * region_size, L)
                        routing_mask[b, h, start_q:end_q, start_k:end_k] = True
        
        # Apply routing mask
        attn_full = attn_full.masked_fill(~routing_mask, float('-inf'))
        
        # Apply padding mask if provided
        if mask is not None:
            # mask should be (N, L) - True = valid token, expand for broadcasting
            # attn_full is (N, H, L, L), we need mask to broadcast as (N, 1, 1, L) for keys
            key_mask = mask.view(N, 1, 1, L)  # (N, 1, 1, L) - broadcast over heads and queries
            attn_full = attn_full.masked_fill(~key_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_full, dim=-1)  # (N, H, L, L)
        attn_weights = F.dropout(attn_weights, p=self.proj_drop.p, training=self.training)
        
        # Compute attention output
        output = torch.einsum('nhql,nhld->nhqd', attn_weights, v)  # (N, H, L, D)
        output = output.transpose(1, 2).reshape(N, L, C)  # (N, L, C)
        
        # Add Local Context Enhancement (LCE)
        if self.lepe is not None:
            v_global = v.transpose(1, 2).reshape(N, L, C)  # (N, L, C)
            lce = self.lepe(v_global.transpose(1, 2)).transpose(1, 2)  # (N, L, C)
            output = output + lce
        
        # Final projection
        output = self.proj(output)
        output = self.proj_drop(output)
        
        # Return sparse attention mask for visualization
        attn_mask_sparse = routing_mask.float()
        
        return output, attn_mask_sparse


class BiFormerAttentionWrapper(nn.Module):
    """
    Wrapper to match DCM-MSR interface for fair comparison.
    Replaces MultiheadAttention with BiFormer's routing attention.
    """
    
    def __init__(self, embed_dim, num_heads, n_regions=7, topk=4, dropout=0.0):
        super().__init__()
        self.attention = BiLevelRoutingAttention1D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            n_regions=n_regions,
            topk=topk,
            dropout=dropout
        )
    
    def forward(self, query, key, value, key_padding_mask=None, 
                need_weights=False, attn_mask=None):
        """
        Interface compatible with nn.MultiheadAttention for drop-in replacement.
        
        Args:
            query: (L, N, C) - PyTorch MHA format
            key: (L, N, C)
            value: (L, N, C)
            key_padding_mask: (N, L) - True = ignore
            need_weights: bool
            attn_mask: Not used in BiFormer
        
        Returns:
            output: (L, N, C)
            attn_weights: (N, H, L, L) or None
        """
        # Convert from MHA format (L, N, C) to batch-first (N, L, C)
        query = query.transpose(0, 1)
        
        # Convert padding mask: True = ignore -> True = valid
        mask = ~key_padding_mask if key_padding_mask is not None else None
        
        # Forward pass
        output, attn_weights = self.attention(query, mask)
        
        # Convert back to MHA format
        output = output.transpose(0, 1)  # (L, N, C)
        
        if need_weights:
            return output, attn_weights
        return output, None


if __name__ == "__main__":
    # Test the adapter
    print("Testing BiFormer 1D Adapter...")
    
    batch_size = 2
    seq_len = 128
    embed_dim = 256
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -10:] = False  # Mask last 10 tokens
    
    biformer = BiLevelRoutingAttention1D(
        embed_dim=embed_dim,
        num_heads=num_heads,
        n_regions=8,
        topk=4
    )
    
    output, attn_mask = biformer(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention mask shape: {attn_mask.shape}")
    print(f"Attention sparsity: {(attn_mask == 0).float().mean().item():.2%}")
    print("✓ BiFormer adapter working!")

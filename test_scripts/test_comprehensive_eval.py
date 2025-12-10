"""Quick test to verify comprehensive_eval.py works without errors."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing imports...")

try:
    import torch
    print("✓ torch imported")
except Exception as e:
    print(f"✗ torch import failed: {e}")
    sys.exit(1)

try:
    from evaluation.comprehensive_eval import count_attention_flops, measure_ensemble_metrics
    print("✓ comprehensive_eval functions imported")
except Exception as e:
    print(f"✗ comprehensive_eval import failed: {e}")
    sys.exit(1)

try:
    from analysis.bigbird_faithful import BigBirdBlockAttention
    print("✓ BigBirdBlockAttention imported")
except Exception as e:
    print(f"✗ BigBirdBlockAttention import failed: {e}")
    sys.exit(1)

try:
    from dcm_msr.attention import DCMMSRAttention
    print("✓ DCMMSRAttention imported")
except Exception as e:
    print(f"✗ DCMMSRAttention import failed: {e}")
    sys.exit(1)

try:
    from dcm_msr.quantum_utils import compute_purity, compute_fidelity
    print("✓ quantum_utils functions imported")
except Exception as e:
    print(f"✗ quantum_utils import failed: {e}")
    sys.exit(1)

print("\nTesting BigBird forward pass...")
try:
    device = 'cpu'
    batch_size, seq_len, embed_dim = 2, 512, 256
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    bigbird = BigBirdBlockAttention(embed_dim, num_heads=4, block_size=64)
    output, _ = bigbird(x)  # Returns (output, None) tuple
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ BigBird forward pass successful: {x.shape} -> {output.shape}")
except Exception as e:
    print(f"✗ BigBird forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting DCM-MSR forward pass...")
try:
    dcm = DCMMSRAttention(embed_dim, num_heads=4, window_size=64, top_k=2)
    output, _ = dcm(x, x, x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ DCM-MSR forward pass successful: {x.shape} -> {output.shape}")
except Exception as e:
    print(f"✗ DCM-MSR forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting FLOPs counting...")
try:
    flops_std = count_attention_flops(512, 256, 4, 'standard')
    flops_bb = count_attention_flops(512, 256, 4, 'bigbird', block_size=64)
    flops_dcm = count_attention_flops(512, 256, 4, 'dcm_msr', window_size=64, top_k=2)
    
    print(f"✓ Standard FLOPs: {flops_std['total']:,.0f}")
    print(f"✓ BigBird FLOPs: {flops_bb['total']:,.0f}")
    print(f"✓ DCM-MSR FLOPs: {flops_dcm['total']:,.0f}")
except Exception as e:
    print(f"✗ FLOPs counting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed! comprehensive_eval.py is ready.")

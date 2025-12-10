"""
Quick test to validate comparison script dependencies and basic functionality.
"""

import sys
from pathlib import Path

print("="*80)
print("Testing DCM-MSR vs BiFormer Comparison Script")
print("="*80)

# Test 1: Check imports
print("\n[Test 1] Checking imports...")
try:
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    print("  ✓ PyTorch, NumPy, Matplotlib available")
    try:
        import seaborn as sns
        print("  ✓ Seaborn available")
    except ImportError:
        print("  ⚠ Seaborn not available (optional, will use matplotlib only)")
except ImportError as e:
    print(f"  ✗ Missing required dependency: {e}")
    sys.exit(1)

# Test 2: Check custom modules
print("\n[Test 2] Checking custom modules...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from dcm_msr import DCMMSRSelfAttention
    print("  ✓ DCMMSRSelfAttention imported")
except ImportError as e:
    print(f"  ✗ Failed to import DCM-MSR: {e}")
    sys.exit(1)

try:
    from analysis.lra_data_loader import get_lra_data
    print("  ✓ LRA data loader imported")
except ImportError as e:
    print(f"  ✗ Failed to import LRA data loader: {e}")
    sys.exit(1)

try:
    from analysis.biformer_adapter import BiFormerAttentionWrapper
    print("  ✓ BiFormer adapter imported")
except ImportError as e:
    print(f"  ✗ Failed to import BiFormer adapter: {e}")
    sys.exit(1)

# Test 3: Create minimal models
print("\n[Test 3] Creating test models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

try:
    # Test DCM-MSR
    dcm_attn = DCMMSRSelfAttention(
        embed_dim=128,
        num_heads=4,
        window_size=32,
        top_k=2,
        dropout=0.1
    )
    print("  ✓ DCM-MSR attention created")
    
    # Test BiFormer
    biformer_attn = BiFormerAttentionWrapper(
        embed_dim=128,
        num_heads=4,
        n_regions=8,
        topk=4,
        dropout=0.1
    )
    print("  ✓ BiFormer attention created")
    
    # Test standard attention
    std_attn = nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True)
    print("  ✓ Standard attention created")
    
except Exception as e:
    print(f"  ✗ Failed to create models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test forward pass
print("\n[Test 4] Testing forward passes...")
try:
    batch_size, seq_len, embed_dim = 2, 64, 128
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    # DCM-MSR
    dcm_attn = dcm_attn.to(device)
    out_dcm, _ = dcm_attn(x)  # Returns (output, attention_info)
    print(f"  ✓ DCM-MSR forward: {tuple(out_dcm.shape)}")
    
    # BiFormer (expects L, N, C format like nn.MultiheadAttention)
    biformer_attn = biformer_attn.to(device)
    key_padding_mask = ~mask
    x_lnc = x.transpose(0, 1)  # (N, L, C) -> (L, N, C)
    out_bi, _ = biformer_attn(x_lnc, x_lnc, x_lnc, key_padding_mask=key_padding_mask, need_weights=False)
    print(f"  ✓ BiFormer forward: {tuple(out_bi.shape)}")
    
    # Standard (batch_first=True, so expects N, L, C format)
    std_attn = std_attn.to(device)
    out_std, _ = std_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
    print(f"  ✓ Standard forward: {tuple(out_std.shape)}")
    
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test data loading
print("\n[Test 5] Testing synthetic data generation...")
try:
    lra_data = get_lra_data(
        task_name='ListOps',
        seq_len=128,  # Small for test
        use_real_data=False,
        vocab_size=256,
        num_train=50,  # Small dataset
        num_test=10,
        seed=42
    )
    print(f"  ✓ Synthetic ListOps data: {len(lra_data['train'])} train, {len(lra_data['test'])} test")
    
    # Test batch
    from torch.utils.data import DataLoader
    loader = DataLoader(lra_data['train'], batch_size=4, shuffle=False)
    inputs, targets = next(iter(loader))
    print(f"  ✓ Batch shape: inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}")
    
except Exception as e:
    print(f"  ✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test complete model
print("\n[Test 6] Testing UnifiedTransformer...")
try:
    from analysis.compare_dcm_biformer import UnifiedTransformer
    
    model = UnifiedTransformer(
        vocab_size=256,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=10,
        max_seq_len=128,
        attention_type='dcm_msr',
        window_size=32,
        top_k=2,
        n_regions=8,
        biformer_topk=4,
        dropout=0.1,
        pad_id=0
    ).to(device)
    
    # Test forward
    inputs = torch.randint(0, 256, (2, 128)).to(device)
    outputs = model(inputs)
    print(f"  ✓ Model forward: input={tuple(inputs.shape)}, output={tuple(outputs.shape)}")
    
    # Test with attention return
    outputs, attn_masks = model(inputs, return_attention=True)
    print(f"  ✓ Model with attention: {len(attn_masks)} layers")
    
except Exception as e:
    print(f"  ✗ UnifiedTransformer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Quick training test
print("\n[Test 7] Testing training loop (1 batch)...")
try:
    from analysis.compare_dcm_biformer import train_epoch, evaluate
    
    model = UnifiedTransformer(
        vocab_size=256,
        embed_dim=64,  # Smaller for speed
        num_heads=2,
        num_layers=1,
        num_classes=10,
        max_seq_len=128,
        attention_type='standard',  # Fastest
        dropout=0.1,
        pad_id=0
    ).to(device)
    
    # Mini dataset
    lra_data = get_lra_data(
        task_name='ListOps',
        seq_len=128,
        use_real_data=False,
        vocab_size=256,
        num_train=20,
        num_test=10,
        seed=42
    )
    
    train_loader = DataLoader(lra_data['train'], batch_size=4, shuffle=True)
    test_loader = DataLoader(lra_data['test'], batch_size=4, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # One epoch
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"  ✓ Training: loss={train_loss:.4f}, acc={train_acc:.2f}%")
    print(f"  ✓ Evaluation: loss={test_loss:.4f}, acc={test_acc:.2f}%")
    
except Exception as e:
    print(f"  ✗ Training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - Ready for Palmetto!")
print("="*80)

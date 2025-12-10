"""
Single model runner for comparison experiments.
Runs one attention mechanism at a time for accurate timing.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dcm_msr import DCMMSRSelfAttention
from analysis.lra_data_loader import get_lra_data
from analysis.biformer_adapter import BiFormerAttentionWrapper


class UnifiedTransformer(nn.Module):
    """Unified Transformer for fair comparison."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, 
                 num_classes, max_seq_len, attention_type='standard',
                 window_size=32, top_k=2, n_regions=8, biformer_topk=4,
                 dropout=0.1, pad_id=0):
        super().__init__()
        
        self.attention_type = attention_type
        self.embed_dim = embed_dim
        self.pad_id = pad_id
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if attention_type == 'dcm_msr':
                attn = DCMMSRSelfAttention(embed_dim, num_heads, window_size, top_k, dropout)
                self.layers.append(TransformerLayer(embed_dim, attn, dropout))
            elif attention_type == 'biformer':
                attn = BiFormerAttentionWrapper(embed_dim, num_heads, n_regions, biformer_topk, dropout)
                self.layers.append(TransformerLayer(embed_dim, attn, dropout, is_biformer=True))
            elif attention_type == 'standard':
                attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
                self.layers.append(TransformerLayer(embed_dim, attn, dropout))
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self.attention_masks = []
    
    def forward(self, x, return_attention=False):
        mask = (x != self.pad_id)
        
        x_emb = self.embedding(x)
        seq_len = x.size(1)
        x_emb = x_emb + self.pos_encoding[:, :seq_len, :]
        x_emb = self.dropout(x_emb)
        
        self.attention_masks = []
        for layer in self.layers:
            x_emb, attn_mask = layer(x_emb, mask, return_attention=return_attention)
            if return_attention:
                self.attention_masks.append(attn_mask)
        
        x_emb = self.norm(x_emb)
        mask_expanded = mask.unsqueeze(-1).float()
        x_pooled = (x_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        logits = self.classifier(x_pooled)
        
        if return_attention:
            return logits, self.attention_masks
        return logits


class TransformerLayer(nn.Module):
    """Single transformer layer with pluggable attention."""
    
    def __init__(self, embed_dim, attention, dropout, is_biformer=False):
        super().__init__()
        self.attention = attention
        self.is_biformer = is_biformer
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask, return_attention=False):
        x_norm = self.norm1(x)
        
        if self.is_biformer:
            key_padding_mask = ~mask
            attn_out, attn_mask = self.attention(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention
            )
        elif isinstance(self.attention, DCMMSRSelfAttention):
            attention_mask = mask if mask is not None else None
            attn_out, attn_info = self.attention(x_norm, attention_mask, return_attention)
            attn_mask = attn_info[0] if return_attention and attn_info else None
        else:
            key_padding_mask = ~mask
            attn_out, attn_mask = self.attention(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention
            )
        
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def compute_attention_sparsity(model, dataloader, device, num_batches=5):
    """Compute average attention sparsity."""
    model.eval()
    sparsities = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _, attn_masks = model(inputs, return_attention=True)
            
            for attn_mask in attn_masks:
                if attn_mask is not None:
                    sparsity = (attn_mask == 0).float().mean().item()
                    sparsities.append(sparsity)
    
    return np.mean(sparsities) if sparsities else 0.0


def save_attention_sample(model, dataloader, device, output_path):
    """Save first attention pattern for visualization."""
    model.eval()
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        _, attn_masks = model(inputs, return_attention=True)
    
    if attn_masks and attn_masks[0] is not None:
        # Save first layer, first head, first sample
        attn = attn_masks[0][0, 0].cpu().numpy()
        np.save(output_path, attn)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Run single model comparison')
    parser.add_argument('--task', type=str, required=True, choices=['ListOps', 'Text', 'Retrieval'])
    parser.add_argument('--seq-len', type=int, required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--num-train', type=int, required=True)
    parser.add_argument('--num-test', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--attention-type', type=str, required=True, 
                       choices=['standard', 'dcm_msr', 'biformer'])
    parser.add_argument('--output-prefix', type=str, required=True)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Training {args.attention_type.upper()} on {args.task}")
    print("=" * 80)
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\nLoading synthetic {args.task} data...")
    lra_data = get_lra_data(
        task_name=args.task,
        seq_len=args.seq_len,
        use_real_data=False,
        vocab_size=args.vocab_size,
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed
    )
    
    train_loader = DataLoader(lra_data['train'], batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(lra_data['test'], batch_size=args.batch_size * 2, shuffle=False)
    
    # Create model
    model = UnifiedTransformer(
        vocab_size=lra_data['vocab_size'],
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        max_seq_len=args.seq_len,
        attention_type=args.attention_type,
        window_size=32,
        top_k=2,
        n_regions=8,
        biformer_topk=4,
        dropout=0.1,
        pad_id=lra_data.get('pad_id', 0)
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    train_accs = []
    test_accs = []
    epoch_times = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, "
              f"Time={epoch_time:.1f}s")
    
    # Compute metrics
    print("\nComputing final metrics...")
    sparsity = compute_attention_sparsity(model, test_loader, device)
    
    # Save attention sample
    attn_saved = save_attention_sample(model, test_loader, device, 
                                       f"{args.output_prefix}_attention.npy")
    
    # Estimate FLOPs per forward pass
    # Attention FLOPs: 2 * seq_len^2 * embed_dim (for standard attention)
    # DCM-MSR adds routing overhead but operates on windows, so comparable
    seq_len = args.seq_len
    embed_dim = args.embed_dim
    num_layers = args.num_layers
    
    # Rough FLOP estimate (forward pass only)
    attn_flops = 2 * seq_len * seq_len * embed_dim * num_layers  # QK^T and softmax @ V
    ffn_flops = 2 * seq_len * embed_dim * (4 * embed_dim) * num_layers  # Two linear layers
    total_flops = attn_flops + ffn_flops
    
    # Results
    results = {
        'task': args.task,
        'attention_type': args.attention_type,
        'config': {
            'embed_dim': args.embed_dim,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'seq_len': args.seq_len,
            'vocab_size': args.vocab_size,
            'num_classes': args.num_classes,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size
        },
        'results': {
            'final_train_acc': float(train_accs[-1]),
            'final_test_acc': float(test_accs[-1]),
            'best_test_acc': float(max(test_accs)),
            'avg_epoch_time': float(np.mean(epoch_times)),
            'total_time': float(sum(epoch_times)),
            'attention_sparsity': float(sparsity),
            'num_params': num_params,
            'estimated_flops': int(total_flops),
            'flops_per_second': float(total_flops / np.mean(epoch_times)) if np.mean(epoch_times) > 0 else 0,
            'train_accs': [float(x) for x in train_accs],
            'test_accs': [float(x) for x in test_accs],
            'epoch_times': [float(x) for x in epoch_times],
            'attention_saved': attn_saved
        }
    }
    
    # Save results
    output_file = f"{args.output_prefix}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"  Best Test Acc: {max(test_accs):.2f}%")
    print(f"  Avg Epoch Time: {np.mean(epoch_times):.2f}s")
    print(f"  Attention Sparsity: {sparsity:.1%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

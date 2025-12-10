"""
DCM-MSR vs BiFormer: Direct Comparison on Synthetic LRA Data

Controlled experiment with:
- Same backbone architecture (Transformer)
- Same number of layers, hidden size, heads
- Same training hyperparameters
- Only difference: Attention mechanism (DCM-MSR vs BiFormer vs Standard)

Metrics:
1. Accuracy (test)
2. Attention sparsity
3. Training time per epoch
4. GPU memory usage
5. Attention pattern visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib only")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dcm_msr import DCMMSRSelfAttention
from analysis.lra_data_loader import get_lra_data
from analysis.biformer_adapter import BiFormerAttentionWrapper


class UnifiedTransformer(nn.Module):
    """
    Unified Transformer for fair comparison.
    Attention mechanism is pluggable.
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, 
                 num_classes, max_seq_len, attention_type='standard',
                 # DCM-MSR specific
                 window_size=32, top_k=2,
                 # BiFormer specific  
                 n_regions=7, biformer_topk=4,
                 # General
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
        
        # For attention visualization
        self.attention_masks = []
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len) token ids
            return_attention: If True, also return attention masks
        Returns:
            logits: (batch, num_classes)
            attn_masks: List of attention masks from each layer (optional)
        """
        mask = (x != self.pad_id)
        
        # Embedding + positional encoding
        x_emb = self.embedding(x)
        seq_len = x.size(1)
        x_emb = x_emb + self.pos_encoding[:, :seq_len, :]
        x_emb = self.dropout(x_emb)
        
        # Transformer layers
        self.attention_masks = []
        for layer in self.layers:
            x_emb, attn_mask = layer(x_emb, mask, return_attention=return_attention)
            if return_attention:
                self.attention_masks.append(attn_mask)
        
        # Classification: mean pooling over valid tokens
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
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) boolean
            return_attention: Return attention mask
        """
        # Self-attention
        x_norm = self.norm1(x)
        
        if self.is_biformer:
            # BiFormer uses padding mask directly
            key_padding_mask = ~mask
            attn_out, attn_mask = self.attention(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention
            )
        elif isinstance(self.attention, DCMMSRSelfAttention):
            # DCM-MSR returns (output, attention_info)
            attention_mask = mask if mask is not None else None
            attn_out, attn_info = self.attention(x_norm, attention_mask, return_attention)
            # Extract attention weights if available
            attn_mask = attn_info[0] if return_attention and attn_info else None
        else:
            # Standard attention
            key_padding_mask = ~mask
            attn_out, attn_mask = self.attention(
                x_norm, x_norm, x_norm,
                key_padding_mask=key_padding_mask,
                need_weights=return_attention
            )
        
        x = x + self.dropout(attn_out)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
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
            
            # Compute sparsity for each layer
            for attn_mask in attn_masks:
                if attn_mask is not None:
                    # Count zero entries
                    sparsity = (attn_mask == 0).float().mean().item()
                    sparsities.append(sparsity)
    
    return np.mean(sparsities) if sparsities else 0.0


def visualize_attention_comparison(models_dict, dataloader, device, save_path):
    """Visualize attention patterns from all models side-by-side."""
    fig, axes = plt.subplots(1, len(models_dict), figsize=(6*len(models_dict), 5))
    if len(models_dict) == 1:
        axes = [axes]
    
    # Get one batch
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    
    for idx, (name, model) in enumerate(models_dict.items()):
        model.eval()
        with torch.no_grad():
            _, attn_masks = model(inputs, return_attention=True)
        
        # Visualize first layer, first head, first sample
        if attn_masks and attn_masks[0] is not None:
            attn = attn_masks[0][0, 0].cpu().numpy()  # (seq_len, seq_len)
            
            if HAS_SEABORN:
                sns.heatmap(attn, cmap='YlOrRd', ax=axes[idx], 
                           cbar=True, square=True, vmin=0, vmax=1)
            else:
                im = axes[idx].imshow(attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
                plt.colorbar(im, ax=axes[idx])
            
            axes[idx].set_title(f'{name}\nSparsity: {(attn==0).mean():.1%}')
            axes[idx].set_xlabel('Key Position')
            axes[idx].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention visualization saved to {save_path}")


def run_comparison(task_name, seq_len, vocab_size, num_classes, 
                   num_train=1000, num_test=200, seed=42):
    """
    Run complete comparison experiment.
    
    Args:
        task_name: 'ListOps', 'Text', or 'Retrieval'
        seq_len: Sequence length
        vocab_size: Vocabulary size
        num_classes: Number of output classes
        num_train, num_test: Dataset sizes
        seed: Random seed
    """
    print("\n" + "="*80)
    print(f"DCM-MSR vs BiFormer Comparison: {task_name}")
    print("="*80)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading synthetic {task_name} data...")
    lra_data = get_lra_data(
        task_name=task_name,
        seq_len=seq_len,
        use_real_data=False,
        vocab_size=vocab_size,
        num_train=num_train,
        num_test=num_test,
        seed=seed
    )
    
    train_loader = DataLoader(lra_data['train'], batch_size=16, shuffle=True)
    test_loader = DataLoader(lra_data['test'], batch_size=32, shuffle=False)
    
    # Fixed hyperparameters (same for all models)
    config = {
        'vocab_size': lra_data['vocab_size'],
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'num_classes': num_classes,
        'max_seq_len': seq_len,
        'dropout': 0.1,
        'lr': 0.001,
        'num_epochs': 10,
        'pad_id': lra_data.get('pad_id', 0)
    }
    
    print(f"\nShared configuration:")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Vocabulary: {config['vocab_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    
    # Models to compare
    attention_types = ['standard', 'dcm_msr', 'biformer']
    results = {}
    models_dict = {}
    
    for attn_type in attention_types:
        print(f"\n{'='*80}")
        print(f"Training: {attn_type.upper()}")
        print(f"{'='*80}")
        
        # Create model
        model = UnifiedTransformer(
            attention_type=attn_type,
            window_size=32,  # DCM-MSR
            top_k=2,  # DCM-MSR
            n_regions=8,  # BiFormer
            biformer_topk=4,  # BiFormer
            **config
        ).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        # Training loop
        train_accs = []
        test_accs = []
        epoch_times = []
        
        for epoch in range(config['num_epochs']):
            start_time = time.time()
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            print(f"Epoch {epoch+1}/{config['num_epochs']}: "
                  f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, "
                  f"Time={epoch_time:.1f}s")
        
        # Compute attention sparsity
        sparsity = compute_attention_sparsity(model, test_loader, device)
        
        # Store results
        results[attn_type] = {
            'final_train_acc': train_accs[-1],
            'final_test_acc': test_accs[-1],
            'best_test_acc': max(test_accs),
            'avg_epoch_time': np.mean(epoch_times),
            'attention_sparsity': sparsity,
            'num_params': num_params,
            'train_accs': train_accs,
            'test_accs': test_accs
        }
        
        models_dict[attn_type] = model
        
        print(f"\nResults:")
        print(f"  Best Test Acc: {max(test_accs):.2f}%")
        print(f"  Avg Epoch Time: {np.mean(epoch_times):.2f}s")
        print(f"  Attention Sparsity: {sparsity:.1%}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON: {task_name}")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Best Acc':<12} {'Sparsity':<12} {'Time/Epoch':<12} {'Params':<12}")
    print(f"{'-'*80}")
    for attn_type in attention_types:
        r = results[attn_type]
        print(f"{attn_type:<15} {r['best_test_acc']:>10.2f}% {r['attention_sparsity']:>10.1%} "
              f"{r['avg_epoch_time']:>10.2f}s {r['num_params']:>10,}")
    
    # Visualize attention patterns
    vis_path = f"comparison_{task_name.lower()}_attention.png"
    visualize_attention_comparison(models_dict, test_loader, device, vis_path)
    
    # Save results
    output_file = f"comparison_{task_name.lower()}_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'task': task_name,
            'config': config,
            'results': {k: {**v, 'train_accs': [float(x) for x in v['train_accs']],
                                'test_accs': [float(x) for x in v['test_accs']]}
                       for k, v in results.items()}
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    # Run comparisons on all LRA tasks
    tasks = [
        {'name': 'ListOps', 'seq_len': 2048, 'vocab_size': 256, 'num_classes': 10},
        {'name': 'Text', 'seq_len': 4096, 'vocab_size': 256, 'num_classes': 2},
        {'name': 'Retrieval', 'seq_len': 4096, 'vocab_size': 256, 'num_classes': 2},
    ]
    
    all_results = {}
    for task in tasks:
        results = run_comparison(
            task_name=task['name'],
            seq_len=task['seq_len'],
            vocab_size=task['vocab_size'],
            num_classes=task['num_classes'],
            num_train=1000,
            num_test=200
        )
        all_results[task['name']] = results
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)

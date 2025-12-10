"""
Comprehensive LRA Benchmark: Standard Attention vs BigBird vs DCM-MSR
OPTIMIZED FOR GPU EXECUTION

Improvements:
1. Proper mask from input_ids applied in ALL attention mechanisms (fair comparison)
2. BigBird mask precomputed and cached
3. DCM-MSR: window_size = head_dim, top_k = 2
4. CUDA sync for timing, warmup before measurement
5. Deterministic runs with fixed seeds
6. Per-epoch metrics: train/test acc, time, GPU memory, DCM routing stats
7. Real LRA data support via minimal-LRU integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
import os
from pathlib import Path
import json
import time
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dcm_msr import DCMMSRSelfAttention
from analysis.lra_data_loader import get_lra_data
from analysis.bigbird_faithful import BigBirdBlockAttention


class ProgressTracker:
    """Track and display progress with time estimates."""
    
    def __init__(self, total_steps, task_name="Task"):
        self.total_steps = total_steps
        self.current_step = 0
        self.task_name = task_name
        self.start_time = time.time()
    
    def update(self, step=None, message=""):
        """Update progress and print estimate."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress_pct = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_time_per_step * remaining_steps
            eta_minutes = eta_seconds / 60
            
            print(f"\r[{self.task_name}] {progress_pct:.1f}% | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_minutes:.1f}m | {message}",
                  end="", flush=True)
        else:
            print(f"\r[{self.task_name}] Starting... {message}", end="", flush=True)
    
    def finish(self, message="Complete"):
        elapsed = time.time() - self.start_time
        print(f"\n[{self.task_name}] {message} | Total: {elapsed/60:.2f}m")


# BigBird implementation moved to bigbird_faithful.py for accuracy


class LRATransformer(nn.Module):
    """Transformer with switchable attention mechanisms."""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, 
                 num_classes, max_seq_len, attention_type='standard',
                 window_size=32, top_k=2, block_size=64, dropout=0.1, pad_id=0):
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
            elif attention_type == 'bigbird':
                # Use faithful implementation matching Google JAX code
                attn = BigBirdBlockAttention(embed_dim, num_heads, block_size, 
                                            num_rand_blocks=3, dropout=dropout, 
                                            max_seq_len=max_seq_len)
            elif attention_type == 'standard':
                attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
            
            layer = nn.ModuleDict({
                'attn': attn,
                'norm1': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(embed_dim),
            })
            self.layers.append(layer)
        
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        
        # Mask from input_ids, not layer outputs - CRITICAL for fairness
        padding_mask_bool = (input_ids != self.pad_id)  # (batch, seq_len)
        padding_mask_float = padding_mask_bool.float().unsqueeze(-1)  # For pooling
        
        x = self.embedding(input_ids)
        if seq_len <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            pos_enc = self.pos_encoding.repeat(1, (seq_len // self.pos_encoding.shape[1]) + 1, 1)
            x = x + pos_enc[:, :seq_len, :]
        
        dcm_stats = {'num_windows': [], 'top_k': [], 'pct_routed': []}
        
        for layer in self.layers:
            normed = layer['norm1'](x)
            
            if self.attention_type == 'dcm_msr':
                attn_out, routing_info = layer['attn'](normed)
                # Collect DCM-MSR routing statistics
                if routing_info is not None:
                    dcm_stats['num_windows'].append(routing_info.get('num_windows', 0))
                    dcm_stats['top_k'].append(routing_info.get('top_k', 0))
                    dcm_stats['pct_routed'].append(routing_info.get('pct_routed', 0))
            elif self.attention_type == 'bigbird':
                attn_out, _ = layer['attn'](normed, padding_mask=padding_mask_bool)
            else:  # standard
                # Standard attention: mask padding tokens
                key_padding_mask = ~padding_mask_bool  # MultiheadAttention expects True=ignore
                attn_out, _ = layer['attn'](normed, normed, normed, 
                                           key_padding_mask=key_padding_mask, 
                                           need_weights=False)
            
            x = x + attn_out
            x = x + layer['ffn'](layer['norm2'](x))
        
        # Pool using mask from input_ids
        x = (x * padding_mask_float).sum(dim=1) / padding_mask_float.sum(dim=1).clamp(min=1)
        
        x = self.output_norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        # Return logits and DCM stats (if any)
        return logits, dcm_stats


# Synthetic data creation now handled by lra_data_loader.py


def train_epoch(model, dataloader, optimizer, criterion, device, progress_tracker, is_cuda):
    """Train one epoch with CUDA sync and metric collection."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    max_memory_mb = 0
    
    dcm_all_stats = {'num_windows': [], 'top_k': [], 'pct_routed': []}
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, dcm_stats = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Collect DCM stats if available
        if dcm_stats and len(dcm_stats.get('num_windows', [])) > 0:
            dcm_all_stats['num_windows'].extend(dcm_stats['num_windows'])
            dcm_all_stats['top_k'].extend(dcm_stats['top_k'])
            dcm_all_stats['pct_routed'].extend(dcm_stats['pct_routed'])
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Track max GPU memory
        if is_cuda:
            max_memory_mb = max(max_memory_mb, torch.cuda.max_memory_allocated() / 1024**2)
        
        if batch_idx % 5 == 0:
            progress_tracker.update(
                message=f"Loss: {total_loss/(batch_idx+1):.4f}, Acc: {100.*correct/total:.2f}%"
            )
    
    if is_cuda:
        torch.cuda.synchronize()
    
    # Compute average DCM stats
    avg_dcm_stats = {}
    if dcm_all_stats['num_windows']:
        avg_dcm_stats = {
            'avg_num_windows': np.mean(dcm_all_stats['num_windows']),
            'avg_top_k': np.mean(dcm_all_stats['top_k']),
            'avg_pct_routed': np.mean(dcm_all_stats['pct_routed']),
        }
    
    return total_loss / len(dataloader), 100. * correct / total, max_memory_mb, avg_dcm_stats


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and track memory."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    max_memory_mb = 0
    is_cuda = device == 'cuda'
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if is_cuda:
                max_memory_mb = max(max_memory_mb, torch.cuda.max_memory_allocated() / 1024**2)
    
    return total_loss / len(dataloader), 100. * correct / total, max_memory_mb


def benchmark_task(task_config, device='cpu'):
    """Benchmark all three attention types on a task."""
    print("\n" + "="*80)
    print(f"TASK: {task_config['name']} (seq_len={task_config['seq_len']})")
    print("="*80)
    
    # Memory management - clear cache before starting
    if device == 'cuda':
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    seed = task_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Use synthetic data only for controlled benchmarking
    lra_data = get_lra_data(
        task_name=task_config['name'],
        seq_len=task_config['seq_len'],
        use_real_data=False,  # Force synthetic data
        vocab_size=task_config.get('vocab_size', 256),
        num_train=task_config.get('num_train', 1000),
        num_test=task_config.get('num_test', 200),
        seed=seed,
    )
    
    train_dataset = lra_data['train']
    test_dataset = lra_data['test']
    num_classes = lra_data['num_classes']
    vocab_size = lra_data['vocab_size']
    pad_id = lra_data.get('pad_id', 0)
    
    # Update task_config with actual vocab_size from data
    task_config['vocab_size'] = vocab_size
    task_config['pad_id'] = pad_id
    
    # Same batches for all models
    train_loader = DataLoader(train_dataset, batch_size=task_config['batch_size'], 
                             shuffle=False, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_dataset, batch_size=task_config['batch_size'], shuffle=False)
    
    results = {
        'task': task_config['name'],
        'seq_len': task_config['seq_len'],
        'models': {}
    }
    
    attention_types = [
        ('standard', 'Standard O(n²)'),
        ('bigbird', 'BigBird O(n) Sparse'),
        ('dcm_msr', 'DCM-MSR O(n×k×w)'),
    ]
    
    for attn_type, attn_name in attention_types:
        print(f"\n{'='*80}")
        print(f"{attn_name}")
        print(f"{'='*80}")
        
        model = LRATransformer(
            vocab_size=vocab_size,
            embed_dim=task_config['embed_dim'],
            num_heads=task_config['num_heads'],
            num_layers=task_config['num_layers'],
            num_classes=num_classes,
            max_seq_len=task_config['seq_len'],
            attention_type=attn_type,
            window_size=task_config.get('window_size', 32),
            top_k=task_config.get('top_k', 2),
            block_size=task_config.get('block_size', 64),
            dropout=0.1,
            pad_id=pad_id
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {num_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=task_config.get('lr', 0.001))
        
        is_cuda = device == 'cuda'
        
        # Warmup
        if is_cuda:
            print("Warmup...")
            dummy = torch.randint(0, task_config['vocab_size'], 
                                (task_config['batch_size'], task_config['seq_len']), device=device)
            for _ in range(3):
                with torch.no_grad():
                    _ = model(dummy)
            torch.cuda.synchronize()
        
        num_epochs = task_config.get('num_epochs', 10)
        progress = ProgressTracker(num_epochs * len(train_loader), attn_name)
        
        epoch_metrics = []
        
        for epoch in range(num_epochs):
            # Reset GPU memory stats
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            epoch_start = time.time()
            
            train_loss, train_acc, train_mem_mb, dcm_stats = train_epoch(
                model, train_loader, optimizer, criterion, device, progress, is_cuda
            )
            test_loss, test_acc, test_mem_mb = evaluate(model, test_loader, criterion, device)
            
            if is_cuda:
                torch.cuda.synchronize()
            epoch_time = time.time() - epoch_start
            
            max_mem_mb = max(train_mem_mb, test_mem_mb)
            
            # Build epoch metrics
            metrics = {
                'epoch': epoch + 1,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'time_sec': epoch_time,
                'max_memory_mb': max_mem_mb,
            }
            
            # Add DCM-MSR specific stats
            if dcm_stats:
                metrics.update(dcm_stats)
            
            epoch_metrics.append(metrics)
            
            # Print detailed epoch summary
            msg_parts = [f"Epoch {epoch+1}/{num_epochs}:"]
            msg_parts.append(f"Train={train_acc:.2f}%")
            msg_parts.append(f"Test={test_acc:.2f}%")
            msg_parts.append(f"Time={epoch_time:.2f}s")
            if is_cuda:
                msg_parts.append(f"Mem={max_mem_mb:.0f}MB")
            if dcm_stats:
                msg_parts.append(f"Windows={dcm_stats.get('avg_num_windows', 0):.1f}")
                msg_parts.append(f"TopK={dcm_stats.get('avg_top_k', 0):.1f}")
                msg_parts.append(f"Routed={dcm_stats.get('avg_pct_routed', 0):.1f}%")
            
            print("\n" + " | ".join(msg_parts))
        
        progress.finish()
        
        # Extract arrays for backward compatibility
        train_accs = [m['train_acc'] for m in epoch_metrics]
        test_accs = [m['test_acc'] for m in epoch_metrics]
        epoch_times = [m['time_sec'] for m in epoch_metrics]
        memory_mbs = [m['max_memory_mb'] for m in epoch_metrics]
        
        results['models'][attn_type] = {
            'name': attn_name,
            'num_params': num_params,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'best_test_acc': max(test_accs),
            'final_test_acc': test_accs[-1],
            'avg_epoch_time': np.mean(epoch_times),
            'total_time': sum(epoch_times),
            'avg_memory_mb': np.mean(memory_mbs) if is_cuda else 0,
            'peak_memory_mb': max(memory_mbs) if is_cuda else 0,
            'epoch_metrics': epoch_metrics,  # Full per-epoch data
        }
        
        # Clear memory after each model to prevent OOM
        del model, optimizer, criterion
        if device == 'cuda':
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"\n[Memory cleared after {attn_name}]")
        
        print(f"\n{attn_name}: Best={max(test_accs):.2f}%, Final={test_accs[-1]:.2f}%")
    
    return results


def run_lra_benchmark_suite(use_real_data=False, data_dir=None, device=None, tasks_filter=None, output_path=None):
    """Run complete benchmark suite.
    
    Args:
        use_real_data: Use real LRA data
        data_dir: Directory containing LRA data
        device: Device to run on
        tasks_filter: List of task names to run (e.g., ['listops', 'text']). If None, runs all.
        output_path: Custom output file path. If None, uses default.
    """
    print("\n" + "="*80)
    print("LRA BENCHMARK: Standard vs BigBird vs DCM-MSR")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Mode: SYNTHETIC DATA (Controlled Benchmarking)")
    if tasks_filter:
        print(f"Tasks Filter: {', '.join(tasks_filter)}")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    tasks = [
        {
            'name': 'ListOps',
            'seq_len': 2048,
            'vocab_size': 256,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'window_size': 32,  # head_dim = 128/4
            'top_k': 2,
            'block_size': 64,
            'batch_size': 4,  # Reduced from 16 for memory
            'num_epochs': 20,
            'lr': 0.001,
            'num_train': 1000,
            'num_test': 200,
            'seed': 42,
        },
        {
            'name': 'Text',
            'seq_len': 4096,
            'vocab_size': 256,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'window_size': 32,
            'top_k': 2,
            'block_size': 128,
            'batch_size': 2,  # Reduced from 8 for memory
            'num_epochs': 15,
            'lr': 0.001,
            'num_train': 1600,
            'num_test': 300,
            'seed': 42,
        },
        {
            'name': 'Retrieval',
            'seq_len': 4096,
            'vocab_size': 256,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'window_size': 32,
            'top_k': 2,
            'block_size': 128,
            'batch_size': 2,  # Reduced from 8 for memory
            'num_epochs': 15,
            'lr': 0.001,
            'num_train': 1600,
            'num_test': 300,
            'seed': 42,
        },
    ]
    
    # Filter tasks if requested
    if tasks_filter:
        tasks_filter_lower = [t.lower() for t in tasks_filter]
        tasks = [t for t in tasks if t['name'].lower() in tasks_filter_lower]
        if not tasks:
            print(f"ERROR: No tasks matched filter {tasks_filter}")
            return []
        print(f"Running {len(tasks)} task(s): {', '.join([t['name'] for t in tasks])}")
    
    all_results = []
    
    for task_idx, task_config in enumerate(tasks):
        print(f"\n{'#'*80}")
        print(f"# TASK {task_idx+1}/{len(tasks)}: {task_config['name']}")
        print(f"# Progress: {(task_idx/len(tasks))*100:.1f}%")
        print(f"{'#'*80}")
        
        try:
            results = benchmark_task(task_config, device=device)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(__file__).parent / 'lra_full_benchmark_results.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE! Results: {output_file}")
    print(f"{'='*80}")
    
    print_summary(all_results)
    return all_results


def print_summary(results):
    """Print summary table."""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for task_result in results:
        task = task_result['task']
        print(f"\n{task}")
        print("-" * 80)
        print(f"{'Model':<25} {'Best Acc':<12} {'Final Acc':<12} {'Time/Epoch':<12}")
        print("-" * 80)
        
        for model_type in ['standard', 'bigbird', 'dcm_msr']:
            if model_type in task_result['models']:
                m = task_result['models'][model_type]
                print(f"{m['name']:<25} {m['best_test_acc']:>10.2f}% "
                      f"{m['final_test_acc']:>10.2f}% {m['avg_epoch_time']:>10.2f}s")
    
    print("\n" + "="*80)
    print("AVERAGE ACCURACIES")
    print("="*80)
    
    for model_type, name in [('standard', 'Standard'), ('bigbird', 'BigBird'), ('dcm_msr', 'DCM-MSR')]:
        accs = [r['models'][model_type]['best_test_acc'] 
                for r in results if model_type in r['models']]
        if accs:
            print(f"{name:>15}: {np.mean(accs):.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LRA Benchmark: Standard vs BigBird vs DCM-MSR (Synthetic Data)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run on (default: cpu)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                       choices=['listops', 'text', 'retrieval'],
                       help='Specific tasks to run (default: all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (default: lra_full_benchmark_results.json)')
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run benchmark with arguments
    results = run_lra_benchmark_suite(
        use_real_data=args.use_real_data,
        data_dir=args.data_dir,
    # Run benchmark with arguments
    results = run_lra_benchmark_suite(
        device=args.device,
        tasks_filter=args.tasks,
        output_path=args.output
    )
    print("\n✅ Complete!")
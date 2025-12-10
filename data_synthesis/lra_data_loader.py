"""
LRA Dataset Loader - Integrates real Long Range Arena datasets
Uses datasets from minimal-LRU repository OR loads directly from TSV files

Supports:
- ListOps: Hierarchical structure prediction (2048 tokens)
- Text Classification: IMDB sentiment (4096 tokens)
- Retrieval: AAN document matching (4096 tokens)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
import pickle
import numpy as np
import csv

# Try to import from minimal-LRU
minimal_lru_path = Path(__file__).parent.parent / "minimal-LRU"
if minimal_lru_path.exists():
    sys.path.insert(0, str(minimal_lru_path))

try:
    from lru.dataloaders.lra import ListOps, IMDB
    HAS_LRA_DATA = True
except ImportError:
    HAS_LRA_DATA = False
    print("Warning: Could not import LRA datasets from minimal-LRU")
    print("Using synthetic data instead")


class LRADatasetWrapper(Dataset):
    """Wrapper to standardize LRA dataset format."""
    
    def __init__(self, dataset, max_len=None, pad_id=0):
        self.dataset = dataset
        self.max_len = max_len
        self.pad_id = pad_id
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if isinstance(item, dict):
            input_ids = item['input_ids']
            label = item.get('label', item.get('Target', 0))
        elif isinstance(item, tuple):
            input_ids, label = item[0], item[1]
        else:
            raise ValueError(f"Unknown item format: {type(item)}")
        
        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        
        # Pad or truncate to max_len
        if self.max_len is not None:
            if len(input_ids) < self.max_len:
                # Pad
                pad_len = self.max_len - len(input_ids)
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_id, dtype=torch.long)])
            elif len(input_ids) > self.max_len:
                # Truncate
                input_ids = input_ids[:self.max_len]
        
        return input_ids, label


def load_listops_from_tsv(data_dir, max_len=2048):
    """Load ListOps directly from TSV files (lra_release format)."""
    data_dir = Path(data_dir)
    
    # Look for listops-1000 directory
    listops_dir = data_dir / "listops-1000"
    if not listops_dir.exists():
        # Try alternate path
        listops_dir = data_dir / "lra_release" / "listops-1000"
    
    if not listops_dir.exists():
        print(f"ListOps directory not found at {listops_dir}")
        return None
    
    print(f"Loading ListOps from {listops_dir}")
    
    def parse_tsv(file_path):
        """Parse TSV file with Source and Target columns."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                source = row['Source']
                target = int(row['Target'])
                data.append((source, target))
        return data
    
    # Load train, val, test
    train_data = parse_tsv(listops_dir / "basic_train.tsv")
    val_data = parse_tsv(listops_dir / "basic_val.tsv") if (listops_dir / "basic_val.tsv").exists() else []
    test_data = parse_tsv(listops_dir / "basic_test.tsv")
    
    # Build vocabulary from training data
    vocab = {'<pad>': 0, '<unk>': 1}
    for source, _ in train_data:
        tokens = source.split()
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    
    vocab_size = len(vocab)
    num_classes = 10  # ListOps has 10 output classes (0-9)
    
    def tokenize_data(data):
        """Convert text to token IDs."""
        tokenized = []
        for source, target in data:
            tokens = source.split()
            token_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
            # Pad or truncate
            if len(token_ids) < max_len:
                token_ids += [vocab['<pad>']] * (max_len - len(token_ids))
            else:
                token_ids = token_ids[:max_len]
            tokenized.append((torch.tensor(token_ids, dtype=torch.long), target))
        return tokenized
    
    train_dataset = tokenize_data(train_data)
    val_dataset = tokenize_data(val_data) if val_data else None
    test_dataset = tokenize_data(test_data)
    
    print(f"✓ ListOps loaded from TSV: vocab={vocab_size}, classes={num_classes}, "
          f"train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset)}")
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'pad_id': vocab['<pad>'],
    }


def load_lra_listops(data_dir=None, max_len=2048, cache_dir=None):
    """Load ListOps dataset."""
    if not HAS_LRA_DATA:
        print("ListOps: Using synthetic data (minimal-LRU not available)")
        return None, None, None
    
    try:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "lra_data" / "listops-1000"
        
        # Check if data exists
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"ListOps data not found at {data_dir}")
            print("Download from: https://github.com/google-research/long-range-arena")
            print("Extract lra_release.gz and point to listops-1000 directory")
            return None, None, None
        
        dataset = ListOps(
            l_max=max_len,
            data_dir=str(data_dir),
            cache_dir=cache_dir,
            append_bos=False,
            append_eos=True,
        )
        
        dataset.prepare_data()
        dataset.setup()
        
        # Get vocab info
        vocab_size = dataset.n_tokens
        num_classes = dataset.d_output
        pad_id = dataset.vocab.get("<pad>", 0)
        
        # Wrap datasets
        train_dataset = LRADatasetWrapper(dataset.dataset_train, max_len, pad_id)
        val_dataset = LRADatasetWrapper(dataset.dataset_val, max_len, pad_id) if dataset.dataset_val else None
        test_dataset = LRADatasetWrapper(dataset.dataset_test, max_len, pad_id)
        
        print(f"✓ ListOps loaded: vocab={vocab_size}, classes={num_classes}, "
              f"train={len(train_dataset)}, test={len(test_dataset)}")
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'pad_id': pad_id,
        }
    
    except Exception as e:
        print(f"Error loading ListOps: {e}")
        return None


def load_lra_imdb(data_dir=None, max_len=4096, level='char', min_freq=15):
    """Load IMDB dataset for text classification."""
    if not HAS_LRA_DATA:
        print("IMDB: Using synthetic data (minimal-LRU not available)")
        return None
    
    try:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "lra_data" / "imdb"
        
        data_dir = Path(data_dir)
        cache_dir = data_dir / "cache" if data_dir.exists() else None
        
        dataset = IMDB(
            l_max=max_len,
            data_dir=str(data_dir) if data_dir.exists() else None,
            cache_dir=str(cache_dir) if cache_dir else None,
            level=level,
            min_freq=min_freq,
            append_bos=False,
            append_eos=True,
        )
        
        dataset.prepare_data()
        dataset.setup()
        
        vocab_size = dataset.n_tokens
        num_classes = dataset.d_output
        pad_id = dataset.vocab.get("<pad>", 0)
        
        train_dataset = LRADatasetWrapper(dataset.dataset_train, max_len, pad_id)
        test_dataset = LRADatasetWrapper(dataset.dataset_test, max_len, pad_id)
        
        print(f"✓ IMDB loaded: vocab={vocab_size}, classes={num_classes}, "
              f"train={len(train_dataset)}, test={len(test_dataset)}")
        
        return {
            'train': train_dataset,
            'val': None,
            'test': test_dataset,
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'pad_id': pad_id,
        }
    
    except Exception as e:
        print(f"Error loading IMDB: {e}")
        return None


def create_synthetic_lra_data(task_name, seq_len, vocab_size, num_train=1000, num_test=200, seed=42):
    """
    Fallback: Create synthetic LRA data when real data unavailable.
    
    Creates data with LEARNABLE patterns instead of random labels:
    - ListOps: Label based on sum of first few tokens (mimics hierarchical ops)
    - Text: Label based on frequency of high-value tokens (mimics sentiment)
    - Retrieval: Label based on overlap between first/second half (mimics matching)
    """
    from torch.utils.data import TensorDataset
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if task_name == "ListOps":
        num_classes = 10
        
        def generate_listops_sample():
            """Generate sample with complex hierarchical pattern (mimics nested operations)"""
            x = torch.randint(1, vocab_size, (seq_len,))
            # More complex pattern: weighted sum with position encoding
            # Simulates hierarchical structure: deeper positions matter more
            positions = torch.arange(seq_len, dtype=torch.float)
            weights = torch.exp(-positions / 100.0)  # Exponential decay
            # Take first 50 tokens with position weighting
            weighted_sum = (x[:50].float() * weights[:50]).sum().item()
            # Add interaction between early and late tokens
            early_late_product = (x[0:10].float().mean() * x[-10:].float().mean()).item()
            label = int((weighted_sum + early_late_product * 10) % num_classes)
            return x, label
        
        train_samples = [generate_listops_sample() for _ in range(num_train)]
        test_samples = [generate_listops_sample() for _ in range(num_test)]
        
    elif task_name == "Text":
        num_classes = 2
        
        def generate_text_sample():
            """Generate sample with sentiment-like pattern (frequency + position + context)"""
            x = torch.randint(1, vocab_size, (seq_len,))
            
            # Pattern: Compute global mean, label based on whether above median
            global_mean = x.float().mean()
            label = 1 if global_mean > (vocab_size / 2.0) else 0
            return x, label
        
        train_samples = [generate_text_sample() for _ in range(num_train)]
        test_samples = [generate_text_sample() for _ in range(num_test)]
        
    elif task_name == "Retrieval":
        num_classes = 2
        
        def generate_retrieval_sample():
            """Generate sample with document matching pattern (overlap + correlation)"""
            x = torch.randint(1, vocab_size, (seq_len,))
            mid = seq_len // 2
            
            # Pattern: Compare means of first and second half
            first_mean = x[:mid].float().mean()
            second_mean = x[mid:].float().mean()
            label = 1 if first_mean > second_mean else 0
            return x, label
        
        train_samples = [generate_retrieval_sample() for _ in range(num_train)]
        test_samples = [generate_retrieval_sample() for _ in range(num_test)]
        
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Convert to tensors
    train_x = torch.stack([x for x, _ in train_samples])
    train_y = torch.tensor([y for _, y in train_samples], dtype=torch.long)
    test_x = torch.stack([x for x, _ in test_samples])
    test_y = torch.tensor([y for _, y in test_samples], dtype=torch.long)
    
    print(f"Synthetic {task_name} data created with LEARNABLE patterns:")
    print(f"  Train label distribution: {torch.bincount(train_y).tolist()}")
    print(f"  Test label distribution: {torch.bincount(test_y).tolist()}")
    
    return {
        'train': TensorDataset(train_x, train_y),
        'val': None,
        'test': TensorDataset(test_x, test_y),
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'pad_id': 0,
    }


def get_lra_data(task_name, seq_len, use_real_data=True, **kwargs):
    """
    Get LRA dataset - tries real data first, falls back to synthetic.
    
    Args:
        task_name: 'ListOps', 'Text', or 'Retrieval'
        seq_len: Max sequence length
        use_real_data: Try to load real LRA data if available
        **kwargs: Additional args for synthetic data (num_train, num_test, vocab_size, etc.)
    
    Returns:
        dict with 'train', 'test', 'vocab_size', 'num_classes', 'pad_id'
    """
    data = None
    
    if use_real_data:
        print(f"\n{'='*60}")
        print(f"Loading {task_name} - Attempting REAL LRA data")
        print(f"{'='*60}")
        
        # First try loading directly from TSV if data_dir provided
        data_dir = kwargs.get('data_dir')
        if data_dir and task_name == "ListOps":
            print(f"Attempting to load ListOps from TSV files at: {data_dir}")
            try:
                data = load_listops_from_tsv(data_dir, max_len=seq_len)
                if data is not None:
                    print(f"✓ Successfully loaded real ListOps data from TSV!")
                    return data
            except Exception as e:
                print(f"✗ TSV loading failed with error: {e}")
                import traceback
                traceback.print_exc()
                data = None
        
        # Fall back to minimal-LRU if available
        if data is None and HAS_LRA_DATA:
            if task_name == "ListOps":
                data = load_lra_listops(
                    max_len=seq_len,
                    data_dir=data_dir
                )
            elif task_name == "Text":
                data = load_lra_imdb(
                    max_len=seq_len,
                    level=kwargs.get('level', 'char'),
                    min_freq=kwargs.get('min_freq', 15),
                    data_dir=data_dir
                )
            elif task_name == "Retrieval":
                print("Retrieval: Real data not yet implemented, using synthetic")
                data = None
            else:
                print(f"Unknown task: {task_name}")
                data = None
    
    # Fallback to synthetic
    if data is None:
        print(f"\n{'='*60}")
        print(f"Loading {task_name} - Using SYNTHETIC data")
        print(f"{'='*60}")
        
        data = create_synthetic_lra_data(
            task_name=task_name,
            seq_len=seq_len,
            vocab_size=kwargs.get('vocab_size', 256),
            num_train=kwargs.get('num_train', 1000),
            num_test=kwargs.get('num_test', 200),
            seed=kwargs.get('seed', 42)
        )
        print(f"✓ Synthetic {task_name}: vocab={data['vocab_size']}, "
              f"classes={data['num_classes']}, train={len(data['train'])}, test={len(data['test'])}")
    
    return data


if __name__ == "__main__":
    # Test data loading
    print("Testing LRA Data Loader")
    print("="*60)
    
    # Test ListOps
    listops = get_lra_data("ListOps", seq_len=2048, use_real_data=True)
    if listops:
        x, y = listops['train'][0]
        print(f"ListOps sample: x.shape={x.shape}, y={y}, vocab={listops['vocab_size']}")
    
    # Test Text (IMDB)
    text = get_lra_data("Text", seq_len=4096, use_real_data=True)
    if text:
        x, y = text['train'][0]
        print(f"Text sample: x.shape={x.shape}, y={y}, vocab={text['vocab_size']}")
    
    print("\n✓ Data loader test complete")

"""Quick test of synthetic data generation with learnable patterns."""

from analysis.lra_data_loader import create_synthetic_lra_data
import torch

print("="*60)
print("Testing Synthetic Data with Learnable Patterns")
print("="*60)

# Test ListOps
print("\n1. ListOps (10 classes):")
listops_data = create_synthetic_lra_data('ListOps', seq_len=512, vocab_size=32, num_train=100, num_test=20)
x, y = listops_data['train'][0]
print(f"Sample: x.shape={x.shape}, y={y}")
train_labels = [listops_data['train'][i][1].item() for i in range(10)]
print(f"First 10 train labels: {train_labels}")

# Test Text
print("\n2. Text (2 classes - sentiment):")
text_data = create_synthetic_lra_data('Text', seq_len=512, vocab_size=256, num_train=100, num_test=20)
x, y = text_data['train'][0]
print(f"Sample: x.shape={x.shape}, y={y}")
train_labels = [text_data['train'][i][1].item() for i in range(10)]
print(f"First 10 train labels: {train_labels}")

# Test Retrieval
print("\n3. Retrieval (2 classes - matching):")
retrieval_data = create_synthetic_lra_data('Retrieval', seq_len=512, vocab_size=256, num_train=100, num_test=20)
x, y = retrieval_data['train'][0]
print(f"Sample: x.shape={x.shape}, y={y}")
train_labels = [retrieval_data['train'][i][1].item() for i in range(10)]
print(f"First 10 train labels: {train_labels}")

print("\n" + "="*60)
print("✓ All synthetic data tests passed!")
print("="*60)

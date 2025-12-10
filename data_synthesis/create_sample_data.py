"""
Create sample sentiment data for quick testing.
Generates synthetic sentiment examples if you don't have real data yet.
"""

import json
import csv
from pathlib import Path


def create_sample_sentiment_data():
    """Create small sample dataset for testing."""
    
    # Sample positive sentences
    positive_samples = [
        "This movie was absolutely fantastic and I loved every minute of it!",
        "Great product, highly recommend to everyone!",
        "Amazing experience, will definitely come back again.",
        "The service was excellent and the staff were very friendly.",
        "Best purchase I've made in years, totally worth it.",
        "Wonderful atmosphere and delicious food.",
        "I'm so happy with this decision, exceeded all expectations!",
        "Outstanding quality and great value for money.",
        "Impressive performance and beautiful design.",
        "Absolutely brilliant, one of the best I've seen!",
    ] * 10  # 100 positive samples
    
    # Sample negative sentences
    negative_samples = [
        "Terrible experience, would not recommend at all.",
        "Very disappointed with the quality and service.",
        "This was a complete waste of time and money.",
        "Poor performance and not worth the price.",
        "Horrible customer service, never going back.",
        "The worst product I've ever purchased.",
        "Extremely dissatisfied, complete disaster.",
        "Not good at all, very frustrating experience.",
        "Really bad quality, broke after one use.",
        "Awful, save your money and avoid this.",
    ] * 10  # 100 negative samples
    
    # Combine and create train/val split
    all_samples = [(text, 1) for text in positive_samples] + [(text, 0) for text in negative_samples]
    
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    # 80/20 split
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    print("Creating sample data files...")
    
    with open(data_dir / 'train.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for text, label in train_samples:
            writer.writerow([text, label])
    
    with open(data_dir / 'val.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for text, label in val_samples:
            writer.writerow([text, label])
    
    print(f"✅ Created data/train.csv ({len(train_samples)} samples)")
    print(f"✅ Created data/val.csv ({len(val_samples)} samples)")
    print()
    
    # Also save as JSON
    train_json = [{'text': text, 'label': label} for text, label in train_samples]
    val_json = [{'text': text, 'label': label} for text, label in val_samples]
    
    with open(data_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_json, f, indent=2)
    
    with open(data_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_json, f, indent=2)
    
    print(f"✅ Created data/train.json")
    print(f"✅ Created data/val.json")
    print()
    print("Sample data ready! Run:")
    print("  python train_sentiment.py data/train.csv data/val.csv")


if __name__ == "__main__":
    create_sample_sentiment_data()

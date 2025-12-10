# Quick Start Guide - DCM-MSR Experiments

## What's in This Folder?

All experimental code used for DCM-MSR research, organized by purpose:
- **models/**: DCM-MSR + baseline attention mechanisms
- **data_synthesis/**: Data generation scripts
- **training_scripts/**: Main training/comparison runners
- **test_scripts/**: Local validation tests
- **palmetto_batch_scripts/**: HPC batch job scripts

## Quick Commands

### Generate Synthetic Data
```bash
cd data_synthesis
python create_sample_data.py --output ../data/
```

### Run Local Tests (Before Deployment)
```bash
cd test_scripts
python test_comparison.py  # Validates all mechanisms work
```

### Train Single Model
```bash
cd training_scripts
python compare_single_model.py \
    --task listops \
    --attention-type dcm_msr \
    --data-dir ../data/ \
    --device cuda
```

### Submit Palmetto Job
```bash
cd palmetto_batch_scripts
qsub palmetto_compare_biformer.sh
```

## File Dependencies

**Training scripts need**:
- Models: `../models/dcm_msr/` and `../models/baselines/`
- Data loader: `../data_synthesis/lra_data_loader.py`

**If running standalone**: Adjust import paths to point to model directories.

## Key Results

### Job 7823737 (Dec 8)
- Script: `palmetto_eval_batch.sh`
- DCM-MSR vs BigBird: **2-4x faster, 6-8x sparser**
- No accuracy measured

### Job 7860145 (Dec 9)
- Script: `palmetto_compare_biformer.sh` + `compare_single_model.py`
- DCM-MSR vs Standard: **Within 2-3% accuracy, 2.4x slower**
- Validated quantum routing preserves learning

## Model Quick Reference

| Model | Complexity | Sparsity | Location |
|-------|-----------|----------|----------|
| **DCM-MSR** | O(n×k×w) | 1.56-12.5% | `models/dcm_msr/` |
| **BiFormer** | O(n) | ~10-20% | `models/baselines/biformer_adapter.py` |
| **BigBird** | O(n) | 12-81% | `models/baselines/bigbird_faithful.py` |
| **Standard** | O(n²) | 100% | PyTorch `nn.MultiheadAttention` |

See `README.md` for full documentation.

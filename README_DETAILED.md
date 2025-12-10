# DCM-MSR Experimental Framework

This folder contains all the code used to run experiments for the DCM-MSR (Dynamic Coarse Masking with Multi-Scale Routing) research project.

## Folder Structure

```
experiments/
├── models/                      # Model implementations
│   ├── dcm_msr/                # DCM-MSR quantum-inspired attention
│   │   ├── attention.py        # Main DCM-MSR attention mechanism
│   │   ├── quantum_utils.py    # Quantum ensemble operations (density matrix, fidelity)
│   │   ├── windowing.py        # Window-based key/value selection
│   │   └── __init__.py
│   └── baselines/              # Baseline attention mechanisms
│       ├── biformer_adapter.py # BiFormer (Bi-Level Routing Attention)
│       └── bigbird_faithful.py # BigBird (Sparse block attention)
│
├── data_synthesis/             # Data generation and loading
│   ├── create_sample_data.py  # Synthetic LRA data generation
│   └── lra_data_loader.py     # Data loading utilities
│
├── training_scripts/           # Main training/comparison scripts
│   ├── compare_single_model.py        # Single model runner (Job 7860145)
│   ├── lra_full_benchmark_fixed.py    # Full LRA benchmark suite
│   └── compare_dcm_biformer.py        # Unified comparison framework
│
├── test_scripts/               # Local validation tests
│   ├── test_comparison.py          # Pre-deployment validation
│   ├── test_synthetic_data.py      # Data generation tests
│   └── test_comprehensive_eval.py  # Comprehensive evaluation tests
│
└── palmetto_batch_scripts/     # Palmetto2 HPC batch scripts
    ├── palmetto_compare_biformer.sh   # Job 7860145 (Standard + DCM-MSR)
    ├── palmetto_biformer_test.sh      # BiFormer-only test
    ├── palmetto_eval_batch.sh         # Comprehensive evaluation (Job 7823737)
    ├── palmetto_lra_accuracy.sh       # LRA accuracy benchmark
    └── run_lra_palmetto.sh            # Basic LRA runner
```

## Key Experiments

### 1. Job 7823737 (Dec 8) - Comprehensive Evaluation
**Script**: `palmetto_eval_batch.sh`
- **Compared**: DCM-MSR vs BigBird vs Standard
- **Metrics**: Sparsity, FLOPs, wall-clock time, quantum ensemble metrics
- **Results**: DCM-MSR 2-4x faster than BigBird at 1024-4096 tokens, 6-8x sparser

### 2. Job 7860145 (Dec 9) - Accuracy Validation
**Script**: `palmetto_compare_biformer.sh` → `compare_single_model.py`
- **Compared**: DCM-MSR vs Standard on synthetic LRA tasks
- **Tasks**: ListOps (seq=1024), Text (seq=2048)
- **Results**: DCM-MSR within 2-3% accuracy, 2.4x slower (classical overhead)

## Model Descriptions

### DCM-MSR (Dynamic Coarse Masking with Multi-Scale Routing)
**Location**: `models/dcm_msr/`

Quantum-inspired attention mechanism with:
- Per-query window routing via quantum fidelity: Tr(ρ_q × ρ_k)
- Density matrix construction from Q/K embeddings
- Top-k window selection (k=2, window_size=32)
- O(n × k × w) complexity vs O(n²) standard attention

**Key Components**:
- `attention.py`: Main attention layer with fidelity-based routing
- `quantum_utils.py`: Density matrix creation, fidelity computation
- `windowing.py`: Key/value windowing and selection

### BiFormer (Baseline)
**Location**: `models/baselines/biformer_adapter.py`

Bi-Level Routing Attention from CVPR 2023:
- Region-to-region routing (coarse)
- Token-to-token attention (fine)
- Detached gradients (non-parametric)

### BigBird (Baseline)
**Location**: `models/baselines/bigbird_faithful.py`

Sparse block attention with:
- Random blocks
- Global tokens
- Sliding window
- Fixed sparsity pattern

## Data Synthesis

**Script**: `data_synthesis/create_sample_data.py`

Generates synthetic LRA-style datasets:
- **ListOps**: Nested bracket operations, 10-class classification
- **Text**: Document classification with sentiment patterns, binary
- **Retrieval**: Sequence matching with shared/unshared patterns, binary

**Configuration**:
- Balanced class distribution
- Configurable sequence lengths (512-4096)
- Train/test splits (500-1600 train, 100-300 test)

## Running Experiments

### Local Testing
```bash
# Validate before deployment
python test_scripts/test_comparison.py

# Generate synthetic data
python data_synthesis/create_sample_data.py --output data/
```

### Single Model Training
```bash
# Run DCM-MSR on ListOps
python training_scripts/compare_single_model.py \
    --task listops \
    --attention-type dcm_msr \
    --data-dir data/ \
    --device cuda
```

### Palmetto2 Batch Jobs
```bash
# Submit comparison job
qsub palmetto_batch_scripts/palmetto_compare_biformer.sh

# Monitor output
tail -f compare_biformer_<JOB_ID>.out
```

## Configuration Details

### Model Hyperparameters
- Embed dim: 64
- Num heads: 4
- Num layers: 2
- Dropout: 0.1

### DCM-MSR Specific
- Window size: 32
- Top-k: 2
- Head dim: 16 (64/4)

### Training
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10
- Batch size: 1-2 (memory constrained)

## Results Summary

### Job 7823737 (vs BigBird)
- **Speed**: DCM-MSR 2-4x faster at 1024-4096 tokens
- **Sparsity**: 6-8x sparser (1.56% at 4096 vs 12.21%)
- **FLOPs Paradox**: More FLOPs but faster (better memory patterns)

### Job 7860145 (vs Standard)
- **Accuracy**: Within 2-3% (ListOps: -3%, Text: -2%)
- **Speed**: 2.4x slower (classical simulation overhead)
- **Validation**: Quantum routing works, preserves learning

## Dependencies

```bash
# Core
torch>=2.0.0
numpy>=1.20.0

# Data processing
tqdm
matplotlib
seaborn (optional)

# Testing
pytest
```

## Citation

If you use this experimental framework, please cite:

```bibtex
@article{holcomb2025dcmmsr,
  title={DCM-MSR: Dynamic Coarse Masking with Multi-Scale Routing for Efficient Long Sequence Attention},
  author={Holcomb, Landon},
  year={2025}
}
```

## Contact

For questions about the experimental setup:
- GitHub: [LandonHolcomb/Dynamic-Coarse-masking](https://github.com/LandonHolcomb/Dynamic-Coarse-masking)

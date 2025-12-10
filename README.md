# DCM-MSR Experimental Code

Experimental framework for **Dynamic Coarse Masking with Multi-Scale Routing (DCM-MSR)** - a quantum-inspired attention mechanism for efficient long sequence processing.

## Quick Links

- 📖 [Full Documentation](README.md)
- 🚀 [Quick Start Guide](QUICKSTART.md)
- 📋 [Complete File Index](FILE_INDEX.md)

## Overview

This repository contains all experimental code used in DCM-MSR research:
- ✅ DCM-MSR quantum-inspired attention implementation
- ✅ Baseline models (BiFormer, BigBird)
- ✅ Synthetic LRA data generation
- ✅ Training and evaluation scripts
- ✅ HPC batch job scripts (Palmetto2)

## Key Results

### Job 7823737 (Dec 8, 2025) - vs BigBird
- **2-4x faster** at 1024-4096 tokens
- **6-8x sparser** attention (1.56% at 4096 tokens)
- Better memory patterns despite more FLOPs

### Job 7860145 (Dec 9, 2025) - vs Standard Attention
- **Within 2-3% accuracy** on synthetic LRA tasks
- **2.4x slower** (classical simulation overhead)
- Validates quantum routing preserves learning

## Quick Start

```bash
# Generate synthetic data
cd data_synthesis
python create_sample_data.py --output ../data/

# Run local tests
cd test_scripts
python test_comparison.py

# Train single model
cd training_scripts
python compare_single_model.py \
    --task listops \
    --attention-type dcm_msr \
    --data-dir ../data/ \
    --device cuda
```

## Repository Structure

```
experiments/
├── models/                 # DCM-MSR + baseline implementations
├── data_synthesis/         # Synthetic LRA data generation
├── training_scripts/       # Training and comparison runners
├── test_scripts/          # Local validation tests
└── palmetto_batch_scripts/ # HPC batch job scripts
```

## Installation

```bash
pip install torch numpy tqdm matplotlib
```

## Citation

If you use this code, please cite:

```bibtex
@article{holcomb2025dcmmsr,
  title={DCM-MSR: Dynamic Coarse Masking with Multi-Scale Routing for Efficient Long Sequence Attention},
  author={Holcomb, Landon},
  year={2025}
}
```

## Main Repository

This is the experimental code repository. For the full project including analysis and documentation:
- [LandonHolcomb/Dynamic-Coarse-masking](https://github.com/LandonHolcomb/Dynamic-Coarse-masking)

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contact

- GitHub: [@LandonHolcomb](https://github.com/LandonHolcomb)
- Repository: [Dynamic-Coarse-masking-experiments](https://github.com/LandonHolcomb/Dynamic-Coarse-masking-experiments)

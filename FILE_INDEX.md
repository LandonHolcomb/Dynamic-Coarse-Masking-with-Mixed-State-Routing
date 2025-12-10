# Experiments Folder - File Index

**Total Files**: 21 (organized into 6 categories)

---

## Models (6 files)

### DCM-MSR Implementation (4 files)
- `models/dcm_msr/__init__.py` - Package initialization
- `models/dcm_msr/attention.py` - Main DCM-MSR attention mechanism
- `models/dcm_msr/quantum_utils.py` - Quantum ensemble operations (density matrix, fidelity)
- `models/dcm_msr/windowing.py` - Window-based key/value selection

### Baseline Implementations (2 files)
- `models/baselines/biformer_adapter.py` - BiFormer (Bi-Level Routing Attention, CVPR 2023)
- `models/baselines/bigbird_faithful.py` - BigBird (Sparse block attention)

---

## Data Synthesis (2 files)

- `data_synthesis/create_sample_data.py` - Synthetic LRA data generation (ListOps, Text, Retrieval)
- `data_synthesis/lra_data_loader.py` - Data loading utilities for LRA tasks

---

## Training Scripts (3 files)

- `training_scripts/compare_single_model.py` - **Primary training script** (used in Job 7860145)
  - Command-line interface for task/attention-type selection
  - Single model runner for accurate timing
  - Saves JSON results with accuracy, timing, FLOPs

- `training_scripts/lra_full_benchmark_fixed.py` - Full LRA benchmark suite
  - Runs all 3 tasks (ListOps, Text, Retrieval)
  - Tests multiple attention mechanisms
  - Comprehensive evaluation framework

- `training_scripts/compare_dcm_biformer.py` - Unified comparison framework
  - `UnifiedTransformer` class with pluggable attention
  - Handles different attention interfaces
  - Used for early development

---

## Test Scripts (3 files)

- `test_scripts/test_comparison.py` - **Pre-deployment validation**
  - 7 comprehensive tests
  - Validates DCM-MSR, BiFormer, Standard attention
  - Checks dimensions, gradients, routing diversity

- `test_scripts/test_synthetic_data.py` - Data generation tests
  - Validates synthetic LRA data format
  - Checks class balance and sequence lengths

- `test_scripts/test_comprehensive_eval.py` - Comprehensive evaluation tests
  - Tests full evaluation pipeline
  - Validates metrics collection

---

## Palmetto Batch Scripts (5 files)

### Successful Jobs
- `palmetto_batch_scripts/palmetto_compare_biformer.sh` - **Job 7860145** (Dec 9)
  - Standard + DCM-MSR comparison on ListOps & Text
  - 2 GPUs (V100-32GB), 128GB RAM, 15 hours
  - Results: DCM-MSR within 2-3% accuracy, 2.4x slower

- `palmetto_batch_scripts/palmetto_eval_batch.sh` - **Job 7823737** (Dec 8)
  - DCM-MSR vs BigBird comprehensive evaluation
  - 2 GPUs (V100-16GB), measured sparsity, FLOPs, timing
  - Results: DCM-MSR 2-4x faster than BigBird, 6-8x sparser

### Other Scripts
- `palmetto_batch_scripts/palmetto_biformer_test.sh` - BiFormer-only test (hung during training)
- `palmetto_batch_scripts/palmetto_lra_accuracy.sh` - LRA accuracy benchmark (3 GPUs)
- `palmetto_batch_scripts/run_lra_palmetto.sh` - Basic LRA runner

---

## Documentation (2 files)

- `README.md` - Full documentation
  - Folder structure explanation
  - Key experiments description
  - Model descriptions
  - Configuration details
  - Results summary

- `QUICKSTART.md` - Quick reference guide
  - Essential commands
  - File dependencies
  - Results summary table
  - Model comparison

---

## Key Usage Patterns

### Local Development Workflow
1. `test_scripts/test_comparison.py` - Validate implementations
2. `data_synthesis/create_sample_data.py` - Generate test data
3. `training_scripts/compare_single_model.py` - Train locally

### Palmetto Deployment Workflow
1. Prepare batch script (`palmetto_batch_scripts/`)
2. Upload code and data
3. Submit job: `qsub <script>.sh`
4. Monitor: `tail -f <job_id>.out`

### Results Analysis
- Training scripts save JSON files: `results_<task>_<model>.json`
- Contains: accuracy, timing, parameters, configuration
- Load and compare results for paper

---

## Import Dependencies

All training/test scripts expect:
```python
# Relative imports from project root
from dcm_msr.attention import DCMMSRAttention
from analysis.biformer_adapter import BiLevelRoutingAttention1D
from analysis.bigbird_faithful import BigBirdAttention
```

If running standalone, adjust paths:
```python
import sys
sys.path.insert(0, '../models')
from dcm_msr.attention import DCMMSRAttention
```

---

## File Sizes (Approximate)

| Category | Total Lines | Total Size |
|----------|-------------|------------|
| Models | ~1,500 LOC | ~60 KB |
| Data Synthesis | ~600 LOC | ~25 KB |
| Training Scripts | ~2,000 LOC | ~80 KB |
| Test Scripts | ~800 LOC | ~30 KB |
| Batch Scripts | ~300 LOC | ~12 KB |
| **Total** | **~5,200 LOC** | **~207 KB** |

---

## Version History

- **v1.0** (Dec 9, 2025): Initial organization
  - Extracted experimental code from main project
  - Organized into logical categories
  - Added comprehensive documentation

---

For detailed usage instructions, see `README.md` or `QUICKSTART.md`.

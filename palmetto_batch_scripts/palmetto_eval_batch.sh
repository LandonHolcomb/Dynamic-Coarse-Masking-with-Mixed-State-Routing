#!/bin/bash
#SBATCH --job-name=dcm_msr_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus v100:2
#SBATCH --time=12:00:00
#SBATCH --output=dcm_msr_eval_%j.out
#SBATCH --error=dcm_msr_eval_%j.err

# DCM-MSR Comprehensive Evaluation on Palmetto2

module purge
module load anaconda3
module load cuda

# Activate existing environment (should already exist)
source activate dcm_msr

# Change to project directory first
cd "$SLURM_SUBMIT_DIR"

# Set Python path to find modules
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "======================================================================"
echo "DCM-MSR Comprehensive Evaluation"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $PWD"
echo ""

# Check GPU availability
echo "--- GPU Information ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ----------------------------------------------------------------------
# 1. Mask Comparison Test (GPU 0)  -- run first
# ----------------------------------------------------------------------
echo "======================================================================"
echo "1. MASK COMPARISON TEST on GPU 0"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=0 python analysis/isolated_mask_comparison.py \
    --device cuda \
    --seq-lens 512 1024 2048 4096
echo ""

# ----------------------------------------------------------------------
# 2. Comprehensive Evaluation (GPU 0)  -- run in background
# ----------------------------------------------------------------------
echo "======================================================================"
echo "2. COMPREHENSIVE EVALUATION (FLOPs + Ensemble Metrics) on GPU 0"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=0 python evaluation/comprehensive_eval.py \
    --device cuda \
    --seq-lens 512 1024 2048 4096 \
    --output comprehensive_eval_${SLURM_JOB_ID}.json &
PID_COMP=$!

# ----------------------------------------------------------------------
# 3. Ablation Study (GPU 1)  -- run in background
# ----------------------------------------------------------------------
echo "======================================================================"
echo "3. ABLATION STUDY (window_size and top_k) on GPU 1"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=1 python evaluation/ablation_study.py \
    --device cuda \
    --window-sizes 16 32 64 128 \
    --top-k-values 1 2 4 8 \
    --seq-lens 512 1024 2048 4096 \
    --trials 10 \
    --output ablation_study_${SLURM_JOB_ID}.json &
PID_ABL=$!

# Wait for both GPU jobs to finish
wait $PID_COMP
wait $PID_ABL

# ----------------------------------------------------------------------
# 4. Full LRA Benchmark (run after the above finish)
# ----------------------------------------------------------------------
echo "======================================================================"
echo "4. FULL LRA BENCHMARK (with real ListOps data) on GPU 0"
echo "======================================================================"
if [ -d "/scratch/$USER/lra/lra_release" ]; then
    echo "Found LRA data, running with real data..."
    CUDA_VISIBLE_DEVICES=0 python analysis/lra_full_benchmark_fixed.py \
        --use-real-data \
        --data-dir /scratch/$USER/lra/lra_release \
        --device cuda
else
    echo "LRA data not found, running with synthetic data..."
    CUDA_VISIBLE_DEVICES=0 python analysis/lra_full_benchmark_fixed.py --device cuda
fi
echo ""

# ----------------------------------------------------------------------
# Results summary
# ----------------------------------------------------------------------
echo "======================================================================"
echo "RESULTS SUMMARY"
echo "======================================================================"
echo "Output files generated:"
ls -lh *_${SLURM_JOB_ID}.* 2>/dev/null || echo "  (no job-specific files)"
ls -lh analysis/lra_benchmark_results*.json 2>/dev/null || echo "  (no LRA benchmark)"
echo ""

echo "End time: $(date)"
echo "======================================================================"

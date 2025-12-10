#!/bin/bash
#SBATCH --job-name=dcm_lra_accuracy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gpus v100:3
#SBATCH --time=24:00:00
#SBATCH --output=lra_accuracy_%j.out
#SBATCH --error=lra_accuracy_%j.err

# DCM-MSR LRA Accuracy Benchmark on Palmetto2
# Tests Standard, BigBird, and DCM-MSR attention mechanisms on LRA tasks
# Runs 3 tasks in parallel on 3 GPUs

module purge
module load anaconda3
module load cuda

# Change to project directory first
cd "$SLURM_SUBMIT_DIR"

# Check if environment exists, create if not
if ! conda env list | grep -q "^dcm_msr "; then
    echo "Creating dcm_msr environment..."
    conda create -n dcm_msr python=3.10 -y
fi

# Activate environment
source activate dcm_msr

# Install/upgrade packages if needed
echo "Checking dependencies..."
pip install --quiet --upgrade "numpy<2" torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Set Python path
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"

echo "======================================================================"
echo "DCM-MSR LRA Accuracy Benchmark"
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

# Using synthetic data for controlled benchmarking
echo "Using synthetic LRA data for controlled comparison"
echo ""

# ----------------------------------------------------------------------
# Run 3 LRA tasks in parallel on separate GPUs
# ----------------------------------------------------------------------

echo "======================================================================"
echo "Starting LRA Tasks in Parallel"
echo "======================================================================"
echo "GPU 0: ListOps (seq_len=2048)"
echo "GPU 1: Text Classification (seq_len=4096)"
echo "GPU 2: Retrieval (seq_len=4096)"
echo ""

# GPU 0: ListOps
echo "--- Starting ListOps on GPU 0 ---"
CUDA_VISIBLE_DEVICES=0 python analysis/lra_full_benchmark_fixed.py \
    --tasks listops \
    --device cuda \
    --output lra_listops_${SLURM_JOB_ID}.json \
    2>&1 | tee lra_listops_${SLURM_JOB_ID}.log &
PID_LISTOPS=$!

# GPU 1: Text Classification
echo "--- Starting Text Classification on GPU 1 ---"
CUDA_VISIBLE_DEVICES=1 python analysis/lra_full_benchmark_fixed.py \
    --use-real-data \
    --data-dir /scratchpython analysis/lra_full_benchmark_fixed.py \
    --tasks text \
    --device cuda \
    --output lra_text_${SLURM_JOB_ID}.json \
    2>&1 | tee lra_text_${SLURM_JOB_ID}.log &
# GPU 2: Retrieval
echo "--- Starting Retrieval on GPU 2 ---"
CUDA_VISIBLE_DEVICES=2 python analysis/lra_full_benchmark_fixed.py \
    --use-real-data \
    --data-dir /scratchpython analysis/lra_full_benchmark_fixed.py \
    --tasks retrieval \
    --device cuda \
    --output lra_retrieval_${SLURM_JOB_ID}.json \
    2>&1 | tee lra_retrieval_${SLURM_JOB_ID}.log &
echo ""
echo "All tasks started. Waiting for completion..."
echo "ListOps PID: $PID_LISTOPS"
echo "Text PID: $PID_TEXT"
echo "Retrieval PID: $PID_RETRIEVAL"
echo ""

# Wait for all tasks to complete
wait $PID_LISTOPS
STATUS_LISTOPS=$?
echo "ListOps completed with status: $STATUS_LISTOPS"

wait $PID_TEXT
STATUS_TEXT=$?
echo "Text completed with status: $STATUS_TEXT"

wait $PID_RETRIEVAL
STATUS_RETRIEVAL=$?
echo "Retrieval completed with status: $STATUS_RETRIEVAL"

echo ""
echo "======================================================================"
echo "All Tasks Completed"
echo "======================================================================"

# ----------------------------------------------------------------------
# Combine results
# ----------------------------------------------------------------------
echo ""
echo "--- Results Summary ---"
echo ""

if [ -f "lra_listops_${SLURM_JOB_ID}.json" ]; then
    echo "ListOps results:"
    cat lra_listops_${SLURM_JOB_ID}.json | python -m json.tool 2>/dev/null || cat lra_listops_${SLURM_JOB_ID}.json
    echo ""
fi

if [ -f "lra_text_${SLURM_JOB_ID}.json" ]; then
    echo "Text Classification results:"
    cat lra_text_${SLURM_JOB_ID}.json | python -m json.tool 2>/dev/null || cat lra_text_${SLURM_JOB_ID}.json
    echo ""
fi

if [ -f "lra_retrieval_${SLURM_JOB_ID}.json" ]; then
    echo "Retrieval results:"
    cat lra_retrieval_${SLURM_JOB_ID}.json | python -m json.tool 2>/dev/null || cat lra_retrieval_${SLURM_JOB_ID}.json
    echo ""
fi

echo "======================================================================"
echo "Output Files Generated:"
echo "======================================================================"
ls -lh lra_*_${SLURM_JOB_ID}.* 2>/dev/null || echo "No output files found"
echo ""

echo "End time: $(date)"
echo "======================================================================"

# Exit with error if any task failed
if [ $STATUS_LISTOPS -ne 0 ] || [ $STATUS_TEXT -ne 0 ] || [ $STATUS_RETRIEVAL -ne 0 ]; then
    echo "ERROR: One or more tasks failed"
    exit 1
fi

echo "✅ All tasks completed successfully!"

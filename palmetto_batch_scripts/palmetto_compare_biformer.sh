#!/bin/bash
#SBATCH --job-name=biformer_dcm_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --gpus v100:2
#SBATCH --output=logs/compare_biformer_%j.out
#SBATCH --error=logs/compare_biformer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lholco2@clemson.edu

# BiFormer vs DCM-MSR Comparison on Synthetic LRA Data
# Parallel execution on 2 GPUs for timing comparison

module purge
module load anaconda3
module load cuda

# Use base conda environment (avoid environment issues)
# Install matplotlib if not present (needed for summarize_comparison.py)
pip install --user matplotlib >/dev/null 2>&1 || true

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=================================================================================================="
echo "DCM-MSR vs BiFormer Head-to-Head Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=================================================================================================="

# Set CUDA devices for parallel execution
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Test on all three LRA tasks
TASKS=("ListOps" "Text" "Retrieval")

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "=================================================================================================="
    echo "Running comparison on $TASK task"
    echo "=================================================================================================="
    
    # Determine task-specific parameters
    if [ "$TASK" = "ListOps" ]; then
        SEQ_LEN=1024
        VOCAB_SIZE=256
        NUM_CLASSES=10
        NUM_TRAIN=500
        NUM_TEST=100
        BATCH_SIZE=2
        EMBED_DIM=64
        NUM_LAYERS=2
    else
        SEQ_LEN=2048
        VOCAB_SIZE=256
        NUM_CLASSES=2
        NUM_TRAIN=800
        NUM_TEST=150
        BATCH_SIZE=1
        EMBED_DIM=64
        NUM_LAYERS=2
    fi
    
    echo "Task parameters: seq_len=$SEQ_LEN, vocab=$VOCAB_SIZE, classes=$NUM_CLASSES"
    echo "Dataset: train=$NUM_TRAIN, test=$NUM_TEST, batch=$BATCH_SIZE"
    echo ""
    
    # Run Standard attention first (baseline, single GPU)
    echo "Running Standard attention (baseline)..."
    CUDA_VISIBLE_DEVICES=0 python -u analysis/compare_single_model.py \
        --task "$TASK" \
        --seq-len $SEQ_LEN \
        --vocab-size $VOCAB_SIZE \
        --num-classes $NUM_CLASSES \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --embed-dim $EMBED_DIM \
        --num-layers $NUM_LAYERS \
        --attention-type standard \
        --output-prefix "results_${TASK,,}_standard" \
        2>&1 | tee "logs/standard_${TASK,,}_${SLURM_JOB_ID}.log"
    
    echo ""
    echo "Running BiFormer and DCM-MSR in parallel..."
    
    # Run BiFormer and DCM-MSR in parallel on separate GPUs
    CUDA_VISIBLE_DEVICES=0 python -u analysis/compare_single_model.py \
        --task "$TASK" \
        --seq-len $SEQ_LEN \
        --vocab-size $VOCAB_SIZE \
        --num-classes $NUM_CLASSES \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --embed-dim $EMBED_DIM \
        --num-layers $NUM_LAYERS \
        --attention-type biformer \
        --output-prefix "results_${TASK,,}_biformer" \
        2>&1 | tee "logs/biformer_${TASK,,}_${SLURM_JOB_ID}.log" &
    BIFORMER_PID=$!
    
    CUDA_VISIBLE_DEVICES=1 python -u analysis/compare_single_model.py \
        --task "$TASK" \
        --seq-len $SEQ_LEN \
        --vocab-size $VOCAB_SIZE \
        --num-classes $NUM_CLASSES \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --embed-dim $EMBED_DIM \
        --num-layers $NUM_LAYERS \
        --attention-type dcm_msr \
        --output-prefix "results_${TASK,,}_dcmmsr" \
        2>&1 | tee "logs/dcmmsr_${TASK,,}_${SLURM_JOB_ID}.log" &
    DCMMSR_PID=$!
    
    # Wait for both to complete
    echo "Waiting for BiFormer (PID $BIFORMER_PID) and DCM-MSR (PID $DCMMSR_PID)..."
    wait $BIFORMER_PID
    BIFORMER_STATUS=$?
    wait $DCMMSR_PID
    DCMMSR_STATUS=$?
    
    if [ $BIFORMER_STATUS -eq 0 ]; then
        echo "✓ BiFormer completed successfully"
    else
        echo "✗ BiFormer failed with status $BIFORMER_STATUS"
    fi
    
    if [ $DCMMSR_STATUS -eq 0 ]; then
        echo "✓ DCM-MSR completed successfully"
    else
        echo "✗ DCM-MSR failed with status $DCMMSR_STATUS"
    fi
    
    echo ""
    echo "Task $TASK complete"
    echo "=================================================================================================="
done

echo ""
echo "=================================================================================================="
echo "All comparisons complete!"
echo "Results saved in current directory as results_*_*.json"
echo "Visualizations saved as results_*_attention.png"
echo "=================================================================================================="

# Generate summary comparison
echo ""
echo "Generating summary comparison..."
python -u analysis/summarize_comparison.py 2>&1 | tee "logs/summary_${SLURM_JOB_ID}.log"

echo ""
echo "Job complete: $(date)"

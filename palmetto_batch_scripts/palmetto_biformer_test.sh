#!/bin/bash
#SBATCH --job-name=biformer_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gpus v100:1
#SBATCH --output=logs/biformer_test_%j.out
#SBATCH --error=logs/biformer_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lholco2@clemson.edu

# Test BiFormer only on one task with minimal resources

module purge
module load anaconda3
module load cuda

# Install matplotlib if not present
pip install --user matplotlib >/dev/null 2>&1 || true

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=================================================================================================="
echo "BiFormer Test Run (Single GPU, Minimal Resources)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================================================================="

# Test on ListOps only (smallest task)
TASK="ListOps"
SEQ_LEN=1024
VOCAB_SIZE=256
NUM_CLASSES=10
NUM_TRAIN=500
NUM_TEST=100
BATCH_SIZE=2
EMBED_DIM=64
NUM_LAYERS=2

echo ""
echo "Testing BiFormer on $TASK task"
echo "Parameters: seq_len=$SEQ_LEN, batch=$BATCH_SIZE, embed=$EMBED_DIM, layers=$NUM_LAYERS"
echo ""

python -u analysis/compare_single_model.py \
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
    --output-prefix "results_${TASK,,}_biformer_test" \
    2>&1 | tee "logs/biformer_${TASK,,}_test_${SLURM_JOB_ID}.log"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ BiFormer test completed successfully!"
    echo "Results saved to results_listops_biformer_test.json"
else
    echo ""
    echo "✗ BiFormer test failed with exit code $?"
fi

echo ""
echo "Job complete: $(date)"

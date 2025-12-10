#!/bin/bash
#SBATCH --job-name=lra_benchmark
#SBATCH --output=lra_benchmark_%j.out
#SBATCH --error=lra_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

# Load modules
module load anaconda3/2023.03
module load cuda/11.8

# Activate environment (create if needed)
source activate dcm_msr || conda create -n dcm_msr python=3.10 -y && source activate dcm_msr

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run benchmark on GPU
echo "Starting LRA Benchmark on GPU node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run with synthetic data (default)
# To use real LRA data, add: --use-real-data --data-dir /path/to/lra_data
python analysis/lra_full_benchmark_fixed.py --device cuda

echo "Benchmark complete!"
echo "Results saved to analysis/lra_full_benchmark_results.json"

# Optional: To run with real data (if you have it):
# python analysis/lra_full_benchmark_fixed.py --device cuda --use-real-data --data-dir ./lra_data

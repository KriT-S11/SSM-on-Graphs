#!/bin/bash
#SBATCH --job-name=Eval_GraphRAG
#SBATCH --partition=gpu-long
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# 1. Initialize and activate your exact conda environment
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate graphssm_krit

# 2. Navigate to your project directory
cd /nfs_home/users/goel/rs2/process_modeling/llama2/FinalMovedData_RP/VideoMomentRetrivel

# 3. Run the evaluation script
echo "Starting evaluation..."
python s6.py
echo "Evaluation complete!"

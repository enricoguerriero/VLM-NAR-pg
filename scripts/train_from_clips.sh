#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=train_from_clips
#SBATCH --output=outputs/train_from_clips.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=./

python src/training/train_from_clips.py \
    --model_name "LLaVANeXT"

echo "--- THE END ---"
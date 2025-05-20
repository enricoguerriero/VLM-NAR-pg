#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu64
#SBATCH --time=48:00:00
#SBATCH --job-name=export_tokens
#SBATCH --output=outputs/export_tokens.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

python src/data/export_tokens.py \
    --model_name "VideoLLaVA"

echo "--- THE END ---"

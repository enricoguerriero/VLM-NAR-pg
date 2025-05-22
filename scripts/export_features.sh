#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=export_features
#SBATCH --output=outputs/export_features.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTHONPATH=./
python src/data/export_features.py \
    --model_name "VideoLLaVA"

echo "--- THE END ---"

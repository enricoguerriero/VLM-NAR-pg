#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=optimization_hyperparameters
#SBATCH --output=outputs/hyperparameters_optimization.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTHONPATH=./
python src/training/hyperparameter_optimization.py \
    --model_name "VideoLLaVA"

echo "--- THE END ---"

#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=test_0s_binary
#SBATCH --output=outputs/test_0s_binary.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

set -a
source .env
set +a

echo "Environment variables loaded"

export HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN

export PYTHONPATH=./
python src/evaluation/binary_zero-shot.py \
    --model_name "VideoLLaVA"

echo "--- THE END ---"

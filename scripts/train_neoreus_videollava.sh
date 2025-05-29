#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=train_videollava
#SBATCH --output=outputs/train_videollava.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=./

python src/training/train_neoresus_videollava.py   \
    --train_json data/clips/train.jsonl   \
    --valid_json data/clips/validation.jsonl   \
    --output_dir checkpoints/neoresus_lora   \
    --batch_per_gpu 1   \
    --grad_accum 4   \
    --epochs 3   \
    --lr 1e-4

echo "--- THE END ---"

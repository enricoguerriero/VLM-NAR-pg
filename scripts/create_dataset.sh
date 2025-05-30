#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu64
#SBATCH --time=48:00:00
#SBATCH --job-name=create_dataset
#SBATCH --output=outputs/create_dataset.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTHONPATH=./
python src/data/balance_jsonl.py \
    --input data/clips/test.jsonl \
    --output data/clips/test_balanced.jsonl 

echo "--- THE END ---"

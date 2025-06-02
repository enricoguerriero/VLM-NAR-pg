#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=test_llavanext_multiclass
#SBATCH --output=outputs/test_llavanext_multiclass.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=./

python src/evaluation/test_llavanext_judge.py \
        --jsonl data/clips/test_balanced.jsonl \
        --model llava-hf/LLaVA-NeXT-Video-7B-hf \
        --num-frames 16
        
echo "--- THE END ---"

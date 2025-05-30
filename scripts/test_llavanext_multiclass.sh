#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=test_llavanext
#SBATCH --output=outputs/test_llavanext.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=./

python src/evaluation/test_llavanext_per_class.py \
        --jsonl data/clips/test.jsonl \
        --model_name llava-hf/LLaVA-NeXT-Video-7B-hf \
        --prompts prompts.json 
echo "--- THE END ---"

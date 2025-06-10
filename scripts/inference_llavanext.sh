#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=test_llavanext_lora
#SBATCH --output=outputs/test_llavanext_lora.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=./

python src/evaluation/inference_llavanext.py \
  --test_json data/videos/test/20240207_141009_part_1_of_1.mp4 \
  --model_path outputs/llavanext_lora/model_final.pt \
  --model_id llava-hf/LLaVA-NeXT-Video-7B-hf \
  --save_csv outputs/inference_probabilities.csv \
  --num_frames 16 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 

echo "--- THE END ---"
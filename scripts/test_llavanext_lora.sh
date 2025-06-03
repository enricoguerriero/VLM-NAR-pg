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

python test_llava_video_classifier.py \
  --test_json path/to/test_data.jsonl \
  --model_path outputs/model_final.pt \
  --model_id llava-hf/llava-next-large-430k \
  --batch_size 2 \
  --num_frames 8 \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --output_metrics results/test_metrics.json

echo "--- THE END ---"
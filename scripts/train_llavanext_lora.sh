#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=train_llavanext_lora
#SBATCH --output=outputs/train_llavanext_lora.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=./

python src/training/train_llavanext_lora.py \
    --data_dir data/clips \
    --model_name "llava-hf/LLaVA-NeXT-Video-7B-hf" \
    --output_dir models/llavanext_lora \
    --wandb_project llavanext_lora \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --lr 5e-5 \
    --num_frames 16 \
    --fp16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 

echo "--- THE END ---"

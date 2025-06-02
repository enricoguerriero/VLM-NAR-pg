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
    --train_json data/clips/train.jsonl \
    --val_json data/clips/validation.jsonl \
    --test_json data/clips/test.jsonl \
    --model_id llava-hf/LLaVA-NeXT-Video-7B-hf \
    --output_dir outputs/llavanext_lora \
    --batch_size 1 \
    --epochs 2 \
    --lr 5e-5 \
    --num_frames 16 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --wandb_project llavanext_lora \
    --run_name llavanext_lora_run 


echo "--- THE END ---"

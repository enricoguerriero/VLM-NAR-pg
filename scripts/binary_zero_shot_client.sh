#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --job-name=client_test_0s_binary
#SBATCH --output=outputs/client_test_0s_binary.out

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
python src/evaluation/binary_zero_shot_client.py \
    --input_file "VideoLLaVA_stream.ndjson" \
    --output_file "VideoLLaVA_stream_binary_results.ndjson" \
    --poll_interval 1 


echo "--- THE END ---"

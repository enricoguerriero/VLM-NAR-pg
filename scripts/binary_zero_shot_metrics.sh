#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=48:00:00
#SBATCH --job-name=producer_test_0s_binary
#SBATCH --output=outputs/producer_test_0s_binary.out

# Activate the user environment (uenv)
uenv verbose cuda-12.2.0 cudnn-12.x-8.8.0
uenv miniconda3-py39

# Activate the Conda environment
source activate NewbornEnv

echo "virtual environment activated"

export PYTHONPATH=./
python src/evaluation/binary_zero_shot_metrics.py \
    --model_name "VideoLLaVA" \
    --prediction_file "VideoLLaVA_stream_binary_results.ndjson" \

echo "--- THE END ---"

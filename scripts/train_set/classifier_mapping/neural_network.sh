#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

# Use values from config.env or defaults
model=${MODEL:-google-bert/bert-base-cased}
layer=${LAYER:-12}

# Neural Network specific parameters
num_epochs=${NN_EPOCHS:-50}
batch_size=${NN_BATCH_SIZE:-128}
learning_rate=${NN_LEARNING_RATE:-0.001}
hidden_dims=${NN_HIDDEN_DIMS:-512,256,128}
dropout=${NN_DROPOUT:-0.3}
device=${NN_DEVICE:-cuda}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

scriptDir=$PROJECT_ROOT/src/concept_mapper
fileDir="$PROJECT_ROOT/eraser_movie/$model_file_name/split_dataset"
savePath="$PROJECT_ROOT/eraser_movie/$model_file_name/result_nn"

mkdir -p ${savePath}

python ${scriptDir}/neural_network.py \
    --train_file_path ${fileDir}/train/train_df_${layer}.csv \
    --validate_file_path ${fileDir}/validation/validation_df_${layer}.csv \
    --layer ${layer} \
    --save_path ${savePath} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --hidden_dims ${hidden_dims} \
    --dropout ${dropout} \
    --device ${device} \
    --do_train \
    --do_validate

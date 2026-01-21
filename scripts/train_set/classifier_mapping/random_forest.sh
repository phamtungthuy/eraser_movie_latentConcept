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

# Random Forest specific parameters
n_estimators=${RF_N_ESTIMATORS:-200}
max_depth=${RF_MAX_DEPTH:-None}
min_samples_split=${RF_MIN_SAMPLES_SPLIT:-5}
min_samples_leaf=${RF_MIN_SAMPLES_LEAF:-2}
max_features=${RF_MAX_FEATURES:-sqrt}
n_jobs=${RF_N_JOBS:--1}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

scriptDir=$PROJECT_ROOT/src/concept_mapper
fileDir="$PROJECT_ROOT/eraser_movie/$model_file_name/split_dataset"
savePath="$PROJECT_ROOT/eraser_movie/$model_file_name/result_rf"

mkdir -p ${savePath}

# Build command with optional max_depth
cmd="python ${scriptDir}/random_forest.py \
    --train_file_path ${fileDir}/train/train_df_${layer}.csv \
    --validate_file_path ${fileDir}/validation/validation_df_${layer}.csv \
    --layer ${layer} \
    --save_path ${savePath} \
    --n_estimators ${n_estimators} \
    --min_samples_split ${min_samples_split} \
    --min_samples_leaf ${min_samples_leaf} \
    --max_features ${max_features} \
    --n_jobs ${n_jobs} \
    --do_train \
    --do_validate"

# Add max_depth only if it's not "None"
if [ "$max_depth" != "None" ]; then
    cmd="$cmd --max_depth ${max_depth}"
fi

eval $cmd

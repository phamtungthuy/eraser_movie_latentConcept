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

# XGBoost specific parameters
n_estimators=${XGB_N_ESTIMATORS:-200}
max_depth=${XGB_MAX_DEPTH:-6}
learning_rate=${XGB_LEARNING_RATE:-0.1}
subsample=${XGB_SUBSAMPLE:-0.8}
colsample_bytree=${XGB_COLSAMPLE_BYTREE:-0.8}
gamma=${XGB_GAMMA:-0}
reg_alpha=${XGB_REG_ALPHA:-0}
reg_lambda=${XGB_REG_LAMBDA:-1}
n_jobs=${XGB_N_JOBS:--1}
early_stopping_rounds=${XGB_EARLY_STOPPING:-10}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

scriptDir=$PROJECT_ROOT/src/concept_mapper
fileDir="$PROJECT_ROOT/eraser_movie/$model_file_name/split_dataset"
savePath="$PROJECT_ROOT/eraser_movie/$model_file_name/result_xgb"

mkdir -p ${savePath}

python ${scriptDir}/xgboost_classifier.py \
    --train_file_path ${fileDir}/train/train_df_${layer}.csv \
    --validate_file_path ${fileDir}/validation/validation_df_${layer}.csv \
    --layer ${layer} \
    --save_path ${savePath} \
    --n_estimators ${n_estimators} \
    --max_depth ${max_depth} \
    --learning_rate ${learning_rate} \
    --subsample ${subsample} \
    --colsample_bytree ${colsample_bytree} \
    --gamma ${gamma} \
    --reg_alpha ${reg_alpha} \
    --reg_lambda ${reg_lambda} \
    --n_jobs ${n_jobs} \
    --early_stopping_rounds ${early_stopping_rounds} \
    --do_train \
    --do_validate

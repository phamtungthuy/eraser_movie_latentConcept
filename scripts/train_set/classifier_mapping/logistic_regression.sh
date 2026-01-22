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

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

scriptDir=$PROJECT_ROOT/src/classifier_mapping
if [ "$USE_KMEANS" = "true" ]; then
    echo "Using K-Means Folders for Logistic Regression..."
    fileDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/split_dataset_kmeans"
    savePath="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/result_kmeans"
else
    echo "Using Agglomerative Folders for Logistic Regression..."
    fileDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/split_dataset"
    savePath="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/result"
fi

mkdir -p ${savePath}
python ${scriptDir}/logistic_regression.py --train_file_path ${fileDir}/train/train_df_${layer}.csv --validate_file_path ${fileDir}/validation/validation_df_${layer}.csv --layer ${layer} --save_path ${savePath} --do_train --do_validate

#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

model=${MODEL:-google-bert/bert-base-cased}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

if [ "$USE_KMEANS" = "true" ]; then
    fileDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/result_kmeans/validate_predictions/"
else
    fileDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/result/validate_predictions/"
fi
scriptDir="$PROJECT_ROOT/src/classifier_mapping"

layer=${LAYER:-12}
python ${scriptDir}/get_prediction_stats.py \
  --layer ${layer} \
  --file_path ${fileDir}

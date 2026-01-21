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
validation_size=${VALIDATION_SIZE:-0.1}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

scriptDir=$PROJECT_ROOT/src/classifier_mapping

saveDir="$PROJECT_ROOT/eraser_movie/$model_file_name/split_dataset" #'split_dataset_CLS'
mkdir -p ${saveDir}

filePath="$PROJECT_ROOT/eraser_movie/$model_file_name/clusters_csv_train/"

python ${scriptDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer ${layer} \
  --validation_size ${validation_size} \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
 --is_first_file \


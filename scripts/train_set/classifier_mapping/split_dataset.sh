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

if [ "$USE_KMEANS" = "true" ]; then
    echo "Using K-Means Folders for Split..."
    saveDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/split_dataset_kmeans"
    filePath="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/clusters_csv_train_kmeans/"
else
    echo "Using Agglomerative Folders for Split..."
    saveDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/split_dataset"
    filePath="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/clusters_csv_train/"
fi
mkdir -p ${saveDir}

python ${scriptDir}/split_dataset.py \
  --file_path ${filePath} \
  --layer ${layer} \
  --validation_size ${validation_size} \
  --train_dataset_save_path ${saveDir}/train/ \
  --validation_dataset_save_path ${saveDir}/validation/ \
  --id_save_filename ${saveDir}/id.txt \
 --is_first_file \


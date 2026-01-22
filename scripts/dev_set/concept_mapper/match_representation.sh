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
minfreq=${MINFREQ:-5}
maxfreq=${MAXFREQ:-20}
delfreq=${DELFREQ:-1000000}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')


scriptDir="$PROJECT_ROOT/src/concept_mapper"
input=${DEV_DATA_FILE}
working_file=$input.tok.sent_len

dataPath="$PROJECT_ROOT/${DATASET_FOLDER}_dev/$model_file_name"

savePath="$PROJECT_ROOT/${DATASET_FOLDER}_dev/$model_file_name/position_representation_info"
mkdir -p $savePath

layer=12
saveFile=$savePath/explanation_words_representation_layer${layer}.csv
explanation="$PROJECT_ROOT/"${DATASET_FOLDER}_dev"/$model_file_name/CLS_explanation/explanation_CLS.txt"
python ${scriptDir}/match_representation.py --datasetFile $dataPath/layer$layer/${working_file}-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json --explanationFile $explanation --outputFile $saveFile
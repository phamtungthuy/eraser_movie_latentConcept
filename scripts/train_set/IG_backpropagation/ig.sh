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

scriptDir="$PROJECT_ROOT/src/IG_backpropagation"
inputFile="$PROJECT_ROOT/eraser_movie/movie_train.txt.tok.sent_len"
outDir="$PROJECT_ROOT/eraser_movie/$model_file_name/IG_attributions"
layer=${LAYER:-12}
saveFile=${outDir}/IG_explanation_layer_${layer}.csv

mkdir -p ${outDir}

python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} ${layer} ${saveFile}
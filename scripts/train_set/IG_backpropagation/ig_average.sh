#!/bin/bash

# Run IG with Average Baseline (Mean of all word embeddings)

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
inputFile="$PROJECT_ROOT/$DATASET_FOLDER/$TRAIN_DATA_FILE.tok.sent_len"
outDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/IG_attributions"
layer=${LAYER:-12}
saveFile=${outDir}/IG_explanation_layer_${layer}_average.csv

mkdir -p ${outDir}

echo "Running IG with Average Embedding Baseline..."
python ${scriptDir}/ig_for_sequence_classification.py ${inputFile} ${model} ${layer} ${saveFile} --baseline average
echo "Done! Saved to $saveFile"

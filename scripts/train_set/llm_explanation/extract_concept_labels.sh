#!/bin/bash

# Extract concept labels from training data cluster assignments
# Groups sentences by their [CLS] cluster ID


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

model=${MODEL:-google-bert/bert-base-cased}
layer=${LAYER:-12}
# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')
# Input files
CLUSTERS_FILE="${PROJECT_ROOT}/$DATASET_FOLDER/${model_file_name}/layer${layer}/results/clusters-400.txt"
SENTENCES_FILE="${PROJECT_ROOT}/$DATASET_FOLDER/${model_file_name}/layer${layer}/$TRAIN_DATA_FILE.tok.sent_len-layer${LAYER}_min_5_max_20-sentences.json"

# Output
OUTPUT_FILE="${PROJECT_ROOT}/$DATASET_FOLDER/${model_file_name}/layer${layer}/results/concept_labels.json"

echo "=========================================="
echo "Extracting concept labels"
echo "=========================================="
echo "Clusters: ${CLUSTERS_FILE}"
echo "Sentences: ${SENTENCES_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo ""

python "${PROJECT_ROOT}/src/llm_explanation/extract_concept_labels.py" \
    --clusters-file "${CLUSTERS_FILE}" \
    --sentences-file "${SENTENCES_FILE}" \
    --output "${OUTPUT_FILE}" \
    --max-examples 5

echo ""
echo "Done!"

#!/bin/bash

# Extract concept labels from training data cluster assignments
# Groups sentences by their [CLS] cluster ID

SCRIPT_DIR="$(dirname "$0")"
ROOT_DIR="${SCRIPT_DIR}/../../.."

MODEL_NAME="google-bert_bert-base-cased"
LAYER=12

# Input files
CLUSTERS_FILE="${ROOT_DIR}/eraser_movie/${MODEL_NAME}/layer${LAYER}/results/clusters-400.txt"
SENTENCES_FILE="${ROOT_DIR}/eraser_movie/${MODEL_NAME}/layer${LAYER}/movie_train.txt.tok.sent_len-layer${LAYER}_min_5_max_20-sentences.json"

# Output
OUTPUT_FILE="${ROOT_DIR}/eraser_movie/${MODEL_NAME}/layer${LAYER}/results/concept_labels.json"

echo "=========================================="
echo "Extracting concept labels"
echo "=========================================="
echo "Clusters: ${CLUSTERS_FILE}"
echo "Sentences: ${SENTENCES_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo ""

python "${ROOT_DIR}/src/llm_explanation/extract_concept_labels.py" \
    --clusters-file "${CLUSTERS_FILE}" \
    --sentences-file "${SENTENCES_FILE}" \
    --output "${OUTPUT_FILE}" \
    --max-examples 5

echo ""
echo "Done!"

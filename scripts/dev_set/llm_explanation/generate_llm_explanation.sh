#!/bin/bash

# Generate LLM explanations for dev set predictions
# Uses sentence-level concept clusters and example sentences

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
[ -f "$PROJECT_ROOT/.env" ] && source "$PROJECT_ROOT/.env"
set +a

model=${MODEL:-google-bert/bert-base-cased}
LAYER=12
# Convert model name to valid filename
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

# Input files
INPUT_DATA="${PROJECT_ROOT}/data/movie_dev_subset.json"
CLUSTERS_FILE="${PROJECT_ROOT}/eraser_movie_dev/${model_file_name}/layer${LAYER}/results/clusters-400.txt"
CONCEPT_LABELS="${PROJECT_ROOT}/eraser_movie/${model_file_name}/layer${LAYER}/results/concept_labels.json"

# Output
OUTPUT_DIR="${PROJECT_ROOT}/eraser_movie_dev/${model_file_name}/llm_explanation"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/llm_explanations_layer_${LAYER}.json"

# LLM settings
PROVIDER="${LLM_PROVIDER:-ollama}"
MODEL="${LLM_MODEL:-llama3.2}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"

echo "=========================================="
echo "Generating LLM Explanations"
echo "=========================================="
echo "Input: ${INPUT_DATA}"
echo "Clusters: ${CLUSTERS_FILE}"
echo "Concepts: ${CONCEPT_LABELS}"
echo "Output: ${OUTPUT_FILE}"
echo "Provider: ${PROVIDER}"
echo "Model: ${MODEL}"
echo "Max samples: ${MAX_SAMPLES}"
echo ""

python "${PROJECT_ROOT}/src/llm_explanation/generate_llm_explanation.py" \
    --input-data "${INPUT_DATA}" \
    --clusters-file "${CLUSTERS_FILE}" \
    --concept-labels "${CONCEPT_LABELS}" \
    --output "${OUTPUT_FILE}" \
    --provider "${PROVIDER}" \
    --model "${MODEL}" \
    --max-samples "${MAX_SAMPLES}"

echo ""
echo "Done!"

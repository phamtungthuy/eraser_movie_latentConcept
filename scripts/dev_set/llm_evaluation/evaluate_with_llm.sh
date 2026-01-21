#!/bin/bash

# Evaluate LLM explanations using LLM-as-annotator
# Scores: Faithfulness, Plausibility, Concept Coherence

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

# Input: explanations from previous step
EXPLANATIONS="${PROJECT_ROOT}/eraser_movie_dev/${model_file_name}/llm_explanation/llm_explanations_layer_${LAYER}.json"

# Output
OUTPUT_DIR="${PROJECT_ROOT}/eraser_movie_dev/${model_file_name}/llm_evaluation"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/evaluation_results_layer_${LAYER}.json"

# LLM settings
PROVIDER="${LLM_PROVIDER:-ollama}"
MODEL="${LLM_MODEL:-llama3.2}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

echo "=========================================="
echo "Evaluating LLM Explanations"
echo "=========================================="
echo "Explanations: ${EXPLANATIONS}"
echo "Output: ${OUTPUT_FILE}"
echo "Provider: ${PROVIDER}"
echo "Model: ${MODEL}"
echo ""

python "${PROJECT_ROOT}/src/llm_evaluation/evaluate_with_llm.py" \
    --explanations "${EXPLANATIONS}" \
    --output "${OUTPUT_FILE}" \
    --provider "${PROVIDER}" \
    --model "${MODEL}" \
    --max-samples "${MAX_SAMPLES}"

echo ""
echo "Done!"

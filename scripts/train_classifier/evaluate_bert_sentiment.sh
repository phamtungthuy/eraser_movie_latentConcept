#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

# Default values
MODEL_PATH=${1:-"$PROJECT_ROOT/trained_models/bert_sentiment"}
TEST_FILE=${2:-"$PROJECT_ROOT/data/movie_dev_subset.json"}
BATCH_SIZE=32
MAX_LENGTH=256

echo "=============================================="
echo "Evaluating BERT Sentiment Model"
echo "=============================================="
echo "Model Path: $MODEL_PATH"
echo "Test File: $TEST_FILE"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

# Run evaluation
python "$PROJECT_ROOT/src/train_classifier/evaluate_bert_sentiment.py" \
    --model-path "$MODEL_PATH" \
    --test-file "$TEST_FILE" \
    --batch-size $BATCH_SIZE \
    --max-length $MAX_LENGTH \
    --show-errors \
    --num-errors 5

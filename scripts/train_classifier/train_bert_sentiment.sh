#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

# Use values from config.env or defaults
BASE_MODEL=${MODEL:-google-bert/bert-base-cased}

# Training configuration
TRAIN_FILE="$PROJECT_ROOT/data/movie_train.json"
OUTPUT_DIR="$PROJECT_ROOT/trained_models/bert_sentiment"
EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=2e-5
MAX_LENGTH=256
VAL_SPLIT=0.1
SEED=42

echo "=============================================="
echo "Training BERT for Sentiment Classification"
echo "=============================================="
echo "Base Model: $BASE_MODEL"
echo "Train File: $TRAIN_FILE"
echo "Output Dir: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python "$PROJECT_ROOT/src/train_classifier/train_bert_sentiment.py" \
    --train-file "$TRAIN_FILE" \
    --model-name "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --max-length $MAX_LENGTH \
    --val-split $VAL_SPLIT \
    --seed $SEED

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "After training is complete, update your config.env:"
echo "MODEL=$OUTPUT_DIR"
echo ""
echo "Then you can run the LACOAT pipeline with your fine-tuned model."

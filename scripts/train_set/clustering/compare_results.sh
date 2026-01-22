#!/bin/bash

# Compare Agglomerative vs K-Means clustering quality

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

model=${MODEL:-google-bert/bert-base-cased}
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')
layer=${LAYER:-12}
clusters=${CLUSTER_NUM:-400}

# Paths
BASE_DIR="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/layer$layer"
POINT_FILE="$BASE_DIR/processed-point.npy"

# Agglomerative results (default folder)
AGG_LABELS="$BASE_DIR/results/labels-$clusters.txt"

# K-Means results (results_kmeans folder)
KMEANS_LABELS="$BASE_DIR/results_kmeans/labels-$clusters.txt"

echo "Comparing clustering for Layer $layer, K=$clusters"
echo "Points: $POINT_FILE"
echo "Agglomerative: $AGG_LABELS"
echo "K-Means: $KMEANS_LABELS"
echo ""

python "$PROJECT_ROOT/src/clustering/compare_clustering.py" \
    --point-file "$POINT_FILE" \
    --agg-labels "$AGG_LABELS" \
    --kmeans-labels "$KMEANS_LABELS"

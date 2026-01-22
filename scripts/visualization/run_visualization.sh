#!/bin/bash

# Visualize clusters for Layer 0, 6 and 12
# Generates word cloud images and sentence lists (grid + individual files)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

model=${MODEL:-google-bert/bert-base-cased}
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

# Configuration
TOP_K=12
COLS=4
LAYERS=(0 6 12)

for layer in "${LAYERS[@]}"; do
    echo "------------------------------------------------"
    echo "Visualizing Layer $layer Clusters"
    echo "------------------------------------------------"
    
    BASE_DIR="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/layer$layer/results"
    CLUSTER_FILE="$BASE_DIR/clusters-${CLUSTER_NUM:-400}.txt"
    JSON_FILE="$BASE_DIR/concept_labels.json"
    
    # Check if files exist
    HAS_JSON=false
    HAS_CLUSTERS=false
    [ -f "$JSON_FILE" ] && HAS_JSON=true
    [ -f "$CLUSTER_FILE" ] && HAS_CLUSTERS=true
    
    # 1. Generate Sentences View (if available)
    if [ "$HAS_JSON" = true ]; then
        echo "Generating Sentences View..."
        OUTPUT_FILE="$BASE_DIR/clusters_visualization_sentences.png"
        python "$PROJECT_ROOT/src/visualization/visualize_clusters.py" \
            --mode sentences \
            --json-file "$JSON_FILE" \
            --output-file "$OUTPUT_FILE" \
            --top-k $TOP_K \
            --cols 3 \
            --save-individual
    fi
    
    # 2. Generate WordCloud View (if available)
    if [ "$HAS_CLUSTERS" = true ]; then
        echo "Generating WordCloud View..."
        OUTPUT_FILE="$BASE_DIR/clusters_visualization_wordcloud.png"
        python "$PROJECT_ROOT/src/visualization/visualize_clusters.py" \
            --mode wordcloud \
            --clusters-file "$CLUSTER_FILE" \
            --output-file "$OUTPUT_FILE" \
            --top-k $TOP_K \
            --cols $COLS \
            --save-individual
    fi
    
    if [ "$HAS_JSON" = false ] && [ "$HAS_CLUSTERS" = false ]; then
        echo "No data found for Layer $layer."
    fi
    echo ""
done

echo "Done! Check the results folders for grid images and '_individual' subfolders."

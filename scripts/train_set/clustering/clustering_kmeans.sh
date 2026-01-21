#!/bin/bash

# K-Means Clustering Script
# Alternative to Agglomerative Clustering for faster processing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

scriptDir="$PROJECT_ROOT/src/clustering"
ERASER_MOVIE_DIR="$PROJECT_ROOT/eraser_movie"

# Use values from config.env or defaults
model=${MODEL:-google-bert/bert-base-cased}
layer=${LAYER:-12}
minfreq=${MINFREQ:-5}
maxfreq=${MAXFREQ:-20}
delfreq=${DELFREQ:-1000000}
clusters=${CLUSTERS:-400,400,400}

# Convert model name to valid filename
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

outputDir="$ERASER_MOVIE_DIR/$model_file_name/layer${layer}"
input=movie_train.txt
working_file=$input.tok.sent_len

VOCABFILE=${outputDir}/processed-vocab.npy
POINTFILE=${outputDir}/processed-point.npy
RESULTPATH=${outputDir}/results

mkdir -p ${RESULTPATH}

echo "=============================================="
echo "K-Means Clustering for Latent Concepts"
echo "=============================================="
echo "Model: $model"
echo "Layer: $layer"
echo "Clusters: $clusters"
echo "Vocab file: $VOCABFILE"
echo "Point file: $POINTFILE"
echo "Output: $RESULTPATH"
echo "=============================================="

# Check if input files exist
if [ ! -f "$VOCABFILE" ]; then
    echo "ERROR: Vocab file not found: $VOCABFILE"
    echo "Please run the previous pipeline steps first."
    exit 1
fi

if [ ! -f "$POINTFILE" ]; then
    echo "ERROR: Point file not found: $POINTFILE"
    echo "Please run the previous pipeline steps first."
    exit 1
fi

echo "Creating K-Means Clusters!"
python -u ${scriptDir}/get_kmeans_clusters.py \
    --vocab-file $VOCABFILE \
    --point-file $POINTFILE \
    --output-path $RESULTPATH \
    --cluster $clusters \
    --range 1

echo "DONE!"

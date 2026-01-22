#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

# Use values from config.env or defaults
model=${MODEL:-google-bert/bert-base-cased}
cluster_num=${CLUSTER_NUM:-400}
i=${LAYER:-12}
data=$TRAIN_DATA_FILE
minfreq=${MINFREQ:-5}
maxfreq=${MAXFREQ:-20}
delfreq=${DELFREQ:-1000000}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

clusterDir=$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name
scriptDir="$PROJECT_ROOT/src/classifier_mapping"

if [ "$USE_KMEANS" = "true" ]; then
    echo "Using K-Means Clustering Results..."
    CLUSTER_PATH="${clusterDir}/layer$i/results_kmeans/clusters-$cluster_num.txt"
    saveDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/clusters_csv_train_kmeans"
else
    echo "Using Agglomerative Clustering Results..."
    CLUSTER_PATH="${clusterDir}/layer$i/results/clusters-$cluster_num.txt"
    saveDir="$PROJECT_ROOT/$DATASET_FOLDER/$model_file_name/clusters_csv_train"
fi
mkdir -p $saveDir

echo "Cluster Path: $CLUSTER_PATH"
echo "Save Directory: $saveDir"

datasetFile=${clusterDir}/layer$i/$data.tok.sent_len-layer${i}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
python ${scriptDir}/generate_csv_file.py --dataset_file $datasetFile --cluster_file $CLUSTER_PATH --output_file $saveDir/clusters-map$i.csv

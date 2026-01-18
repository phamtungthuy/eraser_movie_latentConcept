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
data=movie_train.txt
minfreq=${MINFREQ:-5}
maxfreq=${MAXFREQ:-20}
delfreq=${DELFREQ:-1000000}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

clusterDir=$PROJECT_ROOT/eraser_movie/$model_file_name
scriptDir="$PROJECT_ROOT/src/classifier_mapping"

saveDir=$PROJECT_ROOT/eraser_movie/$model_file_name/clusters_csv_train
mkdir -p $saveDir
datasetFile=${clusterDir}/layer$i/$data.tok.sent_len-layer${i}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
python ${scriptDir}/generate_csv_file.py --dataset_file $datasetFile --cluster_file ${clusterDir}/layer$i/results/clusters-$cluster_num.txt --output_file $saveDir/clusters-map$i.csv

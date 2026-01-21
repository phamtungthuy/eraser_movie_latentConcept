#!/bin/bash


# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
model=${MODEL:-google-bert/bert-base-cased}
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

scriptDir=$PROJECT_ROOT/src/generate_explanation_files
clusterPath=$PROJECT_ROOT/eraser_movie/$model_file_name
explanation=explanation_words.txt
clusterSize=400
percentage=90

savePath=$PROJECT_ROOT/eraser_movie/$model_file_name/cluster_Labels_$percentage%
mkdir $savePath

layer=12
saveFile=${savePath}/clusterLabel_layer$layer.json
echo Layer$layer
python ${scriptDir}/generate_cluster_label_all_tokens.py -c $clusterPath/layer$layer/results/clusters-$clusterSize.txt -e $explanation -p $percentage -s $saveFile
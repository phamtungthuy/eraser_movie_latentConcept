#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

# Load configuration from config.env
model=${MODEL:-google-bert/bert-base-cased}
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')


scriptDir="$PROJECT_ROOT/src/generate_explanation_files"
inputDir="$PROJECT_ROOT/eraser_movie_dev/$model_file_name/IG_attributions"
outDir="$PROJECT_ROOT/eraser_movie_dev/$model_file_name/IG_explanation_files_mass_50"

mkdir ${outDir}

layer=12
echo ${inputDir}/IG_explanation_layer_${layer}.csv
saveFile=${outDir}/explanation_layer_${layer}.txt
python ${scriptDir}/generate_IG_explanation_salient_words.py ${inputDir}/IG_explanation_layer_${layer}.csv ${saveFile} top-k --attribution_mass 0.5
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
inputFile="$PROJECT_ROOT/eraser_movie_dev/movie_dev_subset.txt.tok"

saveDir="$PROJECT_ROOT/eraser_movie_dev/$model_file_name/CLS_explanation"

mkdir -p ${saveDir}

layer=12
python ${scriptDir}/generate_CLS_explanation.py --dataset-name-or-path ${inputFile} --model-name ${model} --tokenizer-name ${model} --save-dir ${saveDir}




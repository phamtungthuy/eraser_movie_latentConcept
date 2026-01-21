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
layer=${LAYER:-12}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

scriptDir="$PROJECT_ROOT/src/concept_mapper"
fileDir=$PROJECT_ROOT/eraser_movie_dev/$model_file_name/position_representation_info  #saliency_representation_info #
classifierDir=$PROJECT_ROOT/eraser_movie/$model_file_name/result/model

mkdir -p $PROJECT_ROOT/eraser_movie_dev/$model_file_name/latent_concepts
mkdir -p $PROJECT_ROOT/eraser_movie_dev/$model_file_name/latent_concepts/position_prediction


echo ${layer}
python ${scriptDir}/logistic_regression.py \
  --test_file_path ${fileDir}/explanation_words_representation_layer${layer}.csv \
  --layer ${layer} \
  --save_path $PROJECT_ROOT/eraser_movie_dev/$model_file_name/latent_concepts/position_prediction/ \
  --do_predict \
  --classifier_file_path $classifierDir/layer_${layer}_classifier.pkl \
  --load_classifier_from_local
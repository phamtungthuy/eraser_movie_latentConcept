#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

scriptDir="$PROJECT_ROOT/src/clustering"
inputPath="$PROJECT_ROOT/data" # path to a sentence file
ERASER_MOVIE_DIR="$PROJECT_ROOT/eraser_movie"

# Use values from config.env or defaults
model=${MODEL:-google-bert/bert-base-cased}
layer=${LAYER:-12}
input=movie_train.txt
sentence_length=${SENTENCE_LENGTH:-300}
minfreq=${MINFREQ:-5}
maxfreq=${MAXFREQ:-20}
delfreq=${DELFREQ:-1000000}
clusters=${CLUSTERS:-400,400,400}

# Convert model name to valid filename (replace / and other special chars)
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

outputDir="$ERASER_MOVIE_DIR/$model_file_name/layer${layer}" #do not change this
mkdir -p ${outputDir}

working_file=$input.tok.sent_len #do not change this

# Extract layer-wise activations
python ${scriptDir}/neurox_extraction.py \
      --model_desc ${model} \
      --input_corpus ${ERASER_MOVIE_DIR}/${working_file} \
      --output_file ${outputDir}/${working_file}.activations.json \
      --output_type json \
      --decompose_layers \
      --include_special_tokens \
      --filter_layers ${layer} \
      --input_type text

# Create a dataset file with word and sentence indexes
# python ${scriptDir}/create_data_single_layer.py --text-file $ERASER_MOVIE_DIR/${working_file}.modified --activation-file ${outputDir}/${working_file}.activations-layer${layer}.json --output-prefix ${outputDir}/${working_file}-layer${layer}

# Filter number of tokens to fit in the memory for clustering. Input file will be from step 4. minfreq sets the minimum frequency. If a word type appears is coming less than minfreq, it will be dropped. if a word comes
# python ${scriptDir}/frequency_filter_data.py --input-file ${outputDir}/${working_file}-layer${layer}-dataset.json --frequency-file $ERASER_MOVIE_DIR/${working_file}.words_freq --sentence-file ${outputDir}/${working_file}-layer${layer}-sentences.json --minimum-frequency $minfreq --maximum-frequency $maxfreq --delete-frequency ${delfreq} --output-file ${outputDir}/${working_file}-layer${layer}

# Run clustering
mkdir -p ${outputDir}/results
DATASETPATH=${outputDir}/${working_file}-layer${layer}_min_${minfreq}_max_${maxfreq}_del_${delfreq}-dataset.json
VOCABFILE=${outputDir}/processed-vocab.npy
POINTFILE=${outputDir}/processed-point.npy
RESULTPATH=${outputDir}/results
# CLUSTERS variable loaded from config.env
# first number is number of clusters to start with, second is number of clusters to stop at and third one is the increment from the first value
# 600 1000 200 means [600,800,1000] number of clusters

# echo "Extracting Data!"
# python -u ${scriptDir}/extract_data.py --input-file $DATASETPATH --output-path $outputDir

# echo "Creating Agglomerative Clusters!"
# python -u ${scriptDir}/get_agglomerative_clusters.py --vocab-file $VOCABFILE --point-file $POINTFILE --output-path $RESULTPATH  --cluster $CLUSTERS --range 1
# echo "DONE!"

# echo "Creating K-Means Clusters!"
# python -u ${scriptDir}/get_kmeans_clusters.py --vocab-file $VOCABFILE --point-file $POINTFILE --output-path $RESULTPATH --cluster $CLUSTERS --range 1
# echo "DONE!"
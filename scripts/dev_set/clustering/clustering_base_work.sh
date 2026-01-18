#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

scriptDir="$PROJECT_ROOT/src/clustering"
inputPath="$PROJECT_ROOT/data" # path to a sentence file
input=movie_dev_subset.txt #name of the sentence file

mkdir -p "$PROJECT_ROOT/eraser_movie_dev"
ERASER_MOVIE_DIR="$PROJECT_ROOT/eraser_movie_dev"

# maximum sentence length
sentence_length=${SENTENCE_LENGTH:-300}

working_file=$input.tok.sent_len #do not change this

#1. Tokenize text with moses tokenizer
perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${inputPath}/$input > $ERASER_MOVIE_DIR/$input.tok

#2. Do sentence length filtering and keep sentences max length of 300
python ${scriptDir}/sentence_length.py --text-file $ERASER_MOVIE_DIR/$input.tok --length ${sentence_length} --output-file $ERASER_MOVIE_DIR/$input.tok.sent_len

#3. Modify the input file to be compatible with the model
python ${scriptDir}/modify_input.py --text-file $ERASER_MOVIE_DIR/$input.tok.sent_len --output-file $ERASER_MOVIE_DIR/$input.tok.sent_len.modified

#4. Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file $ERASER_MOVIE_DIR/${working_file}.modified --output-file $ERASER_MOVIE_DIR/${working_file}.words_freq


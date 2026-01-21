#!/bin/bash

# Script to compare all concept mapping methods
# This will train and validate all methods and compare their performance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration from config.env
set -a
[ -f "$PROJECT_ROOT/config.env" ] && source "$PROJECT_ROOT/config.env"
set +a

# Use values from config.env or defaults
model=${MODEL:-google-bert/bert-base-cased}
layer=${LAYER:-12}

# Convert model name to valid filename
model_file_name=$(echo "$model" | sed 's/\//_/g' | sed 's/[^a-zA-Z0-9._-]/-/g')

echo "=========================================="
echo "Concept Mapper Comparison"
echo "=========================================="
echo "Model: $model"
echo "Layer: $layer"
echo "=========================================="
echo ""

# Create comparison results directory
comparison_dir="$PROJECT_ROOT/eraser_movie/$model_file_name/comparison_results"
mkdir -p ${comparison_dir}

# Results file
results_file="${comparison_dir}/comparison_results.txt"
echo "Concept Mapper Comparison Results" > ${results_file}
echo "Model: $model" >> ${results_file}
echo "Layer: $layer" >> ${results_file}
echo "Date: $(date)" >> ${results_file}
echo "========================================" >> ${results_file}
echo "" >> ${results_file}

# Function to run a method and capture results
run_method() {
    method_name=$1
    script_path=$2
    
    echo ""
    echo "=========================================="
    echo "Running: $method_name"
    echo "=========================================="
    
    start_time=$(date +%s)
    
    # Run the script and capture output
    output=$(bash ${script_path} 2>&1)
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Extract accuracy from output
    accuracy=$(echo "$output" | grep -oP "Accuracy:\s*\K[0-9.]+|Validation Accuracy:\s*\K[0-9.]+" | tail -1)
    
    # Print results
    echo "Completed in ${duration} seconds"
    if [ -n "$accuracy" ]; then
        echo "Accuracy: $accuracy"
    else
        echo "Could not extract accuracy"
    fi
    
    # Save to results file
    echo "$method_name:" >> ${results_file}
    echo "  Duration: ${duration} seconds" >> ${results_file}
    if [ -n "$accuracy" ]; then
        echo "  Accuracy: $accuracy" >> ${results_file}
    else
        echo "  Accuracy: N/A" >> ${results_file}
    fi
    echo "" >> ${results_file}
    
    # Save full output
    echo "$output" > "${comparison_dir}/${method_name}_output.log"
}

# Run all methods
run_method "Logistic Regression" "$SCRIPT_DIR/logistic_regression.sh"
run_method "Neural Network" "$SCRIPT_DIR/neural_network.sh"
run_method "Random Forest" "$SCRIPT_DIR/random_forest.sh"
run_method "XGBoost" "$SCRIPT_DIR/xgboost_classifier.sh"

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo "Results saved to: ${results_file}"
echo ""
echo "Summary:"
cat ${results_file}

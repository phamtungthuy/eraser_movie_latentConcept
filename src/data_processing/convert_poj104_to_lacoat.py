"""
Convert POJ-104 dataset (Clone Detection) to LACOAT format.

This script converts the CodeXGLUE POJ-104 dataset from JSONL format
to the JSON format used by the LACOAT pipeline (similar to movie_train.json).

POJ-104 Format (JSONL):
    {"label": "1", "index": "0", "code": "..."}

LACOAT Format (JSON):
    [
        {"id": 0, "sentence": "...", "label": 0},
        ...
    ]

Usage:
    python convert_poj104_to_lacoat.py \
        --input-dir CodeXGLUE/Code-Code/Clone-detection-POJ-104/dataset \
        --output-dir data/poj104
"""

import argparse
import json
import os
import re
from pathlib import Path


def clean_code(code: str) -> str:
    """
    Clean code for use with BERT tokenizer.
    - Replace newlines with spaces
    - Remove excessive whitespace
    - Keep code readable as a single line
    """
    # Replace newlines and tabs with spaces
    code = code.replace('\n', ' ').replace('\t', ' ')
    # Remove multiple spaces
    code = re.sub(r'\s+', ' ', code)
    # Strip leading/trailing whitespace
    code = code.strip()
    return code


def convert_jsonl_to_json(input_file: str, output_file: str, create_txt: bool = True):
    """
    Convert a JSONL file to LACOAT JSON format.
    
    Parameters
    ----------
    input_file : str
        Path to input JSONL file
    output_file : str
        Path to output JSON file
    create_txt : bool
        Whether to also create a .txt file with sentences (for clustering pipeline)
    """
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            
            # Convert to LACOAT format
            lacoat_item = {
                "id": idx,
                "sentence": clean_code(item["code"]),
                "label": int(item["label"]) - 1,  # POJ-104 labels are 1-indexed, convert to 0-indexed
                "original_label": item["label"],  # Keep original label for reference
                "original_index": item["index"]
            }
            data.append(lacoat_item)
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Converted {len(data)} samples to {output_file}")
    
    # Also create a .txt file with just the sentences (for clustering pipeline)
    if create_txt:
        txt_file = output_file.replace('.json', '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(item["sentence"] + '\n')
        print(f"Created text file: {txt_file}")
    
    return data


def get_label_distribution(data):
    """Get distribution of labels in the dataset."""
    from collections import Counter
    labels = [item["label"] for item in data]
    return Counter(labels)


def main():
    parser = argparse.ArgumentParser(
        description="Convert POJ-104 dataset to LACOAT format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing train.jsonl, valid.jsonl, test.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Don't create .txt files for clustering"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Files to convert
    files = [
        ("train.jsonl", "poj104_train.json"),
        ("valid.jsonl", "poj104_valid.json"),
        ("test.jsonl", "poj104_test.json"),
    ]
    
    print("=" * 60)
    print("Converting POJ-104 Dataset to LACOAT Format")
    print("=" * 60)
    
    for input_name, output_name in files:
        input_path = os.path.join(args.input_dir, input_name)
        output_path = os.path.join(args.output_dir, output_name)
        
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping...")
            continue
        
        print(f"\nProcessing: {input_name}")
        data = convert_jsonl_to_json(
            input_path, 
            output_path, 
            create_txt=not args.no_txt
        )
        
        # Print statistics
        label_dist = get_label_distribution(data)
        print(f"  Labels distribution (top 10): {dict(list(label_dist.most_common(10)))}")
        print(f"  Unique labels: {len(label_dist)}")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Update config.env to point to new data")
    print("2. Train a classifier on the new dataset")
    print("3. Run the LACOAT clustering pipeline")


if __name__ == "__main__":
    main()

"""
Convert BigCloneBench dataset to LACOAT format.

BigCloneBench is a PAIR-WISE binary classification task:
- Input: Two code snippets (code1, code2)
- Output: 1 if they are clones, 0 otherwise

For LACOAT (single-input classification), we concatenate the pair:
- sentence = "[CLS] code1 [SEP] code2 [SEP]" (handled by tokenizer)
- Or concatenate with a separator

Usage:
    python convert_bigclonebench_to_lacoat.py \
        --input-dir CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/dataset \
        --output-dir data/bigclonebench
"""

import argparse
import json
import os
import re
from collections import defaultdict
from tqdm import tqdm


def clean_code(code: str) -> str:
    """Clean code for use with BERT tokenizer."""
    code = code.replace('\n', ' ').replace('\t', ' ')
    code = re.sub(r'\s+', ' ', code)
    code = code.strip()
    return code


def load_functions(data_file: str) -> dict:
    """Load all functions from data.jsonl into a dictionary."""
    functions = {}
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading functions"):
            item = json.loads(line.strip())
            functions[item['idx']] = clean_code(item['func'])
    return functions


def convert_pairs_to_json(
    pairs_file: str,
    functions: dict,
    output_file: str,
    max_samples: int = None,
    separator: str = " [SEP] "
):
    """
    Convert pair-wise data to single-input format.
    
    We concatenate code1 and code2 with a separator.
    """
    data = []
    skipped = 0
    
    with open(pairs_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if max_samples:
        lines = lines[:max_samples]
    
    for idx, line in enumerate(tqdm(lines, desc="Converting pairs")):
        parts = line.strip().split('\t')
        if len(parts) != 3:
            skipped += 1
            continue
        
        idx1, idx2, label = parts
        
        # Get code snippets
        code1 = functions.get(idx1)
        code2 = functions.get(idx2)
        
        if code1 is None or code2 is None:
            skipped += 1
            continue
        
        # Concatenate codes with separator
        # Note: For BERT, the tokenizer will handle [SEP] automatically
        # Here we just concatenate with a simple separator
        combined = f"{code1}{separator}{code2}"
        
        lacoat_item = {
            "id": idx,
            "sentence": combined,
            "label": int(label),
            "code1_idx": idx1,
            "code2_idx": idx2
        }
        data.append(lacoat_item)
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Converted {len(data)} pairs to {output_file}")
    print(f"Skipped {skipped} pairs (missing functions)")
    
    # Create txt file for clustering
    txt_file = output_file.replace('.json', '.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item["sentence"] + '\n')
    print(f"Created text file: {txt_file}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Convert BigCloneBench dataset to LACOAT format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing data.jsonl, train.txt, valid.txt, test.txt"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=50000,
        help="Maximum training samples (default: 50000 to avoid memory issues)"
    )
    parser.add_argument(
        "--max-valid",
        type=int,
        default=10000,
        help="Maximum validation samples"
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=10000,
        help="Maximum test samples"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Converting BigCloneBench Dataset to LACOAT Format")
    print("=" * 60)
    print("\nNote: BigCloneBench is pair-wise classification.")
    print("We concatenate code pairs into single inputs.\n")
    
    # Load all functions
    data_file = os.path.join(args.input_dir, "data.jsonl")
    functions = load_functions(data_file)
    print(f"Loaded {len(functions)} functions\n")
    
    # Convert each split
    files = [
        ("train.txt", "bcb_train.json", args.max_train),
        ("valid.txt", "bcb_valid.json", args.max_valid),
        ("test.txt", "bcb_test.json", args.max_test),
    ]
    
    for input_name, output_name, max_samples in files:
        input_path = os.path.join(args.input_dir, input_name)
        output_path = os.path.join(args.output_dir, output_name)
        
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping...")
            continue
        
        print(f"\nProcessing: {input_name} (max {max_samples} samples)")
        data = convert_pairs_to_json(
            input_path, functions, output_path, max_samples
        )
        
        # Print label distribution
        labels = [item["label"] for item in data]
        label_0 = labels.count(0)
        label_1 = labels.count(1)
        print(f"  Label 0 (not clone): {label_0} ({label_0/len(labels)*100:.1f}%)")
        print(f"  Label 1 (clone): {label_1} ({label_1/len(labels)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {args.output_dir}")
    print("\nThis is BINARY classification (0/1) similar to movie sentiment!")


if __name__ == "__main__":
    main()

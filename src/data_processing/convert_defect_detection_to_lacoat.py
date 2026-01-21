"""
Convert Defect Detection dataset to LACOAT format.

Defect Detection (Devign) is a binary classification task:
- Input: Source code function
- Output: 1 if insecure/vulnerable, 0 if secure

LACOAT format:
- JSON with fields: id, sentence (the code), label
- TXT file with one sentence per line (for clustering)

Usage:
    python convert_defect_detection_to_lacoat.py \
        --input-dir data/Defect-detection \
        --output-dir data/Defect-detection
"""

import argparse
import json
import os
import re
from tqdm import tqdm


def clean_code(code: str) -> str:
    """Clean code for use with BERT tokenizer."""
    # Replace encoded newlines and tabs
    code = code.replace('\\n', ' ').replace('\\t', ' ')
    code = code.replace('\n', ' ').replace('\t', ' ')
    # Collapse multiple whitespace to single space
    code = re.sub(r'\s+', ' ', code)
    code = code.strip()
    return code


def convert_jsonl_to_lacoat(
    input_file: str,
    output_json: str,
    output_txt: str,
    max_samples: int = None
):
    """
    Convert .jsonl file to LACOAT format (.json and .txt).
    """
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if max_samples:
        lines = lines[:max_samples]
    
    for line in tqdm(lines, desc=f"Processing {os.path.basename(input_file)}"):
        item = json.loads(line.strip())
        
        # Clean the code
        cleaned_code = clean_code(item['func'])
        
        lacoat_item = {
            "id": item['idx'],
            "sentence": cleaned_code,
            "label": item['target']  # 0 = secure, 1 = vulnerable
        }
        data.append(lacoat_item)
    
    # Save as JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Created JSON file: {output_json} ({len(data)} samples)")
    
    # Save as TXT (one sentence per line for clustering)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item["sentence"] + '\n')
    print(f"Created TXT file: {output_txt}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Convert Defect Detection dataset to LACOAT format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/Defect-detection",
        help="Directory containing train.jsonl, valid.jsonl, test.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/Defect-detection",
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split (None for all)"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Converting Defect Detection Dataset to LACOAT Format")
    print("=" * 60)
    print("\nDefect Detection is binary classification:")
    print("  - Label 0: Secure code")
    print("  - Label 1: Vulnerable/insecure code\n")
    
    # Define file mappings
    files = [
        ("train.jsonl", "defect_train.json", "defect_train.txt"),
        ("valid.jsonl", "defect_valid.json", "defect_valid.txt"),
        ("test.jsonl", "defect_test.json", "defect_test.txt"),
    ]
    
    all_stats = []
    
    for input_name, output_json, output_txt in files:
        input_path = os.path.join(args.input_dir, input_name)
        output_json_path = os.path.join(args.output_dir, output_json)
        output_txt_path = os.path.join(args.output_dir, output_txt)
        
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping...")
            continue
        
        print(f"\nProcessing: {input_name}")
        data = convert_jsonl_to_lacoat(
            input_path, output_json_path, output_txt_path, args.max_samples
        )
        
        # Print label distribution
        labels = [item["label"] for item in data]
        label_0 = labels.count(0)
        label_1 = labels.count(1)
        total = len(labels)
        
        stats = {
            "split": input_name.replace(".jsonl", ""),
            "total": total,
            "label_0": label_0,
            "label_1": label_1,
            "pct_0": label_0/total*100 if total > 0 else 0,
            "pct_1": label_1/total*100 if total > 0 else 0,
        }
        all_stats.append(stats)
        
        print(f"  Label 0 (secure): {label_0} ({stats['pct_0']:.1f}%)")
        print(f"  Label 1 (vulnerable): {label_1} ({stats['pct_1']:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"{'Split':<10} {'Total':>8} {'Secure':>10} {'Vulnerable':>12}")
    print("-" * 45)
    for stats in all_stats:
        print(f"{stats['split']:<10} {stats['total']:>8} {stats['label_0']:>10} {stats['label_1']:>12}")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {args.output_dir}")
    print("\nFiles created:")
    for _, output_json, output_txt in files:
        json_path = os.path.join(args.output_dir, output_json)
        txt_path = os.path.join(args.output_dir, output_txt)
        if os.path.exists(json_path):
            print(f"  - {output_json} (LACOAT JSON format)")
            print(f"  - {output_txt} (for clustering)")
    
    print("\nUsage with LACOAT:")
    print("  1. Use .txt files for generating embeddings and clustering")
    print("  2. Use .json files for training classifier with evidence")


if __name__ == "__main__":
    main()

"""
Extract concept labels from cluster assignments.

Groups sentences by cluster ID for ALL tokens.
Format: token|||occurrence|||sent_idx|||position|||cluster_id
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def extract_all_cluster_sentences(clusters_file: str, sentences_file: str) -> dict:
    """
    Parse clusters file and get sentences for ALL clusters.
    
    Format: token|||occurrence|||sent_idx|||position|||cluster_id
    Example: version|||1|||84|||15|||114
    
    Returns:
        dict: cluster_id -> list of sentences (deduplicated)
    """
    # Load sentences
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    print(f"Loaded {len(sentences)} sentences")
    
    # Parse clusters file - map each cluster to sentences
    cluster_sentences = defaultdict(set)  # Use set to deduplicate
    
    with open(clusters_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|||')
            if len(parts) >= 5:
                # Format: token|||occurrence|||sent_idx|||position|||cluster_id
                sent_idx = int(parts[2])
                cluster_id = parts[4]
                
                if sent_idx < len(sentences):
                    # Clean sentence
                    sent = sentences[sent_idx].replace("[CLS]", "").replace("[SEP]", "").strip()
                    if sent:
                        cluster_sentences[cluster_id].add(sent)
    
    print(f"Found {len(cluster_sentences)} clusters with sentences")
    
    # Convert sets to lists
    return {k: list(v) for k, v in cluster_sentences.items()}


def compute_concept_labels(cluster_sentences: dict, max_examples: int = 5) -> dict:
    """
    Create concept labels with representative sentences.
    """
    concept_labels = {}
    
    for cluster_id, sents in cluster_sentences.items():
        if not sents:
            continue
        
        concept_labels[cluster_id] = {
            "example_sentences": sents[:max_examples],
            "count": len(sents)
        }
    
    return concept_labels


def main():
    parser = argparse.ArgumentParser(
        description="Extract concept labels from cluster assignments"
    )
    parser.add_argument(
        "--clusters-file",
        type=str,
        required=True,
        help="Path to clusters-400.txt"
    )
    parser.add_argument(
        "--sentences-file",
        type=str,
        required=True,
        help="Path to sentences JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for concept_labels.json"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max example sentences per cluster (default: 5)"
    )
    
    args = parser.parse_args()
    
    print(f"Clusters: {args.clusters_file}")
    print(f"Sentences: {args.sentences_file}")
    
    cluster_sentences = extract_all_cluster_sentences(
        args.clusters_file, args.sentences_file
    )
    
    concept_labels = compute_concept_labels(cluster_sentences, args.max_examples)
    
    # Sort by cluster ID
    sorted_labels = {
        k: concept_labels[k] 
        for k in sorted(concept_labels.keys(), key=lambda x: int(x) if x.isdigit() else x)
    }
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_labels, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_path}")
    
    # Print stats
    print(f"\nTotal clusters: {len(sorted_labels)}")
    total_sents = sum(info['count'] for info in sorted_labels.values())
    print(f"Total unique sentences across all clusters: {total_sents}")
    
    # Print sample
    print("\nSample clusters:")
    for cluster_id in ['114', '141', '67', '91', '115']:
        if cluster_id in sorted_labels:
            info = sorted_labels[cluster_id]
            print(f"  Cluster {cluster_id}: {info['count']} sentences")
            if info['example_sentences']:
                print(f"    - {info['example_sentences'][0][:60]}...")


if __name__ == "__main__":
    main()

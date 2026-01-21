"""
Generate natural language explanations using LLM.

Uses sentence-level concept clusters ([CLS] clusters) and representative
sentences to generate human-readable explanations.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import re


def load_concept_labels(path: str) -> dict:
    """Load concept labels from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_input_data(path: str) -> list:
    """Load input data (sentences with labels)."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_clusters(clusters_file: str) -> dict:
    """
    Load sentence-to-cluster mapping from clusters file.
    
    Returns:
        dict: sentence_idx -> cluster_id (for [CLS] tokens)
    """
    sentence_clusters = {}
    
    with open(clusters_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|||')
            if len(parts) >= 2:
                token_info = parts[0]
                cluster_id = parts[-1]
                
                # [CLS] tokens have format [CLS]0, [CLS]1, etc.
                if token_info.startswith('[CLS]'):
                    match = re.match(r'\[CLS\](\d+)', token_info)
                    if match:
                        sent_idx = int(match.group(1))
                        sentence_clusters[sent_idx] = cluster_id
    
    return sentence_clusters


def build_explanation_prompt(
    sentence: str,
    label: int,
    cluster_id: str,
    concept_labels: dict
) -> str:
    """Build prompt for LLM explanation generation using sentence-level concepts."""
    
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    label_str = label_map.get(label, str(label))
    
    # Get example sentences from the same concept cluster
    cluster_info = concept_labels.get(cluster_id, {})
    example_sentences = cluster_info.get("example_sentences", [])[:3]
    cluster_size = cluster_info.get("count", 0)
    
    # Build examples text
    if example_sentences:
        examples_text = "\n".join([f"  - \"{s[:100]}...\"" if len(s) > 100 else f"  - \"{s}\"" 
                                   for s in example_sentences])
    else:
        examples_text = "  (no examples available)"
    print(examples_text)
    prompt = f"""You are an expert at explaining AI model predictions in natural language.

A sentiment classification model has made a prediction. The model uses latent concept clusters 
to organize sentences with similar characteristics.

**Sentence**: "{sentence}"
**Model Prediction**: {label_str}
**Concept Cluster**: #{cluster_id} ({cluster_size} similar sentences in training data)

**Example sentences from the same concept cluster:**
{examples_text}

Based on the concept cluster and similar sentences, generate a clear explanation (2-3 sentences) for:
1. What semantic pattern or theme this concept cluster represents
2. Why the model classified this sentence as {label_str}
3. How the sentence relates to other sentences in the same cluster

Explanation:"""

    return prompt


def call_openai(prompt: str, model: str = "gpt-3.5-turbo", api_key: str = None) -> str:
    """Call OpenAI API for explanation generation."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains AI model predictions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()


def call_ollama(prompt: str, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> str:
    """Call Ollama API for explanation generation."""
    import requests
    
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 300
            }
        },
        timeout=120
    )
    response.raise_for_status()
    
    return response.json().get("response", "").strip()


def generate_explanations(
    input_data: list,
    sentence_clusters: dict,
    concept_labels: dict,
    provider: str = "ollama",
    model: str = None,
    max_samples: int = None,
    api_key: str = None
) -> list:
    """Generate explanations for all samples."""
    
    results = []
    samples = input_data[:max_samples] if max_samples else input_data
    
    for idx, item in enumerate(samples):
        sentence = item.get("sentence", "")
        label = item.get("label", 0)
        
        # Get cluster ID for this sentence
        cluster_id = sentence_clusters.get(idx, "unknown")
        
        print(f"Processing {idx + 1}/{len(samples)}: Cluster {cluster_id}", flush=True)
        print(f"  Sentence: {sentence[:60]}...", flush=True)
        
        # Build prompt
        prompt = build_explanation_prompt(
            sentence, label, cluster_id, concept_labels
        )
        
        # Generate explanation
        try:
            if provider == "openai":
                explanation = call_openai(prompt, model or "gpt-3.5-turbo", api_key)
            else:
                explanation = call_ollama(prompt, model or "llama3.2")
        except Exception as e:
            print(f"  Error: {e}", flush=True)
            explanation = f"[Error: {e}]"
        
        # Get example sentences for output
        cluster_info = concept_labels.get(cluster_id, {})
        
        results.append({
            "id": item.get("id", idx),
            "sentence": sentence,
            "label": label,
            "cluster_id": cluster_id,
            "cluster_examples": cluster_info.get("example_sentences", [])[:3],
            "explanation": explanation
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM explanations for model predictions"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to input data JSON (sentences with labels)"
    )
    parser.add_argument(
        "--clusters-file",
        type=str,
        required=True,
        help="Path to clusters-400.txt file"
    )
    parser.add_argument(
        "--concept-labels",
        type=str,
        required=True,
        help="Path to concept_labels.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for explanations JSON"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gpt-3.5-turbo for openai, llama3.2 for ollama)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI (or use OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    print("Loading data...", flush=True)
    input_data = load_input_data(args.input_data)
    sentence_clusters = load_clusters(args.clusters_file)
    concept_labels = load_concept_labels(args.concept_labels)
    
    print(f"Loaded {len(input_data)} samples", flush=True)
    print(f"Loaded {len(sentence_clusters)} sentence-to-cluster mappings", flush=True)
    print(f"Loaded {len(concept_labels)} concept labels", flush=True)
    
    print(f"\nGenerating explanations using {args.provider}...", flush=True)
    results = generate_explanations(
        input_data,
        sentence_clusters,
        concept_labels,
        provider=args.provider,
        model=args.model,
        max_samples=args.max_samples,
        api_key=args.api_key
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(results)} explanations to: {output_path}", flush=True)
    
    # Print sample
    if results:
        print("\n=== Sample Output ===")
        sample = results[0]
        print(f"Sentence: {sample['sentence'][:80]}...")
        print(f"Cluster: {sample['cluster_id']}")
        print(f"Explanation: {sample['explanation'][:200]}...")


if __name__ == "__main__":
    main()

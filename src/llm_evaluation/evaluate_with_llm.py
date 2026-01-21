"""
Evaluate explanations using LLM as annotator.

This script evaluates generated explanations on:
- Faithfulness: Does explanation reflect model's reasoning?
- Plausibility: Is explanation convincing to humans?
- Concept Coherence: Are concepts semantically meaningful?
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import re


def load_explanations(path: str) -> list:
    """Load explanations JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_evaluation_prompt(explanation_item: dict) -> str:
    """Build prompt for LLM evaluation using sentence-level concepts."""
    
    sentence = explanation_item.get("sentence", "")
    label = explanation_item.get("label", 0)
    cluster_id = explanation_item.get("cluster_id", "unknown")
    cluster_examples = explanation_item.get("cluster_examples", [])
    explanation = explanation_item.get("explanation", "")
    
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    label_str = label_map.get(label, str(label))
    
    # Format cluster examples
    if cluster_examples:
        examples_text = "\n".join([f"  - \"{ex[:80]}...\"" if len(ex) > 80 else f"  - \"{ex}\"" 
                                    for ex in cluster_examples[:3]])
    else:
        examples_text = "  (no examples available)"
    
    prompt = f"""You are an expert evaluator of AI explanation quality.

Evaluate the following explanation based on three criteria. Rate each from 1-5.

**Original Sentence**: "{sentence}"
**True Label**: {label_str}
**Concept Cluster**: #{cluster_id}

**Similar sentences from the same concept cluster:**
{examples_text}

**Generated Explanation**:
"{explanation}"

Rate the explanation on these criteria:

1. **Faithfulness** (1-5): Does the explanation accurately describe why sentences in this cluster share common characteristics?
   - 1: Completely inaccurate or fabricated
   - 5: Accurately describes the concept cluster's theme

2. **Plausibility** (1-5): Is the explanation convincing and logical to a human reader?
   - 1: Nonsensical or illogical
   - 5: Clear, logical, and convincing

3. **Concept Coherence** (1-5): Does the concept cluster make semantic sense? Are the example sentences truly related?
   - 1: Examples seem random/unrelated
   - 5: Examples form a coherent semantic theme

Respond ONLY in this exact JSON format:
{{
  "faithfulness": <score>,
  "faithfulness_reason": "<brief reason>",
  "plausibility": <score>,
  "plausibility_reason": "<brief reason>",
  "concept_coherence": <score>,
  "concept_coherence_reason": "<brief reason>"
}}"""

    return prompt


def parse_evaluation_response(response: str) -> dict:
    """Parse LLM evaluation response into structured scores."""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract scores manually
    result = {
        "faithfulness": 3,
        "plausibility": 3,
        "concept_coherence": 3,
        "parse_error": True
    }
    
    for metric in ["faithfulness", "plausibility", "concept_coherence"]:
        match = re.search(rf'{metric}["\s:]+(\d)', response.lower())
        if match:
            result[metric] = int(match.group(1))
    
    return result


def call_openai(prompt: str, model: str = "gpt-3.5-turbo", api_key: str = None) -> str:
    """Call OpenAI API for evaluation."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert evaluator. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()


def call_ollama(prompt: str, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> str:
    """Call Ollama API for evaluation."""
    import requests
    
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 500
            }
        },
        timeout=120
    )
    response.raise_for_status()
    
    return response.json().get("response", "").strip()


def evaluate_explanations(
    explanations: list,
    provider: str = "ollama",
    model: str = None,
    max_samples: int = None,
    api_key: str = None
) -> list:
    """Evaluate all explanations."""
    
    results = []
    samples = explanations[:max_samples] if max_samples else explanations
    
    for idx, item in enumerate(samples):
        print(f"Evaluating {idx + 1}/{len(samples)}: Cluster {item.get('cluster_id', 'unknown')}", flush=True)
        
        # Build prompt
        prompt = build_evaluation_prompt(item)
        
        # Get evaluation
        try:
            if provider == "openai":
                response = call_openai(prompt, model or "gpt-3.5-turbo", api_key)
            else:
                response = call_ollama(prompt, model or "llama3.2")
            
            scores = parse_evaluation_response(response)
        except Exception as e:
            print(f"  Error: {e}", flush=True)
            scores = {
                "faithfulness": 0,
                "plausibility": 0,
                "concept_coherence": 0,
                "error": str(e)
            }
        
        results.append({
            "id": item.get("id", idx),
            "sentence": item.get("sentence", "")[:100],
            "cluster_id": item.get("cluster_id", "unknown"),
            "scores": scores
        })
    
    return results


def compute_summary_stats(results: list) -> dict:
    """Compute summary statistics."""
    valid_results = [r for r in results if "error" not in r["scores"]]
    
    if not valid_results:
        return {"error": "No valid evaluations"}
    
    metrics = ["faithfulness", "plausibility", "concept_coherence"]
    stats = {}
    
    for metric in metrics:
        scores = [r["scores"].get(metric, 0) for r in valid_results]
        stats[metric] = {
            "mean": round(sum(scores) / len(scores), 2),
            "min": min(scores),
            "max": max(scores)
        }
    
    # Overall score
    all_scores = []
    for r in valid_results:
        avg = sum(r["scores"].get(m, 0) for m in metrics) / len(metrics)
        all_scores.append(avg)
    
    stats["overall"] = {
        "mean": round(sum(all_scores) / len(all_scores), 2),
        "count": len(valid_results)
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate explanations using LLM as annotator"
    )
    parser.add_argument(
        "--explanations",
        type=str,
        required=True,
        help="Path to explanations JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for evaluation results"
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
        help="Model name"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI"
    )
    
    args = parser.parse_args()
    
    print("Loading explanations...", flush=True)
    explanations = load_explanations(args.explanations)
    print(f"Loaded {len(explanations)} explanations", flush=True)
    
    print(f"Evaluating using {args.provider}...", flush=True)
    results = evaluate_explanations(
        explanations,
        provider=args.provider,
        model=args.model,
        max_samples=args.max_samples,
        api_key=args.api_key
    )
    
    # Compute summary
    summary = compute_summary_stats(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "summary": summary,
        "evaluations": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved evaluation results to: {output_path}", flush=True)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    for metric in ["faithfulness", "plausibility", "concept_coherence"]:
        if metric in summary:
            print(f"  {metric}: {summary[metric]['mean']:.2f} (range: {summary[metric]['min']}-{summary[metric]['max']})")
    
    if "overall" in summary:
        print(f"  Overall: {summary['overall']['mean']:.2f} ({summary['overall']['count']} samples)")


if __name__ == "__main__":
    main()

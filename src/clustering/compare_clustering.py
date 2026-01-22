"""
Compare clustering quality using Silhouette Score and Calinski-Harabasz Index.
Reads points and cluster labels from different methods (Agglomerative vs K-Means).
"""

import argparse
import numpy as np
import os
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time

def load_labels(label_file):
    """Load cluster labels from text file."""
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return np.array(labels)

def evaluate_clustering(points, labels, name="Method"):
    """Calculate and print clustering metrics."""
    print(f"\n--- Evaluating {name} ---")
    start_time = time.time()
    
    # Check if labels size matches points
    if len(labels) != len(points):
        print(f"Error: Label size ({len(labels)}) != Points size ({len(points)})")
        return None
    
    # 1. Silhouette Score (ranges from -1 to 1, higher is better)
    # Note: Expensive for large datasets, use sample_size for approximation
    print("Calculating Silhouette Score...")
    sil = silhouette_score(points, labels, metric='euclidean', sample_size=10000, random_state=42)
    print(f"Silhouette Score: {sil:.4f} (higher is better)")
    
    # 2. Calinski-Harabasz Index (higher is better)
    print("Calculating Calinski-Harabasz Index...")
    ch = calinski_harabasz_score(points, labels)
    print(f"Calinski-Harabasz Index: {ch:.4f} (higher is better)")
    
    # 3. Davies-Bouldin Index (lower is better)
    print("Calculating Davies-Bouldin Index...")
    db = davies_bouldin_score(points, labels)
    print(f"Davies-Bouldin Index: {db:.4f} (lower is better)")
    
    print(f"Evaluation time: {time.time() - start_time:.2f}s")
    return {"silhouette": sil, "calinski": ch, "davies": db}

def main():
    parser = argparse.ArgumentParser(description="Compare clustering methods")
    parser.add_argument("--point-file", required=True, help="Path to processed-point.npy")
    parser.add_argument("--agg-labels", help="Path to Agglomerative labels file (labels-400.txt)")
    parser.add_argument("--kmeans-labels", help="Path to K-Means labels file (labels-400.txt)")
    
    args = parser.parse_args()
    
    print(f"Loading points from {args.point_file}...")
    points = np.load(args.point_file)
    print(f"Points shape: {points.shape}")
    
    results = {}
    
    if args.agg_labels and os.path.exists(args.agg_labels):
        agg_labels = load_labels(args.agg_labels)
        results["Agglomerative"] = evaluate_clustering(points, agg_labels, "Agglomerative Clustering")
        
    if args.kmeans_labels and os.path.exists(args.kmeans_labels):
        kmeans_labels = load_labels(args.kmeans_labels)
        results["K-Means"] = evaluate_clustering(points, kmeans_labels, "K-Means Clustering")
        
    # Summary Comparison
    if len(results) > 1:
        print("\n==================================")
        print("FINAL COMPARISON")
        print("==================================")
        print(f"{'Metric':<25} {'Agglomerative':<15} {'K-Means':<15} {'Winner'}")
        print("-" * 65)
        
        metrics = [
            ("Silhouette (Higher)", "silhouette", lambda x, y: x > y),
            ("Calinski-H (Higher)", "calinski", lambda x, y: x > y),
            ("Davies-B (Lower)", "davies", lambda x, y: x < y)
        ]
        
        for name, key, comparator in metrics:
            val_agg = results["Agglomerative"][key]
            val_km = results["K-Means"][key]
            winner = "K-Means" if comparator(val_km, val_agg) else "Agglomerative"
            print(f"{name:<25} {val_agg:<15.4f} {val_km:<15.4f} {winner}")

if __name__ == "__main__":
    main()

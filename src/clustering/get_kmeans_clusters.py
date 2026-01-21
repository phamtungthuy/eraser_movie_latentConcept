"""
K-Means Clustering for Latent Concept Discovery.

This script provides K-Means clustering as an alternative to Agglomerative Clustering.
Based on the Latent_Concept_Analysis repository (EACL 2024).

Usage:
    python get_kmeans_clusters.py \
        --vocab-file /path/to/processed-vocab.npy \
        --point-file /path/to/processed-point.npy \
        --output-path /path/to/results \
        --cluster 400,400,400

K-Means advantages over Agglomerative:
- Faster for large datasets (O(n*k*i) vs O(n^2) or O(n^3))
- Better scalability
- Can use mini-batch version for very large datasets
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict
import time
import argparse
import dill as pickle


def parse_cluster_args(cluster_str, use_range=False):
    """Parse cluster argument string into list of K values."""
    Ks = [int(k) for k in cluster_str.split(',')]
    print(f"Cluster input: {Ks}")
    
    if use_range and len(Ks) == 3:
        # Format: start, end, step
        Ks = list(range(Ks[0], Ks[1] + 1, Ks[2]))
    
    print(f"Run for the following cluster sizes: {Ks}")
    return Ks


def run_kmeans_clustering(
    points: np.ndarray,
    vocab: np.ndarray,
    n_clusters: int,
    output_path: str,
    use_minibatch: bool = False,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300
):
    """
    Run K-Means clustering and save results.
    
    Parameters
    ----------
    points : np.ndarray
        Point embeddings of shape (n_samples, n_features)
    vocab : np.ndarray
        Vocabulary array containing word identifiers
    n_clusters : int
        Number of clusters to create
    output_path : str
        Directory to save output files
    use_minibatch : bool
        Use MiniBatchKMeans for large datasets (faster but approximate)
    random_state : int
        Random seed for reproducibility
    n_init : int
        Number of initializations to try
    max_iter : int
        Maximum iterations per initialization
    """
    starttime = time.time()
    print(f"Performing K-Means clustering with K={n_clusters}...")
    
    # Choose clustering algorithm
    if use_minibatch:
        print("Using MiniBatchKMeans (faster for large datasets)")
        clustering = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            batch_size=1024
        ).fit(points)
    else:
        clustering = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter
        ).fit(points)
    
    # Save the model
    model_fn = f"{output_path}/model-{n_clusters}-kmeans-clustering.pkl"
    with open(model_fn, "wb") as fp:
        pickle.dump(clustering, fp)
    print(f"Model saved to: {model_fn}")
    
    # Save cluster centers (useful for analysis)
    centers_fn = f"{output_path}/centers-{n_clusters}.npy"
    np.save(centers_fn, clustering.cluster_centers_)
    print(f"Cluster centers saved to: {centers_fn}")
    
    # Organize words by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(vocab[i])
    
    # Save cluster labels
    labels_fn = f"{output_path}/labels-{n_clusters}.txt"
    with open(labels_fn, 'w') as f:
        for label in clustering.labels_:
            f.write(str(label) + '\n')
    print(f"Labels saved to: {labels_fn}")
    
    # Save cluster assignments (Word|||ClusterID format)
    clusters_fn = f"{output_path}/clusters-{n_clusters}.txt"
    with open(clusters_fn, 'w') as target:
        for key in clusters.keys():
            for word in clusters[key]:
                target.write(f"{word}|||{key}\n")
    print(f"Clusters saved to: {clusters_fn}")
    
    # Print statistics
    endtime = time.time()
    diff = endtime - starttime
    
    print(f"\n--- K={n_clusters} Clustering Statistics ---")
    print(f"Inertia (within-cluster sum of squares): {clustering.inertia_:.4f}")
    print(f"Number of iterations: {clustering.n_iter_}")
    print(f"Time taken: {diff:.2f} seconds")
    
    # Cluster size distribution
    unique, counts = np.unique(clustering.labels_, return_counts=True)
    print(f"Cluster size - min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")
    print("-" * 40)
    
    return clustering


def main():
    parser = argparse.ArgumentParser(
        description="K-Means Clustering for Latent Concept Discovery"
    )
    parser.add_argument(
        "--vocab-file", "-v",
        required=True,
        help="Path to vocab numpy file (processed-vocab.npy)"
    )
    parser.add_argument(
        "--point-file", "-p",
        required=True,
        help="Path to point numpy file (processed-point.npy)"
    )
    parser.add_argument(
        "--output-path", "-o",
        required=True,
        help="Output path for clustering model and result files"
    )
    parser.add_argument(
        "--cluster", "-k",
        required=True,
        help="Cluster numbers comma separated (e.g. 5,10,15) or range (e.g. 100,500,100)"
    )
    parser.add_argument(
        "--range", "-r",
        type=bool,
        default=False,
        help="Interpret cluster as range: start,end,step"
    )
    parser.add_argument(
        "--minibatch",
        action="store_true",
        default=False,
        help="Use MiniBatchKMeans for faster clustering on large datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of initializations (default: 10)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum iterations per initialization (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Parse cluster sizes
    Ks = parse_cluster_args(args.cluster, args.range)
    
    # Load data
    print(f"\nLoading vocab from: {args.vocab_file}")
    vocab = np.load(args.vocab_file)
    print(f"Vocab size: {len(vocab)}")
    
    print(f"Loading points from: {args.point_file}")
    points = np.load(args.point_file)
    print(f"Points shape: {points.shape}")
    
    # Run clustering for each K
    for K in Ks:
        run_kmeans_clustering(
            points=points,
            vocab=vocab,
            n_clusters=K,
            output_path=args.output_path,
            use_minibatch=args.minibatch,
            random_state=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter
        )
    
    print("\nAll clustering complete!")


if __name__ == "__main__":
    main()

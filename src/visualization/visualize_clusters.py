"""
Visualize latent concept clusters using Word Clouds (Layer 0) or Example Sentences (Layer 12).
"""

import argparse
import os
import math
import json
import textwrap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm
from collections import Counter
import numpy as np

def load_clusters(cluster_file, top_k_clusters=20):
    """Load clusters and return sorted list of (cluster_id, words)."""
    cluster_words = {}
    print(f"Loading clusters from {cluster_file}...")
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('|||')
            if len(parts) >= 2:
                word = parts[0].strip()
                if word in ['<s>', '</s>', '[CLS]', '[SEP]', '[PAD]'] or word.startswith('__'): continue
                cluster_id = int(parts[-1])
                if cluster_id not in cluster_words: cluster_words[cluster_id] = []
                cluster_words[cluster_id].append(word)
    return sorted(cluster_words.items(), key=lambda x: len(x[1]), reverse=True)[:top_k_clusters]

def load_concept_labels(json_file, top_k_clusters=20):
    """Load concept labels/sentences from JSON."""
    print(f"Loading concept labels from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Sort by count
    sorted_clusters = sorted(data.items(), key=lambda x: x[1].get('count', 0), reverse=True)
    return sorted_clusters[:top_k_clusters] # List of (cluster_id, info_dict)

def generate_wordclouds(clusters_data, output_file, grid_cols=4):
    """Generate grid of word clouds."""
    n_clusters = len(clusters_data)
    grid_rows = math.ceil(n_clusters / grid_cols)
    plt.figure(figsize=(5 * grid_cols, 4 * grid_rows))
    
    print("Generating word clouds...")
    for idx, (cluster_id, words) in enumerate(tqdm(clusters_data)):
        word_freq = Counter(words)
        wc = WordCloud(width=400, height=300, background_color='white', max_words=50, colormap='Dark2').generate_from_frequencies(word_freq)
        
        ax = plt.subplot(grid_rows, grid_cols, idx + 1)
        ax.imshow(np.array(wc.to_image()), interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Cluster #{cluster_id}\n({len(words)} tokens)", fontsize=12)
        for spine in ax.spines.values(): spine.set_edgecolor('lightgray'); spine.set_linewidth(1)

    print(f"Saving visualization to {output_file}...")
    plt.tight_layout(pad=3.0)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def generate_sentence_view(clusters_data, output_file, grid_cols=3):
    """Generate grid of example sentences."""
    n_clusters = len(clusters_data)
    grid_rows = math.ceil(n_clusters / grid_cols)
    
    # Larger figure for text
    plt.figure(figsize=(8 * grid_cols, 6 * grid_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    print("Generating sentence views...")
    for idx, (cluster_id, info) in enumerate(tqdm(clusters_data)):
        sentences = info.get("example_sentences", [])[:5] # Take top 5
        count = info.get("count", 0)
        
        ax = plt.subplot(grid_rows, grid_cols, idx + 1)
        ax.axis('off')
        
        import matplotlib.patches as patches
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='#f0f0f0', edgecolor='lightgray', transform=ax.transAxes))
        
        # Title
        ax.text(0.05, 0.95, f"Cluster #{cluster_id} ({count} items)", transform=ax.transAxes, 
                fontsize=14, fontweight='bold', va='top', color='#333')
        
        # Sentences
        text_content = ""
        for i, sent in enumerate(sentences):
            # Truncate and wrap
            wrapped = textwrap.shorten(sent, width=200, placeholder="...")
            wrapped_lines = textwrap.fill(wrapped, width=60)
            text_content += f"• {wrapped_lines}\n\n"
            
        ax.text(0.05, 0.85, text_content, transform=ax.transAxes, 
                fontsize=10, va='top', family='monospace', color='#444')

    print(f"Saving visualization to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def save_individual_clusters(clusters_data, output_dir, mode='wordcloud'):
    """Save each cluster as a separate image."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving individual cluster images to {output_dir} ({mode})...")
    
    for cluster_id, data in tqdm(clusters_data):
        plt.figure(figsize=(6, 4))
        
        if mode == 'wordcloud':
            words = data # data is list of words
            wc = WordCloud(width=600, height=400, background_color='white', max_words=50, colormap='Dark2').generate_from_frequencies(Counter(words))
            plt.imshow(np.array(wc.to_image()), interpolation='bilinear')
            plt.axis('off')
            # plt.title(f"Cluster #{cluster_id} (WordCloud)", fontsize=14)
            filename = f"cluster_{cluster_id}_wc.png"
            
        elif mode == 'sentences':
            info = data # data is dict
            sentences = info.get("example_sentences", [])[:5]
            count = info.get("count", 0)
            
            ax = plt.gca()
            ax.axis('off')
            
            # Background
            import matplotlib.patches as patches
            rect = patches.Rectangle((0, 0), 1, 1, facecolor='#f8f9fa', edgecolor='#dee2e6', transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Title removed as requested
            # ax.text(0.05, 0.92, f"Cluster #{cluster_id} ({count} items)", transform=ax.transAxes, 
            #        fontsize=16, fontweight='bold', va='top', color='#212529')
            
            # Text
            text_content = ""
            for sent in sentences:
                wrapped = textwrap.shorten(sent, width=200, placeholder="...")
                wrapped_lines = textwrap.fill(wrapped, width=50) # Tighter wrap for single image
                text_content += f"• {wrapped_lines}\n\n"
            
            # Adjusted vertical position since title is gone
            ax.text(0.05, 0.95, text_content, transform=ax.transAxes, 
                    fontsize=11, va='top', family='monospace', color='#495057')
            filename = f"cluster_{cluster_id}_sentences.png"
            
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize clusters")
    parser.add_argument("--mode", choices=['wordcloud', 'sentences'], default='wordcloud', help="Visualization mode")
    parser.add_argument("--clusters-file", type=str, help="Path to clusters.txt (for wordcloud)")
    parser.add_argument("--json-file", type=str, help="Path to concept_labels.json (for sentences)")
    parser.add_argument("--output-file", type=str, required=True, help="Output image path (grid)")
    parser.add_argument("--save-individual", action="store_true", help="Save individual images for each cluster")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    
    args = parser.parse_args()
    
    clusters = None
    if args.mode == 'wordcloud':
        if not args.clusters_file:
            print("Error: --clusters-file required for wordcloud mode")
            return
        clusters = load_clusters(args.clusters_file, args.top_k)
        if clusters:
            generate_wordclouds(clusters, args.output_file, args.cols)
            if args.save_individual:
                output_dir = os.path.splitext(args.output_file)[0] + "_individual"
                save_individual_clusters(clusters, output_dir, mode='wordcloud')
            
    elif args.mode == 'sentences':
        if not args.json_file:
            print("Error: --json-file required for sentences mode")
            return
        clusters = load_concept_labels(args.json_file, args.top_k)
        if clusters:
            generate_sentence_view(clusters, args.output_file, args.cols)
            if args.save_individual:
                output_dir = os.path.splitext(args.output_file)[0] + "_individual"
                save_individual_clusters(clusters, output_dir, mode='sentences')

if __name__ == "__main__":
    main()

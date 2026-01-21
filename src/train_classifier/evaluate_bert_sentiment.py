"""
Evaluate a BERT Sentiment Classification model on a test dataset.

This script evaluates either a base BERT model or a fine-tuned model
on a test dataset (e.g., movie_dev_subset.json).

Usage:
    # Evaluate fine-tuned model
    python evaluate_bert_sentiment.py \
        --model-path trained_models/bert_sentiment \
        --test-file data/movie_dev_subset.json

    # Evaluate base model (to compare)
    python evaluate_bert_sentiment.py \
        --model-path google-bert/bert-base-cased \
        --test-file data/movie_dev_subset.json
"""

import argparse
import json
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


class MovieSentimentDataset(Dataset):
    """Dataset class for Movie Sentiment data."""
    
    def __init__(self, data: List[dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence']
        label = item['label']
        
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'sentence': sentence
        }


def load_data(file_path: str) -> List[dict]:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def deduplicate_data(data: List[dict]) -> List[dict]:
    """Remove duplicate sentences, keeping only unique sentence-label pairs."""
    seen = set()
    unique_data = []
    for item in data:
        key = (item['sentence'], item['label'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    print(f"Deduplicated {len(data)} samples to {len(unique_data)} unique samples")
    return unique_data


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[List[int], List[int], List[str]]:
    """Evaluate the model and return predictions, labels, and sentences."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_sentences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            sentences = batch['sentence']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_sentences.extend(sentences)
    
    return all_predictions, all_labels, all_sentences


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT sentiment classification model")
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the trained model or name of base model (e.g., google-bert/bert-base-cased)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        required=True,
        help='Path to the test JSON file (e.g., movie_dev_subset.json)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--show-errors',
        action='store_true',
        help='Show misclassified examples'
    )
    parser.add_argument(
        '--num-errors',
        type=int,
        default=10,
        help='Number of misclassified examples to show'
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading test data from {args.test_file}")
    data = load_data(args.test_file)
    data = deduplicate_data(data)
    
    # Load tokenizer and model
    print(f"\nLoading model from: {args.model_path}")
    try:
        # Try loading as a local fine-tuned model first
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        print("Loaded fine-tuned model successfully!")
    except Exception as e:
        print(f"Could not load as local model, trying as HuggingFace model...")
        # If that fails, load as a base model with random classification head
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=2,
            id2label={0: 'negative', 1: 'positive'},
            label2id={'negative': 0, 'positive': 1}
        )
        print("WARNING: Loaded base model with RANDOM classification head!")
        print("         Predictions will be essentially random!")
    
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = MovieSentimentDataset(data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    print(f"\nEvaluating on {len(dataset)} samples...")
    predictions, labels, sentences = evaluate(model, dataloader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nModel: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Number of samples: {len(labels)}")
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    target_names = ['Negative (0)', 'Positive (1)']
    print(classification_report(labels, predictions, target_names=target_names))
    
    print("-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(labels, predictions)
    print(f"                Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Pos   {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Show misclassified examples
    if args.show_errors:
        print("\n" + "-"*60)
        print(f"Misclassified Examples (showing first {args.num_errors}):")
        print("-"*60)
        
        error_count = 0
        for i, (pred, label, sent) in enumerate(zip(predictions, labels, sentences)):
            if pred != label and error_count < args.num_errors:
                pred_label = "Positive" if pred == 1 else "Negative"
                true_label = "Positive" if label == 1 else "Negative"
                print(f"\n[{error_count + 1}] Predicted: {pred_label}, True: {true_label}")
                print(f"    Sentence: {sent[:100]}{'...' if len(sent) > 100 else ''}")
                error_count += 1
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

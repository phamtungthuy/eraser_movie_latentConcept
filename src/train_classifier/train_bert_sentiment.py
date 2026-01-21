"""
Fine-tune BERT for Sentiment Classification on Movie Reviews Dataset.

This script trains a BertForSequenceClassification model on the movie_train.json dataset.
The trained model can then be used by the LACOAT pipeline for generating explanations.

Usage:
    python train_bert_sentiment.py \
        --train-file /path/to/movie_train.json \
        --model-name google-bert/bert-base-cased \
        --output-dir /path/to/output \
        --epochs 3 \
        --batch-size 16
"""

import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MovieSentimentDataset(Dataset):
    """Dataset class for Movie Sentiment data."""
    
    def __init__(self, data: List[dict], tokenizer: BertTokenizer, max_length: int = 512):
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
            'labels': torch.tensor(label, dtype=torch.long)
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


def train_epoch(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment classification")
    
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Path to the training JSON file (e.g., movie_train.json)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='google-bert/bert-base-cased',
        help='Name or path of the base BERT model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.train_file}")
    data = load_data(args.train_file)
    data = deduplicate_data(data)
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: 'negative', 1: 'positive'},
        label2id={'negative': 0, 'positive': 1}
    )
    model.to(device)
    
    # Create dataset
    dataset = MovieSentimentDataset(data, tokenizer, max_length=args.max_length)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best validation accuracy! Saving model...")
            
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"\nTo use this model with LACOAT, update config.env:")
    print(f"MODEL={args.output_dir}")


if __name__ == '__main__':
    main()

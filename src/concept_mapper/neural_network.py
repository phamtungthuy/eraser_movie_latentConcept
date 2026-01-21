import argparse
import ast
import os

import dill as pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import time


class ConceptDataset(Dataset):
    """Custom Dataset for concept mapping"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for concept mapping"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_classifier(train_df, num_epochs=50, batch_size=128, learning_rate=0.001, 
                     hidden_dims=[512, 256, 128], dropout=0.3, device='cuda'):
    """
    Train the Neural Network Classifier

    Parameters
    ----------
    train_df : pandas.DataFrame
        The dataframe of the train dataset
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimizer
    hidden_dims : list
        List of hidden layer dimensions
    dropout : float
        Dropout rate
    device : str
        Device to train on ('cuda' or 'cpu')

    Returns
    -------
    model : MLPClassifier
        The trained classifier
    label_encoder : LabelEncoder
        The label encoder for classes
    """
    # Prepare data
    X_train = train_df['embedding']
    X_train = np.array([ast.literal_eval(val) for val in X_train])
    y_train = train_df['cluster_idx'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Get dimensions
    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    # Create dataset and dataloader
    train_dataset = ConceptDataset(X_train, y_train_encoded)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim, hidden_dims, num_classes, dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return model, label_encoder


def save_classifier(model, label_encoder, layer, output_path):
    """
    Save the Neural Network Classifier

    Parameters
    ----------
    model : MLPClassifier
        The classifier to save
    label_encoder : LabelEncoder
        The label encoder
    layer : int
        The layer of the csv file
    output_path : str
        The path to save the classifier

    Returns
    -------
    None
    """
    filename = output_path + 'layer_' + str(layer) + '_nn_classifier.pkl'
    
    with open(filename, 'wb') as file:
        pickle.dump({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': model.network[0].in_features,
                'hidden_dims': [layer.out_features for layer in model.network if isinstance(layer, nn.Linear)][:-1],
                'num_classes': model.network[-1].out_features,
            },
            'label_encoder': label_encoder
        }, file)


def load_classifier(filepath, device='cuda'):
    """Load a saved classifier"""
    with open(filepath, 'rb') as file:
        checkpoint = pickle.load(file)
    
    config = checkpoint['model_config']
    model = MLPClassifier(
        config['input_dim'],
        config['hidden_dims'],
        config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch.device(device if torch.cuda.is_available() else 'cpu'))
    
    return model, checkpoint['label_encoder']


def format_predictions(model, label_encoder, X, df, device='cuda'):
    """
    Format the predictions of the Neural Network Classifier

    Parameters
    ----------
    model : MLPClassifier
        The classifier
    label_encoder : LabelEncoder
        The label encoder
    X : numpy.ndarray
        The input features
    df : pandas.DataFrame
        The dataframe of the dataset

    Returns
    -------
    pred_df : pandas.DataFrame
        The predictions of the classifier
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    
    # Get classes
    classes = label_encoder.classes_
    
    # Make a dataframe for the predictions
    pred_df = pd.DataFrame(columns=['Token', 'line_idx', 'position_idx', 'Top 1', 'Top 2', 'Top 5'])
    
    # Tokens
    tokens = df['token'].values
    line_idx = df['line_idx'].values
    position_idx = df['position_idx'].values
    
    # Get the top 1, 2, 5 predictions
    top1 = []
    top2 = []
    top5 = []
    top5_probs = []
    
    for i in range(len(probs)):
        # Sort the probabilities in increasing order
        sorted_index = np.argsort(probs[i])
        
        # Get the top predictions
        top1.append(classes[sorted_index[-1]])
        top2.append(classes[sorted_index[-2:]])
        top5.append(classes[sorted_index[-5:]])
        top5_probs.append(np.round(probs[i][sorted_index[-5:]], 2))
    
    # Add the predictions to the dataframe
    pred_df['Token'] = tokens
    pred_df['line_idx'] = line_idx
    pred_df['position_idx'] = position_idx
    pred_df['Top 1'] = top1
    pred_df['Top 2'] = top2
    pred_df['Top 5'] = top5
    pred_df['Top 5 Probabilities'] = top5_probs
    
    return pred_df


def validate_classifier(validate_df, model, label_encoder, layer, output_path, device='cuda'):
    """
    Validate the Neural Network Classifier

    Parameters
    ----------
    validate_df : pandas.DataFrame
        The dataframe of the validate dataset
    model : MLPClassifier
        The classifier to validate
    label_encoder : LabelEncoder
        The label encoder
    layer : int
        The layer of the csv file
    output_path : str
        The path to save the predictions
    device : str
        Device to use

    Returns
    -------
    accuracy : float
        The accuracy of the classifier
    """
    # Validate
    X_validate = validate_df['embedding']
    X_validate = np.array([ast.literal_eval(val) for val in X_validate])
    y_validate = validate_df['cluster_idx'].values
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_validate).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
    
    # Decode predictions
    predicted_labels = label_encoder.inverse_transform(predicted)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == y_validate)
    
    # Format predictions
    df = format_predictions(model, label_encoder, X_validate, validate_df, device)
    
    # Add the Actual column at the index 1
    df.insert(1, 'Actual', y_validate)
    
    # Save the predictions
    df.to_csv(output_path + 'predictions_layer_' + str(layer) + '.csv', sep='\t', index=False)
    
    return accuracy


def predict_classifier(test_df, model, label_encoder, layer, output_path, device='cuda'):
    """
    Predict using the Neural Network Classifier

    Parameters
    ----------
    test_df : pandas.DataFrame
        The dataframe of the test dataset
    model : MLPClassifier
        The classifier to predict
    label_encoder : LabelEncoder
        The label encoder
    layer : int
        The layer of the csv file
    output_path : str
        The path to save predictions
    device : str
        Device to use

    Returns
    -------
    predictions : pandas.DataFrame
        The predictions of the classifier
    """
    # Predict
    X_test = test_df['embedding']
    X_test = np.array([ast.literal_eval(val) for val in X_test])
    
    df = format_predictions(model, label_encoder, X_test, test_df, device)
    df.to_csv(output_path + 'predictions_layer_' + str(layer) + '.csv', sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, help='train cluster csv file path')
    parser.add_argument('--validate_file_path', type=str, help='validate cluster csv file path')
    parser.add_argument('--test_file_path', type=str, help='test cluster csv file path')
    parser.add_argument('--classifier_file_path', type=str, help='path to saved classifier')
    parser.add_argument('--layer', default=12, help='the selected layer number')
    parser.add_argument('--save_path', type=str, help='save classification result path')
    parser.add_argument('--do_train', action="store_true", help='whether to train the classifier')
    parser.add_argument('--do_validate', action="store_true", help='whether to validate the classifier')
    parser.add_argument('--do_predict', action="store_true", help='whether to predict the classifier')
    parser.add_argument('--load_classifier_from_local', action="store_true", 
                       help='whether to load the classifier from local')
    
    # Neural network specific arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_dims', type=str, default='512,256,128', 
                       help='hidden layer dimensions (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--device', type=str, default='cuda', help='device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    if args.load_classifier_from_local:
        model, label_encoder = load_classifier(args.classifier_file_path, args.device)
    else:
        model = None
        label_encoder = None
    
    if args.do_train:
        train_df = pd.read_csv(args.train_file_path, sep='\t')
        model, label_encoder = train_classifier(
            train_df, 
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            device=args.device
        )
        
        save_path = args.save_path + '/model/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        save_classifier(model, label_encoder, args.layer, save_path)
    
    if args.do_validate:
        validate_df = pd.read_csv(args.validate_file_path, sep='\t')
        
        save_path = args.save_path + '/validate_predictions/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        accuracy = validate_classifier(validate_df, model, label_encoder, args.layer, save_path, args.device)
        print('Validation Accuracy: ', accuracy)
    
    if args.do_predict:
        test_df = pd.read_csv(args.test_file_path, sep='\t')
        
        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        predict_classifier(test_df, model, label_encoder, args.layer, save_path, args.device)
    
    end_time = time.time()
    print("Time taken: ", end_time - start_time)


if __name__ == '__main__':
    main()

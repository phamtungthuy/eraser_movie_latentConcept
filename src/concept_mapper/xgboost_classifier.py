import argparse
import ast
import os

import dill as pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import time


def train_classifier(train_df, n_estimators=200, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0,
                     reg_lambda=1, n_jobs=-1, early_stopping_rounds=10):
    """
    Train the XGBoost Classifier

    Parameters
    ----------
    train_df : pandas.DataFrame
        The dataframe of the train dataset
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum depth of a tree
    learning_rate : float
        Boosting learning rate
    subsample : float
        Subsample ratio of the training instances
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree
    gamma : float
        Minimum loss reduction required to make a further partition
    reg_alpha : float
        L1 regularization term on weights
    reg_lambda : float
        L2 regularization term on weights
    n_jobs : int
        Number of parallel threads
    early_stopping_rounds : int
        Activates early stopping

    Returns
    -------
    classifier : xgb.XGBClassifier
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
    
    # Initialize classifier
    classifier = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        random_state=42,
        tree_method='hist',  # Use histogram-based algorithm (GPU not available)
        eval_metric='mlogloss',
        verbosity=1
    )
    
    print("Training XGBoost Classifier with hist method (CPU)...")
    
    # Split a small portion for early stopping validation
    split_idx = int(0.9 * len(X_train))
    X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val_split = y_train_encoded[:split_idx], y_train_encoded[split_idx:]
    
    # Train with early stopping
    classifier.fit(
        X_train_split, 
        y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=True
    )
    
    # Print feature importance statistics
    feature_importance = classifier.feature_importances_
    print(f"Feature importance - Mean: {feature_importance.mean():.4f}, Std: {feature_importance.std():.4f}")
    print(f"Best iteration: {classifier.best_iteration}")
    print(f"Best score: {classifier.best_score:.4f}")
    
    return classifier, label_encoder


def save_classifier(classifier, label_encoder, layer, output_path):
    """
    Save the XGBoost Classifier

    Parameters
    ----------
    classifier : xgb.XGBClassifier
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
    filename = output_path + 'layer_' + str(layer) + '_xgb_classifier.pkl'
    
    with open(filename, 'wb') as file:
        pickle.dump({
            'classifier': classifier,
            'label_encoder': label_encoder
        }, file)
    
    print(f"Classifier saved to {filename}")


def load_classifier(filepath):
    """Load a saved classifier"""
    with open(filepath, 'rb') as file:
        checkpoint = pickle.load(file)
    
    return checkpoint['classifier'], checkpoint['label_encoder']


def format_predictions(classifier, label_encoder, X, df):
    """
    Format the predictions of the XGBoost Classifier

    Parameters
    ----------
    classifier : xgb.XGBClassifier
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
    # Get probabilities
    probs = classifier.predict_proba(X)
    
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


def validate_classifier(validate_df, classifier, label_encoder, layer, output_path):
    """
    Validate the XGBoost Classifier

    Parameters
    ----------
    validate_df : pandas.DataFrame
        The dataframe of the validate dataset
    classifier : xgb.XGBClassifier
        The classifier to validate
    label_encoder : LabelEncoder
        The label encoder
    layer : int
        The layer of the csv file
    output_path : str
        The path to save the predictions

    Returns
    -------
    accuracy : float
        The accuracy of the classifier
    """
    # Validate
    X_validate = validate_df['embedding']
    X_validate = np.array([ast.literal_eval(val) for val in X_validate])
    y_validate = validate_df['cluster_idx'].values
    
    # Encode labels
    y_validate_encoded = label_encoder.transform(y_validate)
    
    # Calculate accuracy
    accuracy = classifier.score(X_validate, y_validate_encoded)
    
    # Format predictions
    df = format_predictions(classifier, label_encoder, X_validate, validate_df)
    
    # Add the Actual column at the index 1
    df.insert(1, 'Actual', y_validate)
    
    # Save the predictions
    df.to_csv(output_path + 'predictions_layer_' + str(layer) + '.csv', sep='\t', index=False)
    
    return accuracy


def predict_classifier(test_df, classifier, label_encoder, layer, output_path):
    """
    Predict using the XGBoost Classifier

    Parameters
    ----------
    test_df : pandas.DataFrame
        The dataframe of the test dataset
    classifier : xgb.XGBClassifier
        The classifier to predict
    label_encoder : LabelEncoder
        The label encoder
    layer : int
        The layer of the csv file
    output_path : str
        The path to save predictions

    Returns
    -------
    predictions : pandas.DataFrame
        The predictions of the classifier
    """
    # Predict
    X_test = test_df['embedding']
    X_test = np.array([ast.literal_eval(val) for val in X_test])
    
    df = format_predictions(classifier, label_encoder, X_test, test_df)
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
    
    # XGBoost specific arguments
    parser.add_argument('--n_estimators', type=int, default=200, help='number of boosting rounds')
    parser.add_argument('--max_depth', type=int, default=6, help='maximum depth of a tree')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='boosting learning rate')
    parser.add_argument('--subsample', type=float, default=0.8, 
                       help='subsample ratio of the training instances')
    parser.add_argument('--colsample_bytree', type=float, default=0.8, 
                       help='subsample ratio of columns when constructing each tree')
    parser.add_argument('--gamma', type=float, default=0, 
                       help='minimum loss reduction required to make a further partition')
    parser.add_argument('--reg_alpha', type=float, default=0, help='L1 regularization term on weights')
    parser.add_argument('--reg_lambda', type=float, default=1, help='L2 regularization term on weights')
    parser.add_argument('--n_jobs', type=int, default=-1, help='number of parallel threads')
    parser.add_argument('--early_stopping_rounds', type=int, default=10, help='early stopping rounds')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.load_classifier_from_local:
        classifier, label_encoder = load_classifier(args.classifier_file_path)
    else:
        classifier = None
        label_encoder = None
    
    if args.do_train:
        train_df = pd.read_csv(args.train_file_path, sep='\t')
        classifier, label_encoder = train_classifier(
            train_df,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            gamma=args.gamma,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            n_jobs=args.n_jobs,
            early_stopping_rounds=args.early_stopping_rounds
        )
        
        save_path = args.save_path + '/model/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        save_classifier(classifier, label_encoder, args.layer, save_path)
    
    if args.do_validate:
        validate_df = pd.read_csv(args.validate_file_path, sep='\t')
        
        save_path = args.save_path + '/validate_predictions/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        accuracy = validate_classifier(validate_df, classifier, label_encoder, args.layer, save_path)
        print('Validation Accuracy: ', accuracy)
    
    if args.do_predict:
        test_df = pd.read_csv(args.test_file_path, sep='\t')
        
        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        predict_classifier(test_df, classifier, label_encoder, args.layer, save_path)
    
    end_time = time.time()
    print("Time taken: ", end_time - start_time)


if __name__ == '__main__':
    main()

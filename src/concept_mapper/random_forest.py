import argparse
import ast
import os

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time


def train_classifier(train_df, n_estimators=200, max_depth=None, min_samples_split=5, 
                     min_samples_leaf=2, max_features='sqrt', n_jobs=-1):
    """
    Train the Random Forest Classifier

    Parameters
    ----------
    train_df : pandas.DataFrame
        The dataframe of the train dataset
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of the tree
    min_samples_split : int
        Minimum number of samples required to split an internal node
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node
    max_features : str or int
        Number of features to consider when looking for the best split
    n_jobs : int
        Number of jobs to run in parallel

    Returns
    -------
    classifier : RandomForestClassifier
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
    
    # Initialize and train classifier
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1
    )
    
    print("Training Random Forest Classifier...")
    classifier.fit(X_train, y_train_encoded)
    
    # Print feature importance statistics
    feature_importance = classifier.feature_importances_
    print(f"Feature importance - Mean: {feature_importance.mean():.4f}, Std: {feature_importance.std():.4f}")
    
    return classifier, label_encoder


def save_classifier(classifier, label_encoder, layer, output_path):
    """
    Save the Random Forest Classifier

    Parameters
    ----------
    classifier : RandomForestClassifier
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
    filename = output_path + 'layer_' + str(layer) + '_rf_classifier.pkl'
    
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
    Format the predictions of the Random Forest Classifier

    Parameters
    ----------
    classifier : RandomForestClassifier
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
    Validate the Random Forest Classifier

    Parameters
    ----------
    validate_df : pandas.DataFrame
        The dataframe of the validate dataset
    classifier : RandomForestClassifier
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
    Predict using the Random Forest Classifier

    Parameters
    ----------
    test_df : pandas.DataFrame
        The dataframe of the test dataset
    classifier : RandomForestClassifier
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
    
    # Random Forest specific arguments
    parser.add_argument('--n_estimators', type=int, default=200, help='number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth of the tree')
    parser.add_argument('--min_samples_split', type=int, default=5, 
                       help='minimum number of samples required to split an internal node')
    parser.add_argument('--min_samples_leaf', type=int, default=2, 
                       help='minimum number of samples required to be at a leaf node')
    parser.add_argument('--max_features', type=str, default='sqrt', 
                       help='number of features to consider when looking for the best split')
    parser.add_argument('--n_jobs', type=int, default=-1, help='number of jobs to run in parallel')
    
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
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=args.n_jobs
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

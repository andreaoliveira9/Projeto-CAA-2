#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing for the AIDS Clinical Trials Group Study 175 dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import json
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load the AIDS Clinical Trials Group Study 175 dataset from UCI repository."""
    print("Loading AIDS Clinical Trials Group Study 175 dataset...")
    
    # Fetch data from UCI ML Repository
    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)
    
    # Data (as pandas dataframes)
    X = aids_clinical_trials_group_study_175.data.features
    y = aids_clinical_trials_group_study_175.data.targets
    X = X.drop(columns=['time'])

    # Print dataset information
    print(f"Dataset name: {aids_clinical_trials_group_study_175.metadata.name}")
    print(f"Number of instances: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target variable: {list(y.columns)}")
    
    return X, y, aids_clinical_trials_group_study_175.metadata

def explore_data(X, y):
    """Perform exploratory data analysis on the dataset."""
    # Merge features and target for convenience
    data = pd.concat([X, y], axis=1)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"\nMissing values per column:\n{missing_values[missing_values > 0]}")
    
    # Basic information about the dataset
    print("\nDataset summary:")
    print(data.describe().T)
    
    # Display target distribution
    target_counts = y.value_counts()
    print(f"\nTarget value distribution:\n{target_counts}")
    
    # Plot target distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y.iloc[:, 0])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('output/class_distribution.png')
    
    # Plot correlation matrix
    plt.figure(figsize=(14, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('output/correlation_matrix.png')
    
    # Plot distributions of key features
    key_features = ['age', 'cd40', 'cd420', 'cd80', 'cd820']
    
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(key_features):
        plt.subplot(3, 2, i+1)
        sns.histplot(data=data, x=feature, hue=data.columns[-1], kde=True)
        plt.title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig('output/feature_distributions.png')
    
    return data

def preprocess_data(data):
    """Preprocess the dataset."""
    # Handle missing values
    data_processed = data.fillna(data.median(numeric_only=True))
    
    # Extract features and target
    y = data_processed[data_processed.columns[-1]]
    X = data_processed.drop(columns=[data_processed.columns[-1]])
    
    # Save feature names for later use
    feature_names = list(X.columns)
    with open('output/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    # Save preprocessing info
    preprocessing_info = {
        'num_samples': X.shape[0],
        'num_features': X.shape[1],
        'feature_names': feature_names,
        'class_distribution': y.value_counts().to_dict()
    }
    
    with open('output/preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f)
    
    return X, y, feature_names, preprocessing_info

def main():
    """Main function to load and preprocess data."""
    # First change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory after changing working directory
    if not os.path.exists('output'):
        os.makedirs('output')
    
    print("Starting data preprocessing for AIDS Clinical Trials Group Study 175 dataset...\n")
    
    # Load data
    X, y, metadata = load_data()
    
    # Explore data
    data = explore_data(X, y)
    
    # Preprocess data
    X_processed, y_processed, feature_names, preprocessing_info = preprocess_data(data)
    
    # Save preprocessed data
    joblib.dump((X_processed, y_processed), 'output/preprocessed_data.pkl')
    
    print("\nData preprocessing completed.")
    print(f"Preprocessed data saved to 'output/preprocessed_data.pkl'")
    print(f"Preprocessing information saved to 'output/preprocessing_info.json'")
    
    return X_processed, y_processed, feature_names

if __name__ == "__main__":
    main() 
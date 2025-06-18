#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest model for AIDS Clinical Trials Group Study 175 dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo
import joblib
import json

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess the AIDS Clinical Trials Group Study 175 dataset."""
    print("Loading AIDS Clinical Trials Group Study 175 dataset...")
    
    # Fetch data from UCI ML Repository
    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)
    
    # Data (as pandas dataframes)
    X = aids_clinical_trials_group_study_175.data.features
    y = aids_clinical_trials_group_study_175.data.targets
    
    # Print dataset information
    print(f"Dataset name: {aids_clinical_trials_group_study_175.metadata.name}")
    print(f"Number of instances: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target variable: {list(y.columns)}")
    
    # Merge features and target for convenience
    data = pd.concat([X, y], axis=1)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print(f"\nMissing values per column:\n{missing_values[missing_values > 0]}")
    
    # Handle missing values
    data = data.fillna(data.median(numeric_only=True))
    
    # Basic information about the dataset
    print("\nDataset summary:")
    print(data.describe().T)
    
    # Extract features and target
    X = data.drop(columns=list(y.columns))
    y = data[aids_clinical_trials_group_study_175.data.targets.columns[0]]  # Main target
    
    # Display target distribution
    target_counts = y.value_counts()
    print(f"\nTarget value distribution:\n{target_counts}")
    
    # Save feature names for later use
    feature_names = list(X.columns)
    with open('output/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    return X, y, feature_names

def train_random_forest(X, y, feature_names):
    """Train a Random Forest model on the dataset."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("\nTraining Random Forest model...")
    
    # Simple grid search to find best hyperparameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                              cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print(f"\nClassification report:\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png')
    
    # ROC curve
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('output/roc_curve.png')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png')
    
    print("\nModel feature importance:")
    print(feature_importance.head(20))
    
    # Save the model and test data for XAI methods
    joblib.dump(best_rf, 'output/random_forest_model.pkl')
    joblib.dump((X_test, y_test), 'output/test_data.pkl')
    
    return best_rf, X_train, X_test, y_train, y_test

def main():
    """Main function to load data, train model, and evaluate."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting Random Forest training for AIDS Clinical Trials Group Study 175 dataset...\n")
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()
    
    # Train Random Forest model
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y, feature_names)
    
    print("\nRandom Forest model training and evaluation completed.")
    print("Results and model saved in the 'output' directory.")
    
    return model, X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    main() 
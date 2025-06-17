#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest model training for AIDS Clinical Trials Group Study 175 dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import json

# Set random seed for reproducibility
np.random.seed(42)

def load_preprocessed_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    
    try:
        X, y = joblib.load('output/preprocessed_data.pkl')
        
        # Load feature names
        with open('output/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
        
        return X, y, feature_names
    
    except FileNotFoundError:
        print("Preprocessed data not found. Please run 1_data_preprocessing.py first.")
        return None, None, None

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
    
    # Save the classification report as JSON
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    with open('output/classification_report.json', 'w') as f:
        json.dump(report_dict, f)
    
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
    joblib.dump((X_train, y_train), 'output/train_data.pkl')
    
    # Save model info
    model_info = {
        'model_type': 'RandomForest',
        'best_parameters': grid_search.best_params_,
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'num_train_samples': X_train.shape[0],
        'num_test_samples': X_test.shape[0],
        'feature_importance': {feature: float(importance) for feature, importance 
                              in zip(feature_names, best_rf.feature_importances_)}
    }
    
    with open('output/model_info.json', 'w') as f:
        json.dump(model_info, f)
    
    return best_rf, X_train, X_test, y_train, y_test, model_info

def main():
    """Main function to train and evaluate the model."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting model training for AIDS Clinical Trials Group Study 175 dataset...\n")
    
    # Load preprocessed data
    X, y, feature_names = load_preprocessed_data()
    
    if X is None or y is None:
        return
    
    # Train Random Forest model
    model, X_train, X_test, y_train, y_test, model_info = train_random_forest(X, y, feature_names)
    
    print("\nRandom Forest model training and evaluation completed.")
    print("Results and model saved in the 'output' directory.")
    
    return model, X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    main() 
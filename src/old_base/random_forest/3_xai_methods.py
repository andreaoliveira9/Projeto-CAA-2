#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XAI (Explainable AI) methods for the Random Forest model on AIDS Clinical Trials Group Study 175 dataset.
Includes SHAP analysis and permutation importance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.inspection import permutation_importance
import time
import shap

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_model_and_data():
    """Load the trained model and test data."""
    print("Loading model and data...")
    
    try:
        # Load model
        model = joblib.load('output/random_forest_model.pkl')
        
        # Load test data
        X_test, y_test = joblib.load('output/test_data.pkl')
        
        # Load train data (for background in SHAP)
        X_train, y_train = joblib.load('output/train_data.pkl')
        
        # Load feature names
        with open('output/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        print(f"Model and data loaded successfully.")
        print(f"Test data shape: {X_test.shape}")
        
        return model, X_test, y_test, X_train, y_train, feature_names
    
    except FileNotFoundError:
        print("Model or data files not found. Please run 1_data_preprocessing.py and 2_model_training.py first.")
        return None, None, None, None, None, None

def apply_shap_simple(model, X_train, X_test, feature_names):
    """Apply simplified SHAP (SHapley Additive exPlanations) analysis."""
    print("\nApplying simplified SHAP analysis...")
    start_time = time.time()
    
    # Ensure the output directory exists
    ensure_directory_exists('output/shap_results')
    
    # Convert data to DataFrames if they aren't already
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    # Sample a small subset of test data for SHAP analysis
    sample_size = min(50, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    
    # Create a SHAP explainer and get SHAP values
    print("Creating SHAP explainer and computing values...")
    try:
        # Use KernelExplainer which is more flexible with input types
        # Create a prediction function that outputs a single prediction (not probabilities)
        def predict_fn(x):
            return model.predict_proba(x)[:,1]  # Return probability of class 1
        
        # Create a background dataset (small subset of training data)
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)
        
        # Select a small subset of background samples
        background = shap.sample(X_train, 100)
        
        # Create KernelExplainer
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values (this will be 2D as required)
        shap_values = explainer.shap_values(X_sample)
        
        # Create feature importance DataFrame based on mean absolute SHAP values
        importances = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': importances
        }).sort_values('SHAP_Importance', ascending=False)
        
        # Save feature importance to CSV
        feature_importance.to_csv('output/shap_results/shap_feature_importance.csv', index=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_names)), feature_importance.SHAP_Importance.values)
        plt.yticks(range(len(feature_names)), feature_importance.Feature.values)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Feature Importance Based on SHAP Values')
        plt.tight_layout()
        plt.savefig('output/shap_results/shap_feature_importance.png')
        plt.close()
        
        # Print top 5 important features
        print("\nTop 5 most important features according to SHAP:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"{i+1}. {row['Feature']}: {row['SHAP_Importance']:.4f}")
        
        # Create a summary plot manually
        plt.figure(figsize=(10, 8))
        # Sort features by importance for the plot
        sorted_idx = feature_importance['SHAP_Importance'].argsort()
        plt.barh(range(len(sorted_idx)), feature_importance.iloc[sorted_idx]['SHAP_Importance'])
        plt.yticks(range(len(sorted_idx)), [feature_importance.iloc[i]['Feature'] for i in sorted_idx])
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('output/shap_results/shap_summary_plot.png')
        plt.close()
        
        # Create individual explanations for a few examples
        plt.figure(figsize=(12, 8))
        for i in range(min(5, len(X_sample))):
            # Get feature values for this instance
            instance = X_sample.iloc[i]
            
            # Get SHAP values for this instance
            instance_shap = shap_values[i]
            
            # Get feature importance ranking for this instance
            instance_importance = [(feature_names[j], abs(instance_shap[j])) for j in range(len(feature_names))]
            instance_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Create a force plot for this instance
            plt.figure(figsize=(10, 4))
            # Take top 10 features
            top_features = instance_importance[:10]
            feature_names_top = [f[0] for f in top_features]
            shap_values_top = [instance_shap[feature_names.index(f[0])] for f in top_features]
            
            # Color bars based on contribution direction
            colors = ['blue' if x > 0 else 'red' for x in shap_values_top]
            
            plt.barh(range(len(feature_names_top)), shap_values_top, color=colors)
            plt.yticks(range(len(feature_names_top)), feature_names_top)
            plt.title(f'SHAP Explanations for Instance {i+1}')
            plt.xlabel('SHAP Value')
            plt.tight_layout()
            plt.savefig(f'output/shap_results/instance_{i+1}_explanation.png')
            plt.close()
        
        # Save SHAP results as JSON
        result = {
            'expected_value': explainer.expected_value,
            'top_features': feature_importance.head(10)[['Feature', 'SHAP_Importance']].to_dict('records')
        }
        
        with open('output/shap_results/shap_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=float)
        
        end_time = time.time()
        print(f"SHAP analysis completed in {end_time - start_time:.2f} seconds.")
        return feature_importance
        
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        # Return empty DataFrame with correct columns if SHAP fails
        return pd.DataFrame(columns=['Feature', 'SHAP_Importance'])

def apply_permutation_importance(model, X_test, y_test, feature_names):
    """Apply Permutation Importance to explain model predictions."""
    print("\nCalculating Permutation Importance...")
    start_time = time.time()
    
    # Ensure the output directory exists
    ensure_directory_exists('output/xai_results')
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    # Create dataframe with results
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    # Save to CSV
    perm_importance_df.to_csv('output/xai_results/permutation_importance.csv', index=False)
    
    # Plot permutation importance (top 15 features)
    plt.figure(figsize=(10, 8))
    top_features = perm_importance_df.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Permutation Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig('output/xai_results/permutation_importance.png')
    plt.close()
    
    # Plot permutation importance as a boxplot
    plt.figure(figsize=(10, 8))
    sorted_idx = perm_importance_df['Importance'].sort_values(ascending=False).index
    sorted_features = [perm_importance_df.loc[i, 'Feature'] for i in sorted_idx[:15]]
    
    # Extract and reshape the importance data for the top 15 features
    importance_data = []
    for feature in sorted_features:
        idx = feature_names.index(feature)
        importance_data.append(perm_importance.importances[idx, :])
    
    plt.boxplot(importance_data, labels=sorted_features, vert=False)
    plt.title('Permutation Importance Distribution (Top 15)')
    plt.xlabel('Decrease in Accuracy')
    plt.tight_layout()
    plt.savefig('output/xai_results/permutation_importance_boxplot.png')
    plt.close()
    
    # Save permutation importance results as JSON
    permutation_results = {
        'feature_names': feature_names,
        'importances': {
            'mean': perm_importance.importances_mean.tolist(),
            'std': perm_importance.importances_std.tolist()
        }
    }
    with open('output/xai_results/permutation_results.json', 'w') as f:
        json.dump(permutation_results, f, indent=2)
    
    end_time = time.time()
    print(f"Permutation importance analysis completed in {end_time - start_time:.2f} seconds.")
    print(f"Top 5 important features:")
    for i in range(5):
        feature = perm_importance_df.iloc[i]
        print(f"{i+1}. {feature['Feature']}: {feature['Importance']:.4f} ± {feature['Std']:.4f}")
    
    return perm_importance_df

def compare_xai_methods(shap_df, perm_df):
    """Compare different XAI methods' feature importance rankings."""
    print("\nComparing XAI methods...")
    
    # Ensure the output directory exists
    ensure_directory_exists('output/xai_results')
    
    # If SHAP analysis failed, just return
    if shap_df.empty:
        print("SHAP analysis failed, cannot compare methods.")
        return None
    
    # Get top features from each method
    shap_top = shap_df['Feature'].head(10).tolist()
    perm_top = perm_df['Feature'].head(10).tolist()
    
    # Find common features
    common_features = list(set(shap_top) & set(perm_top))
    
    print(f"\nCommon features in top 10 for both methods: {len(common_features)}")
    for feature in common_features:
        shap_rank = shap_top.index(feature) + 1
        perm_rank = perm_top.index(feature) + 1
        print(f"- {feature}: SHAP rank = {shap_rank}, Permutation rank = {perm_rank}")
    
    # Create comparison DataFrame
    all_features = list(set(shap_top + perm_top))
    comparison = pd.DataFrame(index=all_features, columns=['SHAP_rank', 'Permutation_rank'])
    
    for feature in all_features:
        # SHAP rank
        if feature in shap_top:
            comparison.loc[feature, 'SHAP_rank'] = shap_top.index(feature) + 1
        else:
            comparison.loc[feature, 'SHAP_rank'] = np.nan
        
        # Permutation rank
        if feature in perm_top:
            comparison.loc[feature, 'Permutation_rank'] = perm_top.index(feature) + 1
        else:
            comparison.loc[feature, 'Permutation_rank'] = np.nan
    
    # Save comparison to CSV
    comparison.to_csv('output/xai_results/method_comparison.csv')
    
    # Create visual comparison
    comparison_data = []
    for feature in all_features:
        row = {'Feature': feature}
        
        # SHAP rank (reversed for visualization)
        if feature in shap_top:
            row['SHAP'] = 10 - shap_top.index(feature)
        else:
            row['SHAP'] = 0
            
        # Permutation rank (reversed for visualization)
        if feature in perm_top:
            row['Permutation'] = 10 - perm_top.index(feature)
        else:
            row['Permutation'] = 0
            
        comparison_data.append(row)
    
    # Convert to DataFrame and plot
    comparison_plot_df = pd.DataFrame(comparison_data)
    comparison_melted = pd.melt(comparison_plot_df, id_vars=['Feature'], 
                              var_name='Method', value_name='Importance')
    
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x='Feature', y='Importance', hue='Method', data=comparison_melted)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title('Feature Importance Ranking Comparison')
    plt.tight_layout()
    plt.savefig('output/xai_results/method_comparison.png')
    plt.close()
    
    print("Comparison saved to output/xai_results/method_comparison.csv")
    
    return comparison

def main():
    """Main function to apply XAI methods."""
    # First change to the script directory 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create main output directories
    ensure_directory_exists('output')
    ensure_directory_exists('output/xai_results')
    ensure_directory_exists('output/shap_results')
    
    print("Starting XAI analysis for Random Forest model...")
    
    # Load model and data
    model, X_test, y_test, X_train, y_train, feature_names = load_model_and_data()
    
    if model is None:
        return
    
    # Convert numpy arrays to pandas DataFrame if necessary
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)
    
    # Apply SHAP analysis (simplified version)
    shap_importance = apply_shap_simple(model, X_train, X_test, feature_names)
    
    # Apply Permutation Importance
    perm_importance = apply_permutation_importance(model, X_test, y_test, feature_names)
    
    # Compare XAI methods
    comparison = compare_xai_methods(shap_importance, perm_importance)
    
    print("\nXAI analysis completed.")
    print("SHAP results saved in the 'output/shap_results' directory.")
    print("Permutation results saved in the 'output/xai_results' directory.")
    
    return shap_importance, perm_importance, comparison

if __name__ == "__main__":
    main() 
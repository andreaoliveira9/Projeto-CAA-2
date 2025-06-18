# XAI Methods for AIDS Clinical Trials Group Dataset Analysis

This project demonstrates how to apply Explainable AI (XAI) techniques to a Random Forest model trained on the AIDS Clinical Trials Group Study 175 dataset. The implementation focuses on permutation importance as the primary XAI method for model interpretability.

## Project Overview

The project consists of three main scripts:

1. `1_data_preprocessing.py`: Handles data loading, exploration, visualization, and preprocessing
2. `2_model_training.py`: Trains a Random Forest classifier with hyperparameter tuning, and evaluates model performance
3. `3_xai_methods.py`: Applies XAI methods to explain the model's predictions, focusing on permutation importance

## Dataset

The AIDS Clinical Trials Group Study 175 dataset contains information on 2139 HIV-infected patients with 23 features:

- Key features include time, treatment (trt), age, CD4 counts (cd40, cd420, cd80, cd820), and other clinical measurements
- The target variable ('cid') indicates disease progression

## Implementation Details

### Data Preprocessing

- Load data from UCI Machine Learning Repository
- Explore dataset characteristics and distribution
- Handle missing values
- Generate visualization of class distribution, correlations, and feature distributions
- Save preprocessed data for model training

### Model Training

- Train a Random Forest classifier on the preprocessed data
- Perform hyperparameter tuning using GridSearchCV
- Evaluate model performance with metrics including:
  - Accuracy
  - Precision, recall, and F1-score
  - ROC curve and AUC
- Generate feature importance based on model's built-in importance
- Save model artifacts for XAI analysis

### Explainable AI Methods

#### Permutation Importance

The primary XAI method implemented is permutation importance, which:
- Measures feature importance by randomly shuffling each feature and observing the impact on model performance
- Is model-agnostic and can be applied to any black-box model
- Shows how much the model performance decreases when a feature is permuted

Key findings from permutation importance:
1. The feature `time` has the highest importance (0.22)
2. `offtrt` is the second most important feature (0.03)
3. `cd420` is the third most important feature (0.019)

The implementation includes:
- Feature importance calculation and ranking
- Visualization of feature importance
- Distribution of importance scores for repeated permutations
- Statistical significance of each feature's importance

## Results

The Random Forest model achieves:
- 88.5% accuracy on the test dataset
- Strong precision (89%) and recall (97%) for class 0
- Precision of 85% and recall of 63% for class 1

Feature importance analysis reveals that:
- Temporal features and CD4 counts are the most important predictors
- Patient demographics have moderate importance
- Treatment-related features also contribute to predictions

## How to Run

Run the scripts in sequence:

```bash
python 1_data_preprocessing.py
python 2_model_training.py
python 3_xai_methods.py
```

Results, visualizations, and model artifacts are saved in the `output` directory:
- Model evaluation metrics and feature importance in `output/`
- XAI results in `output/xai_results/`

## Requirements

This project requires:
- Python 3.6+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- ucimlrepo (for dataset access)

## Project Structure

```
structured_data/
├── 1_data_preprocessing.py
├── 2_model_training.py
├── 3_xai_methods.py
├── README.md
└── output/
    ├── preprocessed_data.pkl
    ├── random_forest_model.pkl
    ├── feature_importance.png
    ├── classification_report.json
    ├── xai_results/
    │   ├── permutation_importance.csv
    │   ├── permutation_importance.png
    │   └── permutation_importance_boxplot.png
    └── ...
```

## References

- [UCI Machine Learning Repository - AIDS Clinical Trials Group Study 175](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175)
- Molnar, C. (2020). "Interpretable Machine Learning: A Guide for Making Black Box Models Explainable"
- Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
- Fisher, A., Rudin, C., & Dominici, F. (2019). "All Models Are Wrong, But Many Are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously" 
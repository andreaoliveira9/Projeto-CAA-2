# Projeto-CAA-2
Segundo projeto Complementos de Aprendizagem Automática

# Explainable AI (XAI) for Animal Image Classification

This project implements and compares different Explainable AI (XAI) methods for animal image classification using the "Animal Image Dataset (90 Different Animals)" from Kaggle.

## Overview

The goal of this project is to demonstrate the application of three popular XAI methods to explain the predictions of a deep learning model:

1. **GradCAM** (Gradient-weighted Class Activation Mapping)
2. **LIME** (Local Interpretable Model-agnostic Explanations)
3. **SHAP** (SHapley Additive exPlanations)

## Requirements

The project requires the following dependencies:
- TensorFlow (>= 2.5.0)
- OpenCV
- Lime
- SHAP
- NumPy, Pandas, Matplotlib, etc.

All dependencies can be installed using:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

- `animals.md`: Documentation of state-of-the-art implementations for animal image classification
- `1_data_preprocessing.py`: Data loading, exploration, and preprocessing
- `2_vgg16_implementation.py`: VGG16 model implementation and training
- `3_gradcam_explanation.py`: GradCAM implementation for explaining model predictions
- `4_lime_explanation.py`: LIME implementation for explaining model predictions
- `5_shap_explanation.py`: SHAP implementation for explaining model predictions
- `requirements.txt`: List of dependencies

## Dataset

The project uses the "Animal Image Dataset (90 Different Animals)" from Kaggle:
https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

The dataset should be organized with the following structure:
```
animals/
  antelope/
    image1.jpg
    image2.jpg
    ...
  badger/
    image1.jpg
    ...
  ...
```

## Usage

### 1. Data Preprocessing

```bash
python 1_data_preprocessing.py
```

This script:
- Explores the dataset structure and distribution
- Analyzes image properties (dimensions, aspect ratios, file sizes)
- Displays sample images
- Creates data generators for model training
- Visualizes data augmentation examples

### 2. Training VGG16 Model

```bash
python 2_vgg16_implementation.py
```

This script:
- Loads the preprocessed data
- Builds a VGG16 model with transfer learning
- Trains the model on the animal image dataset
- Fine-tunes the model
- Evaluates the model performance
- Saves the trained model and evaluation results

### 3. GradCAM Explanation

```bash
python 3_gradcam_explanation.py
```

This script:
- Loads the trained model
- Applies GradCAM to explain predictions for sample images
- Analyzes misclassifications using GradCAM
- Provides an interactive mode for exploring GradCAM on user-selected images

### 4. LIME Explanation

```bash
python 4_lime_explanation.py
```

This script:
- Loads the trained model
- Applies LIME to explain predictions for sample images
- Analyzes misclassifications using LIME
- Compares LIME explanations with different parameters
- Provides an interactive mode for exploring LIME on user-selected images

### 5. SHAP Explanation

```bash
python 5_shap_explanation.py
```

This script:
- Loads the trained model
- Creates a SHAP explainer with a background dataset
- Applies SHAP to explain predictions for sample images
- Analyzes misclassifications using SHAP
- Provides an interactive mode for exploring SHAP on user-selected images

## XAI Methods Comparison

### GradCAM
- Uses gradients flowing into the final convolutional layer to generate coarse localization maps
- Highlights the regions in the image that contribute most to the prediction
- Provides visual explanations focused on spatial information

### LIME
- Generates local explanations by perturbing the input and seeing how predictions change
- Creates an interpretable model around the specific prediction
- Shows which parts of the image influence the prediction and with what magnitude

### SHAP
- Based on cooperative game theory (Shapley values)
- Computes the contribution of each feature to the prediction
- Provides consistent and theoretically sound explanations with feature importances

## Results

The results from each XAI method are saved in separate directories:
- `gradcam_results/`
- `lime_results/`
- `shap_results/`

These include visualizations for sample images, misclassified images, and interactive explorations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- GradCAM: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
- LIME: [Why Should I Trust You?: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- SHAP: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- Overview of XAI: [What is Explainable AI?](https://insights.sei.cmu.edu/blog/what-is-explainable-ai/)
- Survey on XAI: [A Survey on Explainable Artificial Intelligence](https://arxiv.org/pdf/2211.06579)

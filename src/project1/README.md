# GradCAM Implementation for Model Explanation

This directory contains a comprehensive implementation of Gradient-weighted Class Activation Mapping (GradCAM) for explaining image classification model predictions. The implementation supports multiple model architectures including ResNet, VGG, and DenseNet.

## Overview

GradCAM is a technique for visualizing which parts of an image most influenced a model's prediction. It produces a heatmap highlighting the regions of the input image that were most important for the model's decision.

This implementation includes:

- Individual model analysis with GradCAM
- Comparison of different model architectures on the same images
- Analysis of correct predictions and misclassifications
- Identification of images where different models disagree

## Files

- `gradcam_utils.py`: Core GradCAM implementation as a reusable class with all necessary utility functions
- `main.py`: Command-line interface for running GradCAM analysis on trained models

## Requirements

- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- pandas
- scikit-learn
- tqdm

## Usage

### Basic Usage

To run a full GradCAM analysis on all available models:

```bash
python main.py
```

### Model Selection

To analyze a specific model architecture:

```bash
python main.py --model resnet  # Options: resnet, vgg, densenet, all
```

### Analysis Types

Different types of analysis can be performed:

```bash
python main.py --analysis_type full  # Options: single, comparison, correct, misclassified, full
```

- `single`: Analyze individual models
- `comparison`: Compare different model architectures on the same images
- `correct`: Only analyze correctly predicted images
- `misclassified`: Only analyze misclassified images
- `full`: Perform all analyses

### Custom Directories

You can specify custom directories for models, dataset, and output:

```bash
python main.py --models_dir /path/to/models --dataset_dir /path/to/dataset --output_dir /path/to/output
```

### Number of Samples

Control the number of samples analyzed per class:

```bash
python main.py --num_samples 5
```

## Example Output

The script will generate visualizations like:

1. **Individual Model Analysis**: GradCAM heatmaps for each model showing how they interpret the images
2. **Architecture Comparison**: Side-by-side comparison of how different architectures interpret the same image
3. **Misclassification Analysis**: Visualization of where models focus when making incorrect predictions

## Customization

To use this implementation with different class names or for different datasets, modify the `WASTE_CLASS_NAMES` variable in `main.py` to match your dataset's classes.

## Adding New Models

To add support for new model architectures:

1. Add the model's filename to the `get_model_files()` function in `main.py`
2. Update the choices for the `--model` argument in `parse_args()`

## Troubleshooting

If you encounter GPU-related errors, the script already forces TensorFlow to use CPU by setting the environment variable `CUDA_VISIBLE_DEVICES="-1"`. You can modify this if you want to use specific GPUs. 
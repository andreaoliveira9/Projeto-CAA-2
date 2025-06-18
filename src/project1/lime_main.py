#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for running LIME analysis on different model architectures.
This script can be used to analyze VGG, ResNet, or DenseNet models.
"""

# Force TensorFlow to use CPU only
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from lime_utils import LIMEUtils

# Define waste classification class names
WASTE_CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LIME analysis for image classification models')
    
    # Model selection arguments
    parser.add_argument('--model', type=str, choices=['resnet', 'vgg', 'densenet', 'all'], 
                        default='all', help='Model architecture to analyze')
    
    # Analysis type arguments
    parser.add_argument('--analysis_type', type=str, 
                        choices=['single', 'comparison', 'correct', 'misclassified', 'full'], 
                        default='full', help='Type of analysis to perform')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save the LIME results')
    
    # Dataset directory
    parser.add_argument('--dataset_dir', type=str, default=None, 
                        help='Directory containing the dataset')
    
    # Models directory
    parser.add_argument('--models_dir', type=str, default=None, 
                        help='Directory containing the trained models')
    
    # Number of samples to analyze
    parser.add_argument('--num_samples', type=int, default=2, 
                        help='Number of samples per class to analyze')
    
    # Number of features (superpixels) for LIME to show
    parser.add_argument('--num_features', type=int, default=5, 
                        help='Number of top features (superpixels) to highlight in LIME explanation')
    
    # Specific image path to analyze
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a specific image to analyze with LIME')
    
    # Flag to display plots instead of saving
    parser.add_argument('--display', action='store_true',
                        help='Display plots instead of saving to files (useful for notebooks)')
    
    return parser.parse_args()

def get_model_files(model_type):
    """Get model filenames based on the selected architecture."""
    if model_type == 'resnet':
        return ['resnet_model_es.h5']
    elif model_type == 'vgg':
        return ['aug_vgg_model_l2.h5']
    elif model_type == 'densenet':
        return ['densenet121_model.h5']
    elif model_type == 'all':
        return None  # Load all available models
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def analyze_single_image(lime_utils, model_type, image_path, num_features, display=False):
    """
    Analyze a single specified image with LIME.
    
    Args:
        lime_utils: LIME utilities instance
        model_type: Type of model to use ('resnet', 'vgg', 'densenet')
        image_path: Path to the image to analyze
        num_features: Number of features to highlight in the explanation
        display: If True, display the plot instead of saving to file (for notebooks)
    
    Returns:
        list or None: List of figure objects if display=True, otherwise None
    """
    print(f"\nAnalyzing image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Load the specified models
    model_files = get_model_files(model_type)
    models = lime_utils.load_models(model_files)
    
    if not models:
        print(f"No {model_type} models found. Exiting.")
        return None
    
    # Try to determine the true class from the image path
    true_class_idx = -1
    for i, class_name in enumerate(lime_utils.CLASS_NAMES):
        if class_name in image_path.lower():
            true_class_idx = i
            print(f"Detected class: {class_name} (index: {i})")
            break
    
    # Create output directory for the image if not displaying
    if not display:
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        single_image_output_dir = os.path.join(lime_utils.OUTPUT_DIR, 'single_image_analysis')
        os.makedirs(single_image_output_dir, exist_ok=True)
    
    # List to collect figures if displaying
    figures = []
    
    # If there's only one model, analyze individually; otherwise do a comparison
    if len(models) == 1:
        # Analyze the image with each model
        for model_name, model in models.items():
            print(f"\nApplying LIME to image with model: {model_name}")
            
            if display:
                # Get figure instead of saving to file
                fig = lime_utils.apply_lime_to_image(image_path, model, output_path=None, num_features=num_features, true_class_idx=true_class_idx)
                if fig is not None:
                    figures.append(fig)
            else:
                # Define output path
                image_basename = os.path.basename(image_path)
                image_name = os.path.splitext(image_basename)[0]
                output_filename = f"lime_{image_name}_{model_name.replace('.h5', '')}.png"
                output_path = os.path.join(single_image_output_dir, output_filename)
                
                # Apply LIME to the image and save to file
                lime_utils.apply_lime_to_image(image_path, model, output_path=output_path, num_features=num_features, true_class_idx=true_class_idx)
    else:
        # If multiple models, do a comparison
        print("\nComparing LIME results across multiple models...")
        
        if display:
            fig = lime_utils.compare_models_on_image(models, image_path, true_class_idx, output_path=None, num_features=num_features)
            if fig is not None:
                figures.append(fig)
        else:
            # Define output path
            image_basename = os.path.basename(image_path)
            image_name = os.path.splitext(image_basename)[0]
            output_filename = f"lime_comparison_{image_name}.png"
            output_path = os.path.join(single_image_output_dir, output_filename)
            
            # Compare models
            lime_utils.compare_models_on_image(models, image_path, true_class_idx, output_path=output_path, num_features=num_features)
    
    if display:
        return figures
    else:
        print(f"Single image analysis completed. Results saved in {single_image_output_dir}")
        return None

def run_single_model_analysis(lime_utils, model_type, num_samples, num_features):
    """Run analysis on a single model architecture."""
    print(f"\nRunning analysis for {model_type.upper()} model(s)...")
    
    # Load the specified models
    model_files = get_model_files(model_type)
    models = lime_utils.load_models(model_files)
    
    if not models:
        print(f"No {model_type} models found. Exiting.")
        return
    
    # Prepare test data
    X_test, y_test, file_paths, y_test_encoded, y_test_raw = lime_utils.prepare_test_data()
    
    # Dictionaries to store indices for consistent analysis across models
    correct_indices_by_class = {}
    misclassified_indices = []
    
    # First model will select the images, other models will use the same ones
    first_model = True
    
    # Analyze each model
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        
        # Evaluate the model
        _, _, _, _ = lime_utils.evaluate_model(model, X_test, y_test)
        
        # Analyze correct predictions
        if first_model:
            # First model: select images and save the indices
            results, selected_indices = lime_utils.analyze_correct_predictions(
                model, model_name, X_test, y_test, file_paths, samples_per_class=num_samples
            )
            correct_indices_by_class = selected_indices
        else:
            # Other models: use the same images selected by the first model
            results, _ = lime_utils.analyze_correct_predictions(
                model, model_name, X_test, y_test, file_paths, 
                samples_per_class=num_samples, specific_indices=correct_indices_by_class
            )
        
        # Analyze misclassifications
        if first_model:
            # First model: select images and save the indices
            results, selected_indices = lime_utils.analyze_misclassifications(
                model, model_name, X_test, y_test, file_paths, num_samples=num_samples
            )
            misclassified_indices = selected_indices
        else:
            # Other models: use the same images selected by the first model
            results, _ = lime_utils.analyze_misclassifications(
                model, model_name, X_test, y_test, file_paths, 
                num_samples=num_samples, specific_indices=misclassified_indices
            )
        
        # After processing the first model, set flag to false
        if first_model:
            first_model = False

def run_model_comparison(lime_utils, model_type, num_samples, num_features):
    """Run comparison analysis between different model architectures.
    
    Args:
        lime_utils: LIME utilities instance
        model_type: Type of models to load ('all', 'resnet', 'vgg', 'densenet')
        num_samples: Number of samples per class to analyze
        num_features: Number of features (superpixels) to show in LIME explanation
    """
    print("\nRunning model comparison analysis...")
    
    # Load the specified models based on model_type
    model_files = get_model_files(model_type)
    models = lime_utils.load_models(model_files)
    
    if len(models) < 2:
        print("Need at least 2 different models for comparison. Exiting.")
        return
    
    # Prepare test data
    X_test, y_test, file_paths, y_test_encoded, y_test_raw = lime_utils.prepare_test_data()
    
    # Create comparison output directory
    comparison_dir = os.path.join(lime_utils.OUTPUT_DIR, 'model_comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Dictionary to store predictions from each model
    all_predictions = {}
    
    # Get predictions from each model
    for model_name, model in models.items():
        print(f"Getting predictions for {model_name}...")
        predictions, _, _, _ = lime_utils.evaluate_model(model, X_test, y_test)
        all_predictions[model_name] = predictions
    
    # Select representative images from each class
    # These will be the SAME images for all models
    comparison_images = lime_utils.select_comparison_images(X_test, y_test_encoded, file_paths, num_per_class=num_samples)
    
    # Compare models on selected images
    print(f"\nComparing models on {len(comparison_images)} representative images...")
    for i, img_info in enumerate(comparison_images):
        output_path = os.path.join(comparison_dir, f"lime_model_comparison_{img_info['class_name']}_{i}.png")
        lime_utils.compare_models_on_image(models, img_info['file_path'], img_info['class_idx'], output_path, num_features=num_features)
    
    # Select images where models disagree
    print("\nSelecting images with disagreement between models...")
    disagreement_images = lime_utils.select_model_disagreement_images(all_predictions, y_test_encoded, file_paths)
    
    # Compare models on disagreement images
    if disagreement_images:
        print(f"\nComparing models on {len(disagreement_images)} disagreement images...")
        for i, img_info in enumerate(disagreement_images):
            output_path = os.path.join(comparison_dir, f"lime_disagreement_comparison_{i}.png")
            lime_utils.compare_models_on_image(models, img_info['file_path'], img_info['class_idx'], output_path, num_features=num_features)
    else:
        print("No disagreement images found.")
    
    print(f"Model comparison analysis completed. Results saved in {comparison_dir}")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize LIME utilities
    lime_utils = LIMEUtils(
        class_names=WASTE_CLASS_NAMES,
        img_size=150,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    # If a specific image path is provided, analyze only that image
    if args.image_path:
        figures = analyze_single_image(lime_utils, args.model, args.image_path, args.num_features, args.display)
        if args.display and figures:
            import matplotlib.pyplot as plt
            for fig in figures:
                plt.figure(fig.number)  # Set the current figure
                plt.show()
        return
    
    # Run the requested analysis
    if args.analysis_type == 'comparison' or args.analysis_type == 'full':
        run_model_comparison(lime_utils, args.model, args.num_samples, args.num_features)
    
    if args.analysis_type == 'single' or args.analysis_type == 'correct' or args.analysis_type == 'misclassified' or args.analysis_type == 'full':
        run_single_model_analysis(lime_utils, args.model, args.num_samples, args.num_features)
    
    print("\nLIME analysis completed successfully!")

if __name__ == "__main__":
    main() 
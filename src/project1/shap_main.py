#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for running SHAP analysis on different model architectures.
This script can be used to analyze VGG, ResNet, or DenseNet models.
"""

# Force TensorFlow to use CPU only
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from shap_utils import SHAPUtils
import random
import numpy as np
import matplotlib.pyplot as plt

# Define waste classification class names
WASTE_CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SHAP analysis for image classification models')
    
    # Model selection arguments
    parser.add_argument('--model', type=str, choices=['resnet', 'vgg', 'densenet', 'all'], 
                        default='all', help='Model architecture to analyze')
    
    # Analysis type arguments
    parser.add_argument('--analysis_type', type=str, 
                        choices=['single', 'comparison', 'correct', 'misclassified', 'full'], 
                        default='full', help='Type of analysis to perform')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save the SHAP results')
    
    # Dataset directory
    parser.add_argument('--dataset_dir', type=str, default=None, 
                        help='Directory containing the dataset')
    
    # Models directory
    parser.add_argument('--models_dir', type=str, default=None, 
                        help='Directory containing the trained models')
    
    # Number of samples to analyze
    parser.add_argument('--num_samples', type=int, default=2, 
                        help='Number of samples per class to analyze')
    
    # Number of background samples for SHAP
    parser.add_argument('--num_background', type=int, default=10, 
                        help='Number of background samples to use for SHAP explanations')
    
    # Specific image path to analyze
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a specific image to analyze with SHAP')
    
    return parser.parse_args()

def get_model_files(model_type):
    """Get model filenames based on the selected architecture."""
    if model_type == 'resnet':
        return ['resnet_model.h5', 'resnet_model_es.h5']
    elif model_type == 'vgg':
        return ['aug_vgg_model_l2.h5']
    elif model_type == 'densenet':
        return ['densenet121_model.h5']
    elif model_type == 'all':
        return None  # Load all available models
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def analyze_single_image(shap_utils, model_type, image_path):
    """
    Analyze a single specified image with SHAP.
    
    Args:
        shap_utils: SHAP utilities instance
        model_type: Type of model to use ('resnet', 'vgg', 'densenet')
        image_path: Path to the image to analyze
    """
    print(f"\nAnalyzing image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load the specified models
    model_files = get_model_files(model_type)
    models = shap_utils.load_models(model_files)
    
    if not models:
        print(f"No {model_type} models found. Exiting.")
        return
    
    # Get background data for SHAP
    try:
        background_data = shap_utils.get_background_data()
        
        # Create output directory for the image
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        single_image_output_dir = os.path.join(shap_utils.OUTPUT_DIR, 'single_image_analysis')
        os.makedirs(single_image_output_dir, exist_ok=True)
        
        # Analyze the image with each model
        for model_name, model in models.items():
            print(f"\nApplying SHAP to image with model: {model_name}")
            
            # Special handling for DenseNet models
            is_densenet = False
            if hasattr(model, 'name'):
                is_densenet = 'densenet' in model.name.lower()
                
            if is_densenet:
                print("Using special handling for DenseNet model")
            
            # Define output path
            output_filename = f"shap_{image_name}_{model_name.replace('.h5', '')}.png"
            output_path = os.path.join(single_image_output_dir, output_filename)
            
            try:
                # Apply SHAP to the image
                shap_utils.apply_shap_to_image(image_path, model, background_data, output_path=output_path)
            except Exception as e:
                print(f"Error analyzing image with {model_name}: {e}")
                print("Trying fallback approach...")
                try:
                    # Preprocess the image directly
                    img, _ = shap_utils.preprocess_image_for_shap(image_path)
                    
                    # Get prediction
                    preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                    predicted_class = np.argmax(preds)
                    confidence = preds[predicted_class]
                    class_name = shap_utils.CLASS_NAMES[predicted_class]
                    
                    # Create a simple visualization with just the prediction
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.imshow(img)
                    ax.set_title(f'Model: {model_name}\nPrediction: {class_name}\nConfidence: {confidence:.2f}\n\nSHAP explanation failed: {str(e)}', 
                               fontsize=12, color='red')
                    ax.axis('off')
                    
                    # Create fallback output path
                    fallback_output = os.path.join(single_image_output_dir, f"fallback_{output_filename}")
                    plt.savefig(fallback_output)
                    plt.close(fig)
                    print(f"Fallback visualization saved to {fallback_output}")
                except Exception as inner_e:
                    print(f"Fallback also failed: {inner_e}")
        
        print(f"Single image analysis completed. Results saved in {single_image_output_dir}")
    except Exception as e:
        print(f"Error during analysis: {e}")

def run_single_model_analysis(shap_utils, model_type, num_samples):
    """Run analysis on a single model architecture."""
    print(f"\nRunning analysis for {model_type.upper()} model(s)...")
    
    # Load the specified models
    model_files = get_model_files(model_type)
    models = shap_utils.load_models(model_files)
    
    if not models:
        print(f"No {model_type} models found. Exiting.")
        return
    
    # Prepare test data
    X_test, y_test, file_paths, y_test_encoded, y_test_raw = shap_utils.prepare_test_data()
    
    # Get background data for SHAP
    background_data = shap_utils.get_background_data()
    
    # Analyze correct and misclassified predictions for each model
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        
        # Create model output directory
        model_output_dir = os.path.join(shap_utils.OUTPUT_DIR, model_name.replace('.h5', ''))
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Evaluate the model
        predictions, y_pred, y_true, _ = shap_utils.evaluate_model(model, X_test, y_test)
        
        # Find correct predictions for each class
        correct_indices = np.where(y_pred == y_true)[0]
        
        # Process samples from each class (correct predictions)
        for class_idx, class_name in enumerate(shap_utils.CLASS_NAMES):
            class_correct_indices = [idx for idx in correct_indices if y_true[idx] == class_idx]
            
            # Skip if no correct predictions for this class
            if not class_correct_indices:
                print(f"No correct predictions for class {class_name}")
                continue
            
            # Sample a few images from each class
            sampled_indices = random.sample(class_correct_indices, min(num_samples, len(class_correct_indices)))
            
            for idx in sampled_indices:
                img_path = file_paths[idx]
                
                # Define output path
                img_filename = os.path.basename(img_path)
                output_filename = f"shap_correct_{class_name}_{os.path.splitext(img_filename)[0]}.png"
                output_path = os.path.join(model_output_dir, output_filename)
                
                # Apply SHAP
                shap_utils.apply_shap_to_image(img_path, model, background_data, class_idx, output_path)
        
        # Find misclassified predictions
        misclassified_indices = np.where(y_pred != y_true)[0]
        
        # Sample from misclassifications
        if len(misclassified_indices) > 0:
            sampled_indices = random.sample(list(misclassified_indices), min(num_samples, len(misclassified_indices)))
            
            for idx in sampled_indices:
                img_path = file_paths[idx]
                true_class = shap_utils.CLASS_NAMES[y_true[idx]]
                pred_class = shap_utils.CLASS_NAMES[y_pred[idx]]
                
                # Define output path
                img_filename = os.path.basename(img_path)
                output_filename = f"shap_misclassified_{true_class}_as_{pred_class}_{os.path.splitext(img_filename)[0]}.png"
                output_path = os.path.join(model_output_dir, output_filename)
                
                # Apply SHAP to the predicted class (to see why the model thought it was this class)
                shap_utils.apply_shap_to_image(img_path, model, background_data, y_pred[idx], output_path)
        else:
            print("No misclassifications found!")
    
    print(f"SHAP analysis for {model_type} model(s) completed.")

def run_model_comparison(shap_utils, model_type, num_samples):
    """Run comparison analysis between different model architectures."""
    print("\nRunning model comparison analysis...")
    
    # Load the specified models based on model_type
    model_files = get_model_files(model_type)
    models = shap_utils.load_models(model_files)
    
    if len(models) < 2:
        print("Need at least 2 different models for comparison. Exiting.")
        return
    
    # Prepare test data
    X_test, y_test, file_paths, y_test_encoded, y_test_raw = shap_utils.prepare_test_data()
    
    # Get background data for SHAP
    background_data = shap_utils.get_background_data()
    
    # Create comparison output directory
    comparison_dir = os.path.join(shap_utils.OUTPUT_DIR, 'model_comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Select representative images from each class
    # Here we'll just randomly select images from each class
    comparison_images = []
    
    # Get unique classes
    unique_classes = np.unique(y_test_encoded)
    
    for class_idx in unique_classes:
        # Get indices of all images in this class
        class_indices = np.where(y_test_encoded == class_idx)[0]
        
        # Skip if no images for this class
        if len(class_indices) == 0:
            continue
        
        # Randomly select images from this class
        selected_indices = random.sample(list(class_indices), min(num_samples, len(class_indices)))
        
        for idx in selected_indices:
            comparison_images.append({
                'index': idx,
                'class_idx': class_idx,
                'class_name': shap_utils.CLASS_NAMES[class_idx],
                'file_path': file_paths[idx]
            })
    
    # Compare models on selected images
    print(f"\nComparing models on {len(comparison_images)} representative images...")
    for i, img_info in enumerate(comparison_images):
        output_path = os.path.join(comparison_dir, f"shap_model_comparison_{img_info['class_name']}_{i}.png")
        shap_utils.compare_models_on_image(models, img_info['file_path'], img_info['class_idx'], background_data, output_path)
    
    print(f"Model comparison analysis completed. Results saved in {comparison_dir}")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize SHAP utilities
    shap_utils = SHAPUtils(
        class_names=WASTE_CLASS_NAMES,
        img_size=150,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    # If a specific image path is provided, analyze only that image
    if args.image_path:
        analyze_single_image(shap_utils, args.model, args.image_path)
        return
    
    # Run the requested analysis
    if args.analysis_type == 'comparison' or args.analysis_type == 'full':
        run_model_comparison(shap_utils, args.model, args.num_samples)
    
    if args.analysis_type == 'single' or args.analysis_type == 'correct' or args.analysis_type == 'misclassified' or args.analysis_type == 'full':
        run_single_model_analysis(shap_utils, args.model, args.num_samples)
    
    print("\nSHAP analysis completed successfully!")

if __name__ == "__main__":
    main() 
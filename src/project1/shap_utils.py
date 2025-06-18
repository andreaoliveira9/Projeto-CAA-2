#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General-purpose SHAP utilities for visualizing and comparing model explanations.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import pandas as pd
import shap
from skimage.segmentation import slic

class SHAPUtils:
    def __init__(self, class_names, img_size=150, dataset_dir=None, models_dir=None, output_dir=None):
        """
        Initialize SHAP utilities with the necessary paths and parameters.
        
        Args:
            class_names (list): List of class names
            img_size (int): Image size for model input
            dataset_dir (str): Path to the dataset directory
            models_dir (str): Path to the directory containing trained models
            output_dir (str): Path to save SHAP results
        """
        self.CLASS_NAMES = class_names
        self.IMG_SIZE = img_size
        
        # Set default paths if not provided
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_DIR = dataset_dir or os.path.join(project_dir, 'dataset_organized')
        self.MODELS_DIR = models_dir or os.path.join(project_dir, 'trained_models')
        self.OUTPUT_DIR = output_dir or os.path.join(project_dir, 'shap_results')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
    def load_models(self, model_names=None):
        """
        Load models from the models directory.
        
        Args:
            model_names (list): Optional list of model filenames to load
                                If None, will try to load all available models
        
        Returns:
            dict: Dictionary of loaded models {model_name: model}
        """
        models = {}
        
        # If no specific models requested, look for all common model files
        if model_names is None:
            model_names = [
                # ResNet models
                'resnet_model.h5',
                'resnet_model_es.h5',
                # VGG models
                'vgg_model.h5',
                # DenseNet models
                'densenet121_model.h5'
            ]
        
        for model_file in model_names:
            model_path = os.path.join(self.MODELS_DIR, model_file)
            if os.path.exists(model_path):
                try:
                    print(f"Loading model: {model_file}")
                    models[model_file] = load_model(model_path)
                    print(f"Successfully loaded {model_file}")
                except Exception as e:
                    print(f"Error loading {model_file}: {e}")
        
        if not models:
            print("No models found in the specified directory.")
        else:
            print(f"Loaded {len(models)} models successfully.")
        
        return models
    
    def load_images(self, path):
        """
        Load and preprocess images from a directory.
        
        Args:
            path (str): Path to the directory containing class subdirectories
        
        Returns:
            tuple: (X, y, file_paths) where X is the image data, y is the labels, 
                  and file_paths are the paths to the original images
        """
        print(f"Loading images from {path}...")
        
        X = []
        y = []
        file_paths = []
        
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if not os.path.isdir(folder_path):
                continue
                
            files = os.listdir(folder_path)
            for file in tqdm(files):
                file_path = os.path.join(folder_path, file)
                try:
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # Convert from BGR to RGB (cv2 loads in BGR format)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(folder)
                    file_paths.append(file_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, file_paths
    
    def prepare_test_data(self, test_path=None):
        """
        Prepare test data for evaluation.
        
        Args:
            test_path (str): Optional path to test data directory
                            If None, will use default test path
        
        Returns:
            tuple: (X_test, y_test_categorical, file_paths, y_test_encoded, y_test)
        """
        print("Preparing test data...")
        
        # Use default test path if not provided
        if test_path is None:
            test_path = os.path.join(self.DATASET_DIR, 'test')
        
        # Load test images
        X_test, y_test, file_paths = self.load_images(test_path)
        
        # Normalize the data
        X_test = X_test / 255.0
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(y_test)
        
        # Convert to one-hot encoding
        y_test_categorical = to_categorical(y_test_encoded)
        
        print(f"Test data prepared: {X_test.shape[0]} images")
        return X_test, y_test_categorical, file_paths, y_test_encoded, y_test
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            model: Trained model to evaluate
            X_test: Test data
            y_test: Test labels (one-hot encoded)
        
        Returns:
            tuple: (predictions, y_pred, y_true, class_accuracies)
        """
        # Get predictions
        predictions = model.predict(X_test, verbose=1)
        
        # Calculate loss and accuracy
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Get predicted and true classes
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate overall accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for i, class_name in enumerate(self.CLASS_NAMES):
            class_indices = np.where(y_true == i)[0]
            if len(class_indices) > 0:
                class_accuracy = np.mean(y_pred[class_indices] == i)
                print(f"{class_name} Accuracy: {class_accuracy:.4f}")
                class_accuracies[class_name] = class_accuracy
        
        return predictions, y_pred, y_true, class_accuracies
    
    def model_predict_fn(self, model):
        """
        Create a prediction function for SHAP that processes images in batches.
        
        Args:
            model: The model to use for predictions
        
        Returns:
            function: A prediction function that takes an array of images and returns predictions
        """
        # Check if this is a DenseNet model
        is_densenet = False
        if hasattr(model, 'name'):
            is_densenet = 'densenet' in model.name.lower()
            
        def predict_fn(images):
            # Normalize the images (convert to float and divide by 255)
            batch = np.array(images).astype('float32') / 255.0
            
            try:
                # Get model predictions
                predictions = model.predict(batch)
                
                # Handle different output formats
                if isinstance(predictions, list):
                    print(f"Model returned a list of {len(predictions)} outputs, using first output")
                    predictions = predictions[0]
                
                # Ensure the output has the right shape for SHAP
                if len(predictions.shape) == 1:
                    # If only one sample and one-dimensional output, reshape to (1, num_classes)
                    predictions = predictions.reshape(1, -1)
                
                # For DenseNet or problematic models, ensure outputs are in range [0,1]
                if is_densenet or np.max(predictions) > 1.0 or np.min(predictions) < 0.0:
                    # Apply softmax to convert logits to probabilities
                    e_x = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
                    predictions = e_x / e_x.sum(axis=1, keepdims=True)
                
                return predictions
                
            except Exception as e:
                print(f"Error in predict function: {e}")
                # Return zeros in case of error
                return np.zeros((len(batch), len(self.CLASS_NAMES)))
                
        return predict_fn
    
    def explain_image_shap(self, model, img, background_data, pred_index=None, num_samples=100):
        """
        Generate SHAP explanation for a single image.
        
        Args:
            model: The model to explain
            img: Image array (already preprocessed and normalized)
            background_data: Background data for SHAP explainer
            pred_index: Optional class index to explain, if None uses the predicted class
            num_samples: Number of samples for SHAP explainer
        
        Returns:
            tuple: (shap_values, predicted_class, class_name, confidence)
        """
        # Create prediction function for SHAP
        predict_fn = self.model_predict_fn(model)
        
        # Get prediction for this image
        preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        predicted_class = np.argmax(preds)
        
        # Use either predicted class or provided index
        class_index = predicted_class if pred_index is None else pred_index
        confidence = preds[class_index]
        class_name = self.CLASS_NAMES[class_index]
        
        # Check if this is a DenseNet model for special handling
        is_densenet = False
        if hasattr(model, 'name'):
            is_densenet = 'densenet' in model.name.lower()
        
        # For DenseNet models, we might need special handling
        if is_densenet:
            print("Using special handling for DenseNet model")
            
            try:
                # For DenseNet, we'll use a direct approach with Kernel SHAP
                # This avoids the input structure warnings from DeepExplainer and GradientExplainer
                
                # Create a masked version of our image - this creates a "mask" over the image
                # so we can see which parts are important
                def mask_image(mask, image):
                    # Apply the mask to the image
                    masked_image = np.zeros(image.shape)
                    for i in range(mask.shape[0]):
                        masked_image[i, :, :, :] = image * mask[i, :, :, :]
                    return masked_image
                
                # Create a function that generates masked versions of the image
                def f(z):
                    # Create a batch of masked images
                    masked_images = mask_image(z, img)
                    # Get predictions for all masked images
                    return model.predict(masked_images)
                
                # Create masker that segments the image into superpixels
                # This creates interpretable segments for SHAP
                from shap.maskers import Image
                masker = Image('inpaint_telea', img.shape)
                
                # Use Partition SHAP explainer - often works better with complex models
                explainer = shap.Explainer(f, masker)
                
                # Get SHAP values - this might take a bit longer but should be more reliable
                shap_explanation = explainer(np.expand_dims(img, axis=0), max_evals=100)
                
                # Convert to the format expected by our visualization functions
                # This creates an array of shape (num_classes, 1, height, width, channels)
                shap_values = []
                for c in range(len(self.CLASS_NAMES)):
                    # If this is the class we're explaining, use the actual values
                    if c == class_index:
                        # Reshape to match expected format
                        values = shap_explanation.values[0]
                        values = np.expand_dims(values, axis=0)  # Add batch dimension
                        shap_values.append(values)
                    else:
                        # For other classes, just use zeros
                        shap_values.append(np.zeros_like(np.expand_dims(img, axis=0)))
                
                return shap_values, predicted_class, class_name, confidence
                
            except Exception as e:
                print(f"Special DenseNet handling failed: {e}")
                print("Trying an alternative approach...")
                
                try:
                    # Alternative approach for DenseNet - use a black box explainer
                    # Create a simplified model function
                    def model_fn(images):
                        # Ensure proper shape
                        if len(images.shape) == 3:
                            images = np.expand_dims(images, axis=0)
                        return model.predict(images)
                    
                    # Use Kernel Explainer as a last resort
                    explainer = shap.KernelExplainer(
                        model_fn, 
                        background_data.reshape(background_data.shape[0], -1)
                    )
                    
                    # Get SHAP values for flattened image
                    flat_img = img.reshape(1, -1)
                    shap_values_flat = explainer.shap_values(flat_img)
                    
                    # Reshape back to image format
                    shap_values = []
                    for sv in shap_values_flat:
                        shap_values.append(sv.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3))
                    
                    return shap_values, predicted_class, class_name, confidence
                    
                except Exception as e2:
                    print(f"Alternative approach also failed: {e2}")
                    print("Trying standard approaches...")
        
        # Standard approach - try different explainers
        try:
            # Initialize the SHAP explainer with DeepExplainer
            explainer = shap.DeepExplainer(model, background_data)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(np.expand_dims(img, axis=0))
            
            return shap_values, predicted_class, class_name, confidence
            
        except Exception as e:
            print(f"Error with DeepExplainer: {e}")
            print("Trying GradientExplainer instead...")
            
            try:
                # Try with GradientExplainer as a fallback
                explainer = shap.GradientExplainer(model, background_data)
                shap_values = explainer.shap_values(np.expand_dims(img, axis=0))
                return shap_values, predicted_class, class_name, confidence
                
            except Exception as e2:
                print(f"Error with GradientExplainer: {e2}")
                print("Trying KernelExplainer as last resort...")
                
                try:
                    # As a last resort, try with KernelExplainer
                    # First flatten the image for KernelExplainer
                    flat_img = img.reshape(1, -1)
                    flat_background = background_data.reshape(background_data.shape[0], -1)
                    
                    # Create a simplified predict function for the flattened data
                    def simplified_predict(x):
                        # Reshape back to original image dimensions
                        reshaped_x = x.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)
                        return predict_fn(reshaped_x)
                    
                    # Initialize KernelExplainer
                    explainer = shap.KernelExplainer(simplified_predict, flat_background)
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(flat_img, nsamples=num_samples)
                    
                    # Reshape SHAP values back to image dimensions
                    shap_values = [sv.reshape(1, self.IMG_SIZE, self.IMG_SIZE, 3) for sv in shap_values]
                    
                    return shap_values, predicted_class, class_name, confidence
                    
                except Exception as e3:
                    print(f"All SHAP explainers failed: {e3}")
                    return None, predicted_class, class_name, confidence
    
    def visualize_shap_explanation(self, img, shap_values, class_index, class_name, confidence):
        """
        Visualize SHAP explanation for a single image.
        
        Args:
            img: Original image (already preprocessed)
            shap_values: SHAP values for the image
            class_index: Class index to explain
            class_name: Name of the class
            confidence: Model confidence for the class
            
        Returns:
            matplotlib figure object
        """
        if shap_values is None:
            # Create a simple visualization with just the prediction if SHAP failed
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img)
            ax.set_title(f'Prediction: {class_name}\nConfidence: {confidence:.2f}\nSHAP explanation failed', fontsize=14)
            ax.axis('off')
            return fig
        
        # Create a figure for visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Display original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Make sure we can handle shap_values in different formats
        try:
            # Display SHAP values as a heatmap
            shap_values_for_class = shap_values[class_index][0]
            
            # Check if shap_values_for_class has the right shape
            if shap_values_for_class.shape != img.shape:
                print(f"Warning: SHAP values shape {shap_values_for_class.shape} doesn't match image shape {img.shape}")
                
                # If it's a 2D matrix, reshape to match image dimensions
                if len(shap_values_for_class.shape) == 2:
                    # Create a 3-channel heatmap from 2D matrix
                    height, width = shap_values_for_class.shape
                    shap_values_for_class = np.repeat(shap_values_for_class[:, :, np.newaxis], 3, axis=2)
                
                # If shapes still don't match, try to resize
                if shap_values_for_class.shape != img.shape:
                    print(f"Resizing SHAP values from {shap_values_for_class.shape} to {img.shape}")
                    # Create a properly sized array and copy values
                    resized_values = np.zeros_like(img)
                    # Copy as much as possible
                    min_h = min(shap_values_for_class.shape[0], img.shape[0])
                    min_w = min(shap_values_for_class.shape[1], img.shape[1])
                    min_c = min(shap_values_for_class.shape[2], img.shape[2])
                    resized_values[:min_h, :min_w, :min_c] = shap_values_for_class[:min_h, :min_w, :min_c]
                    shap_values_for_class = resized_values
            
            # Sum across color channels to get a single channel heatmap
            abs_shap = np.abs(shap_values_for_class).sum(axis=-1)  
            
            # Normalize for visualization
            if np.max(abs_shap) > 0:  # Avoid division by zero
                abs_shap = abs_shap / np.max(abs_shap)
            
            # Display SHAP heatmap
            im = axes[1].imshow(abs_shap, cmap='hot')
            axes[1].set_title(f'SHAP Heatmap for {class_name}', fontsize=14)
            axes[1].axis('off')
            fig.colorbar(im, ax=axes[1], label='SHAP Value Magnitude')
            
            # Create a combined visualization with original image and SHAP overlay
            try:
                # Simple visualization method that avoids broadcasting issues
                blend = img.copy() 
                
                # Create a heatmap in proper RGB format
                cmap = plt.cm.hot
                heatmap = cmap(abs_shap)[:, :, :3]  # Drop alpha channel
                
                # Ensure heatmap has same dimensions as the image
                if heatmap.shape[:2] != img.shape[:2]:
                    # Resize heatmap to match image dimensions
                    from skimage.transform import resize
                    heatmap = resize(heatmap, (img.shape[0], img.shape[1], 3), anti_aliasing=True)
                
                # Apply alpha blending (safely)
                alpha = 0.7
                for i in range(3):  # For each channel
                    blend[:, :, i] = img[:, :, i] * (1 - alpha) + heatmap[:, :, i] * alpha
                
                # Display the blended image
                axes[2].imshow(blend)
                axes[2].set_title(f'SHAP Overlay\nClass: {class_name}\nConfidence: {confidence:.2f}', fontsize=14)
                axes[2].axis('off')
            except Exception as e:
                print(f"Error in overlay generation: {e}")
                # If blending fails, just show the original image with an error message
                axes[2].imshow(img)
                axes[2].set_title(f'Overlay failed: {str(e)}', fontsize=12, color='red')
                axes[2].axis('off')
                
        except Exception as e:
            print(f"Error in SHAP visualization: {e}")
            # If visualization fails, show the original image with error message
            for i in range(1, 3):
                axes[i].imshow(img)
                axes[i].set_title(f'Visualization failed: {str(e)}', fontsize=12, color='red')
                axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def preprocess_image_for_shap(self, img_path):
        """
        Load and preprocess a single image for SHAP explanation.
        
        Args:
            img_path (str): Path to the image file
        
        Returns:
            tuple: (normalized_img, display_img) - the normalized image for model input and the original image for display
        """
        # Load and preprocess the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Create a copy for display
        display_img = img.copy()
        
        # Normalize the image
        normalized_img = img / 255.0
        
        return normalized_img, display_img
    
    def get_background_data(self, num_samples=10):
        """
        Get background data for SHAP explainer from the training set.
        
        Args:
            num_samples (int): Number of background samples to use
        
        Returns:
            numpy array: Background data
        """
        # Use training data as background
        train_path = os.path.join(self.DATASET_DIR, 'train')
        
        if not os.path.exists(train_path):
            print(f"Warning: Training data path {train_path} not found. Using a black image as background.")
            # Use black images as background if no training data
            return np.zeros((num_samples, self.IMG_SIZE, self.IMG_SIZE, 3))
        
        # Load a small subset of training images
        X_train, _, _ = self.load_images(train_path)
        
        # If we have enough training images, sample randomly
        if len(X_train) >= num_samples:
            indices = np.random.choice(len(X_train), num_samples, replace=False)
            background_data = X_train[indices] / 255.0  # Normalize
        else:
            # If not enough images, duplicate some
            background_data = np.zeros((num_samples, self.IMG_SIZE, self.IMG_SIZE, 3))
            for i in range(num_samples):
                background_data[i] = X_train[i % len(X_train)] / 255.0
        
        return background_data
    
    def apply_shap_to_image(self, img_path, model, background_data=None, pred_index=None, output_path=None):
        """
        Apply SHAP to an image and display or save the result.
        
        Args:
            img_path (str): Path to the image file
            model: The model to explain
            background_data: Optional background data for SHAP explainer. If None, will be generated.
            pred_index (int): Optional class index to explain
            output_path (str): Optional path to save the result. If None, just displays the result.
        
        Returns:
            str or None: Path to the saved result if output_path is provided, else None
        """
        # Preprocess the image
        img, display_img = self.preprocess_image_for_shap(img_path)
        
        # Get background data if not provided
        if background_data is None:
            background_data = self.get_background_data()
        
        try:
            # Check if this is a DenseNet model
            is_densenet = False
            if hasattr(model, 'name'):
                is_densenet = 'densenet' in model.name.lower()
            
            # Generate SHAP explanation
            shap_values, predicted_class, class_name, confidence = self.explain_image_shap(
                model, img, background_data, pred_index=pred_index
            )
            
            # Visualize the explanation
            fig = self.visualize_shap_explanation(
                img, shap_values, 
                class_index=predicted_class if pred_index is None else pred_index,
                class_name=class_name, confidence=confidence
            )
            
            # Save the figure if output path is provided, otherwise just display it
            if output_path:
                plt.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                return output_path
            else:
                plt.show()
                plt.close(fig)
                return None
                
        except Exception as e:
            print(f"Error applying SHAP to image {os.path.basename(img_path)}: {e}")
            
            # Create a simple visualization with prediction but without SHAP
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(display_img)
            ax.set_title(f'Error generating SHAP explanation\n{str(e)}', fontsize=14, color='red')
            ax.axis('off')
            
            if output_path:
                # Modify output path to indicate this is a fallback
                base, ext = os.path.splitext(output_path)
                fallback_path = f"{base}_error{ext}"
                plt.savefig(fallback_path, bbox_inches='tight')
                plt.close(fig)
                return fallback_path
            else:
                plt.show()
                plt.close(fig)
                return None
    
    def compare_models_on_image(self, models, img_path, true_class_idx, background_data=None, output_path=None):
        """
        Compare multiple models' SHAP explanations on the same image.
        
        Args:
            models (dict): Dictionary of models {model_name: model}
            img_path (str): Path to the image file
            true_class_idx (int): Index of the true class, or -1 if unknown
            background_data: Optional background data for SHAP explainer. If None, will be generated.
            output_path (str): Optional path to save the result
        
        Returns:
            str or None: Path to the saved result if output_path is provided, else None
        """
        # Preprocess the image
        img, display_img = self.preprocess_image_for_shap(img_path)
        
        # Get background data if not provided
        if background_data is None:
            background_data = self.get_background_data()
        
        # Get the number of models
        num_models = len(models)
        
        # Create a figure with subplots - one row per model
        fig, axes = plt.subplots(num_models, 3, figsize=(18, 6 * num_models))
        
        # If only one model, make axes 2D
        if num_models == 1:
            axes = np.array([axes])
        
        # Add true class label to the figure if known
        if true_class_idx >= 0 and true_class_idx < len(self.CLASS_NAMES):
            true_class_name = self.CLASS_NAMES[true_class_idx]
            fig.suptitle(f'True Class: {true_class_name}', fontsize=16, y=0.98)
        
        # Process each model
        for i, (model_name, model) in enumerate(models.items()):
            # Apply SHAP
            try:
                # Generate SHAP explanation
                shap_values, predicted_class, class_name, confidence = self.explain_image_shap(
                    model, img, background_data
                )
                
                # Display original image
                axes[i, 0].imshow(display_img)
                axes[i, 0].set_title(f'Model: {model_name.replace(".h5", "")}', fontsize=12)
                axes[i, 0].axis('off')
                
                # If SHAP values are available, visualize them
                if shap_values is not None:
                    try:
                        # Get SHAP values for the predicted class
                        shap_values_for_class = shap_values[predicted_class][0]
                        
                        # Check if shap_values_for_class has the right shape
                        if shap_values_for_class.shape != img.shape:
                            print(f"Warning: SHAP values shape {shap_values_for_class.shape} doesn't match image shape {img.shape}")
                            
                            # If it's a 2D matrix, reshape to match image dimensions
                            if len(shap_values_for_class.shape) == 2:
                                # Create a 3-channel heatmap from 2D matrix
                                height, width = shap_values_for_class.shape
                                shap_values_for_class = np.repeat(shap_values_for_class[:, :, np.newaxis], 3, axis=2)
                            
                            # If shapes still don't match, try to resize
                            if shap_values_for_class.shape != img.shape:
                                print(f"Resizing SHAP values from {shap_values_for_class.shape} to {img.shape}")
                                # Create a properly sized array and copy values
                                resized_values = np.zeros_like(img)
                                # Copy as much as possible
                                min_h = min(shap_values_for_class.shape[0], img.shape[0])
                                min_w = min(shap_values_for_class.shape[1], img.shape[1])
                                min_c = min(shap_values_for_class.shape[2], img.shape[2])
                                resized_values[:min_h, :min_w, :min_c] = shap_values_for_class[:min_h, :min_w, :min_c]
                                shap_values_for_class = resized_values
                        
                        # Sum across color channels to get a single channel heatmap
                        abs_shap = np.abs(shap_values_for_class).sum(axis=-1)  
                        
                        # Normalize for visualization
                        if np.max(abs_shap) > 0:  # Avoid division by zero
                            abs_shap = abs_shap / np.max(abs_shap)
                        
                        # Display SHAP heatmap
                        im = axes[i, 1].imshow(abs_shap, cmap='hot')
                        axes[i, 1].set_title(f'SHAP Heatmap', fontsize=12)
                        axes[i, 1].axis('off')
                        
                        # Create a combined visualization with original image and SHAP overlay
                        # Simple visualization method that avoids broadcasting issues
                        blend = img.copy() 
                        
                        # Create a heatmap in proper RGB format
                        cmap = plt.cm.hot
                        heatmap = cmap(abs_shap)[:, :, :3]  # Drop alpha channel
                        
                        # Ensure heatmap has same dimensions as the image
                        if heatmap.shape[:2] != img.shape[:2]:
                            # Resize heatmap to match image dimensions
                            from skimage.transform import resize
                            heatmap = resize(heatmap, (img.shape[0], img.shape[1], 3), anti_aliasing=True)
                        
                        # Apply alpha blending (safely)
                        alpha = 0.7
                        for j in range(3):  # For each channel
                            blend[:, :, j] = img[:, :, j] * (1 - alpha) + heatmap[:, :, j] * alpha
                        
                        # Display the blended image
                        axes[i, 2].imshow(blend)
                        
                        # Add a green border if the prediction is correct, red if incorrect
                        # Only do this if we know the true class
                        if true_class_idx >= 0:
                            if predicted_class == true_class_idx:
                                for spine in axes[i, 0].spines.values():
                                    spine.set_edgecolor('green')
                                    spine.set_linewidth(3)
                                prediction_status = "✓ Correct"
                                color = 'green'
                            else:
                                for spine in axes[i, 0].spines.values():
                                    spine.set_edgecolor('red')
                                    spine.set_linewidth(3)
                                prediction_status = "✗ Incorrect"
                                color = 'red'
                            
                            axes[i, 2].set_title(f'SHAP Overlay\nPrediction: {class_name} ({confidence:.2f})\n{prediction_status}', 
                                                fontsize=12, color=color)
                        else:
                            axes[i, 2].set_title(f'SHAP Overlay\nPrediction: {class_name} ({confidence:.2f})', 
                                                fontsize=12)
                    except Exception as e:
                        print(f"Error in model comparison visualization: {e}")
                        # If visualization fails, show error message
                        for j in range(1, 3):
                            axes[i, j].imshow(display_img)
                            axes[i, j].set_title(f'Visualization failed: {str(e)}', fontsize=12, color='red')
                            axes[i, j].axis('off')
                else:
                    # If SHAP values are not available, show error message
                    for j in range(1, 3):
                        axes[i, j].text(0.5, 0.5, "SHAP explanation failed", 
                                      ha='center', va='center', color='red', fontsize=12)
                        axes[i, j].axis('off')
                    
                    # Still show prediction information
                    if true_class_idx >= 0:
                        if predicted_class == true_class_idx:
                            prediction_status = "✓ Correct"
                            color = 'green'
                        else:
                            prediction_status = "✗ Incorrect"
                            color = 'red'
                        
                        axes[i, 0].set_title(f'Model: {model_name.replace(".h5", "")}\nPrediction: {class_name}\n{prediction_status}', 
                                           fontsize=12, color=color)
                    else:
                        axes[i, 0].set_title(f'Model: {model_name.replace(".h5", "")}\nPrediction: {class_name}', 
                                           fontsize=12)
                
                axes[i, 2].axis('off')
            
            except Exception as e:
                # If there's an error with this model, display error message
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, f"Error with {model_name}:\n{str(e)}", 
                                   ha='center', va='center', color='red', fontsize=10)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            plt.show()
            plt.close(fig)
            return None 
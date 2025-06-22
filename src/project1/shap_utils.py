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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
            # Handle different input shapes from SHAP
            if isinstance(images, list):
                images = np.array(images)
            
            # Fix shape issues - SHAP sometimes passes extra dimensions
            original_shape = images.shape
            print(f"DEBUG: Input shape to predict_fn: {original_shape}")
            
            # If we have more than 4 dimensions, we need to reshape
            if len(images.shape) > 4:
                # Flatten extra dimensions while preserving the image dimensions
                # Expected: (batch_size, height, width, channels)
                # Sometimes SHAP gives: (1, 2, height, width, channels) or similar
                
                # Find the dimensions that look like image dimensions (150, 150, 3)
                target_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)
                
                # Reshape to combine the first dimensions into batch dimension
                # and keep the last 3 dimensions as image dimensions
                if images.shape[-3:] == target_shape:
                    # Last 3 dimensions are the image, reshape everything else to batch
                    batch_size = np.prod(images.shape[:-3])
                    images = images.reshape(batch_size, *target_shape)
                else:
                    # Try to find where the image dimensions are
                    shape = images.shape
                    for i in range(len(shape) - 2):
                        if shape[i:i+3] == target_shape:
                            # Found the image dimensions, reshape accordingly
                            batch_size = np.prod(shape[:i])
                            remaining_dims = shape[i+3:]
                            if len(remaining_dims) == 0:
                                images = images.reshape(batch_size, *target_shape)
                            break
                    else:
                        # Fallback: assume the last 3 dimensions are correct and flatten the rest
                        batch_size = np.prod(shape[:-3])
                        images = images.reshape(batch_size, *shape[-3:])
            
            print(f"DEBUG: Reshaped to: {images.shape}")
            
            # Make sure images have the right shape and are normalized
            batch = np.array(images).astype('float32')
            
            # Normalize if values are not already in [0,1] range
            if np.max(batch) > 1.0:
                batch = batch / 255.0
            
            try:
                # Get model predictions
                predictions = model.predict(batch, verbose=0)
                
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
                
                print(f"DEBUG: Final predictions shape: {predictions.shape}")
                return predictions
            
            except Exception as e:
                print(f"Error in predict function: {e}")
                print(f"Batch shape that caused error: {batch.shape}")
                # Return zeros in case of error
                return np.zeros((len(batch), len(self.CLASS_NAMES)))
            
        return predict_fn
    

    def explain_image_shap(self, model, img, background_data, pred_index=None, num_samples=50):
        """
        Generate SHAP values for an image.
        
        Args:
            model: The model to explain
            img: Image array (already preprocessed)
            background_data: Background data for SHAP explainer
            pred_index: Optional class index to explain, if None uses the predicted class
            num_samples: Number of samples to use in SHAP explanation
        
        Returns:
            tuple: (shap_values, predicted_class, class_name, confidence)
        """
        import shap
        
        # Get prediction for this image
        preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        predicted_class = np.argmax(preds)
        confidence = preds[predicted_class]
        class_name = self.CLASS_NAMES[predicted_class]
        
        # Use either predicted class or provided index for explanation
        explanation_class_idx = predicted_class if pred_index is None else pred_index
        
        try:
            print(f"Creating SHAP explainer for image shape: {img.shape}")
            
            # Method 1: Try Image masker with inpainting
            print("Trying Image masker with inpainting...")
            masker = shap.maskers.Image("inpaint_telea", img.shape)
            explainer = shap.Explainer(self.model_predict_fn(model), masker)
            
            # Generate a single sample batch for SHAP analysis
            img_batch = np.expand_dims(img, axis=0)
            print(f"Input batch shape for SHAP: {img_batch.shape}")
            
            # Get SHAP values with proper parameters for image masker
            shap_values = explainer(img_batch, max_evals=min(num_samples, 100), batch_size=5)
            
            return shap_values, predicted_class, class_name, confidence
            
        except Exception as e:
            print(f"Image masker failed: {e}")
            print(f"Trying segmentation-based approach...")
            
            try:
                # Method 2: Use superpixel segmentation
                from skimage.segmentation import slic
                
                # Create superpixels - fewer segments means fewer features
                segments = slic(img, n_segments=50, compactness=10, sigma=1, 
                            start_label=1, convert2lab=True)
                
                def mask_image(mask, image, background=None):
                    """Apply mask to image using segments"""
                    if background is None:
                        background = np.zeros_like(image)
                    
                    out = image.copy()
                    for i in range(len(mask)):
                        if mask[i] == 0:  # If segment should be masked
                            out[segments == i] = background[segments == i]
                    return out
                
                def model_predict_segments(masks):
                    """Predict function that works with segment masks"""
                    batch_predictions = []
                    for mask in masks:
                        # Apply mask to the image
                        masked_img = mask_image(mask, img)
                        # Get prediction
                        pred = model.predict(np.expand_dims(masked_img, axis=0), verbose=0)
                        batch_predictions.append(pred[0])
                    return np.array(batch_predictions)
                
                # Create SHAP explainer for segments
                print(f"Created {np.max(segments)} segments for SHAP analysis")
                explainer = shap.Explainer(model_predict_segments, 
                                        np.ones((1, np.max(segments))))
                
                # Explain with segments (much fewer features now)
                shap_values = explainer(np.ones((1, np.max(segments))), 
                                    max_evals=min(num_samples, 200))
                
                # Convert segment-based SHAP values back to pixel-based
                pixel_shap = np.zeros_like(img)
                for i in range(len(shap_values.values[0])):
                    if i < np.max(segments):
                        pixel_shap[segments == (i + 1)] = shap_values.values[0][i]
                
                # Create a SHAP values object that looks like the standard format
                class SimpleShapValues:
                    def __init__(self, values, data):
                        self.values = values
                        self.data = data
                
                shap_result = SimpleShapValues([pixel_shap], [img])
                
                return shap_result, predicted_class, class_name, confidence
                
            except Exception as e2:
                print(f"Segmentation approach failed: {e2}")
                print(f"Trying simple gradient-based approach...")
                
                try:
                    # Method 3: Simple gradient-based explanation as fallback
                    import tensorflow as tf
                    
                    # Convert image to tensor
                    img_tensor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)
                    
                    with tf.GradientTape() as tape:
                        tape.watch(img_tensor)
                        predictions = model(img_tensor)
                        target_class_output = predictions[0][explanation_class_idx]
                    
                    # Get gradients
                    gradients = tape.gradient(target_class_output, img_tensor)
                    
                    # Convert to numpy and create simple SHAP-like result
                    grad_values = gradients.numpy()[0]
                    
                    class SimpleShapValues:
                        def __init__(self, values, data):
                            self.values = values
                            self.data = data
                    
                    shap_result = SimpleShapValues([grad_values], [img])
                    print("Using gradient-based explanation as fallback")
                    
                    return shap_result, predicted_class, class_name, confidence
                    
                except Exception as e3:
                    print(f"Gradient approach also failed: {e3}")
                    # Return None to indicate complete failure
                    return None, predicted_class, class_name, confidence
            
    def visualize_shap_explanation(self, img, shap_values, class_index, class_name, confidence):
        """
        Visualize SHAP explanation for a single image with pink-blue colormap.
        
        Args:
            img: Original image (already preprocessed)
            shap_values: SHAP explanation values
            class_index: Class index to explain
            class_name: Name of the class
            confidence: Model confidence for the class

        Returns:
            matplotlib.figure.Figure: Figure with visualization
        """
        import shap

        try:
            # Create the figure with multiple subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # If shap_values is None, create a simple visualization with just the original image
            if shap_values is None:
                for ax in axes:
                    ax.imshow(img)
                    ax.set_title(f"SHAP visualization failed\nClass: {class_name}, Confidence: {confidence:.2f}")
                    ax.axis('off')
                return fig
            
            # Display original image
            axes[0].imshow(img)
            axes[0].set_title(f"Original Image\nPredicted: {class_name}\nConfidence: {confidence:.2f}")
            axes[0].axis('off')
            
            # Extract SHAP values based on the format
            try:
                if hasattr(shap_values, 'values'):
                    # Standard SHAP format
                    if isinstance(shap_values.values, list):
                        attribution = shap_values.values[0]
                    else:
                        attribution = shap_values.values
                else:
                    # Our custom format from fallback methods
                    attribution = shap_values[0] if isinstance(shap_values, list) else shap_values
                
                print(f"DEBUG: Attribution shape: {attribution.shape}")
                
                # Handle different attribution shapes
                if len(attribution.shape) == 5:  # (1, H, W, C, num_classes)
                    # Take the first sample and sum across classes or take max class
                    attribution = attribution[0]  # Now (H, W, C, num_classes)
                    # Sum across all classes to get overall importance
                    attribution = np.sum(attribution, axis=-1)  # Now (H, W, C)
                elif len(attribution.shape) == 4:
                    if attribution.shape[0] == 1:  # (1, H, W, C)
                        attribution = attribution[0]
                    elif attribution.shape[-1] > 3:  # (H, W, C, num_classes)
                        # Sum across classes
                        attribution = np.sum(attribution, axis=-1)
                
                print(f"DEBUG: After processing, attribution shape: {attribution.shape}")
                
                # Create attribution map by summing across channels (preserve sign for diverging colormap)
                if len(attribution.shape) == 3:  # (H, W, C)
                    attribution_map = attribution.sum(axis=-1)
                elif len(attribution.shape) == 2:  # (H, W)
                    attribution_map = attribution
                else:
                    print(f"Unexpected attribution shape: {attribution.shape}")
                    # Flatten to 2D as fallback
                    if len(attribution.shape) > 2:
                        attribution_map = attribution.sum(axis=tuple(range(2, len(attribution.shape))))
                    else:
                        attribution_map = attribution
                
                print(f"DEBUG: Attribution map shape: {attribution_map.shape}")
                
                # Ensure attribution_map is 2D and matches image dimensions
                if len(attribution_map.shape) != 2:
                    print(f"Warning: Attribution map is not 2D, shape: {attribution_map.shape}")
                    # Try to make it 2D
                    while len(attribution_map.shape) > 2:
                        attribution_map = attribution_map.sum(axis=-1)
                
                # Resize attribution map to match image dimensions if needed
                if attribution_map.shape != img.shape[:2]:
                    print(f"Resizing attribution map from {attribution_map.shape} to {img.shape[:2]}")
                    from skimage.transform import resize
                    attribution_map = resize(attribution_map, img.shape[:2], anti_aliasing=True)
                
                # Create custom pink-blue diverging colormap (like the medical imaging example)
                colors_blue_to_pink = [
                    '#0066CC',  # Dark blue (strong negative)
                    '#3399FF',  # Medium blue
                    '#66CCFF',  # Light blue
                    '#FFFFFF',  # White (neutral)
                    '#FFB3D9',  # Light pink
                    '#FF66B3',  # Medium pink
                    '#FF0080'   # Dark pink/magenta (strong positive)
                ]
                
                pink_blue_cmap = LinearSegmentedColormap.from_list(
                    'pink_blue', colors_blue_to_pink, N=256
                )
                
                # Normalize attribution values for the colormap
                # Center around zero and make symmetric
                abs_max = np.max(np.abs(attribution_map))
                if abs_max > 0:
                    vmin, vmax = -abs_max, abs_max
                else:
                    vmin, vmax = -1, 1
                
                # Display SHAP attribution map with pink-blue colormap
                im = axes[1].imshow(attribution_map, cmap=pink_blue_cmap, vmin=vmin, vmax=vmax)
                axes[1].set_title("SHAP Attribution Map\n(Blue: Negative, Pink: Positive)")
                axes[1].axis('off')
                
                # Add colorbar with proper labels
                cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                cbar.set_label('SHAP Value', rotation=270, labelpad=15)
                
                # Create overlay visualization with transparency
                # Normalize the original image for proper blending
                img_normalized = img.copy()
                if np.max(img_normalized) > 1:
                    img_normalized = img_normalized / 255.0
                
                # Create colored attribution overlay
                # Map attribution values to colors
                norm_attribution = (attribution_map - vmin) / (vmax - vmin)  # Normalize to [0,1]
                colored_attribution = pink_blue_cmap(norm_attribution)[:, :, :3]  # Remove alpha channel
                
                # Create a mask for significant attributions (to avoid showing noise)
                significance_threshold = 0.1 * abs_max  # Only show attributions above 10% of max
                significant_mask = np.abs(attribution_map) > significance_threshold
                
                # Create the overlay
                overlay = img_normalized.copy()
                alpha = 0.6  # Transparency for the attribution overlay
                
                # Apply colored attribution only where it's significant
                for i in range(3):  # RGB channels
                    attribution_channel = colored_attribution[:, :, i]
                    # Only blend where attribution is significant
                    overlay[:, :, i] = np.where(
                        significant_mask,
                        img_normalized[:, :, i] * (1 - alpha) + attribution_channel * alpha,
                        img_normalized[:, :, i]
                    )
                
                axes[2].imshow(overlay)
                axes[2].set_title("Original + SHAP Overlay\n(Significant attributions only)")
                axes[2].axis('off')
                
                # Add a small legend explaining the colors
                legend_text = (
                    "Blue regions: Decrease prediction confidence\n"
                    "Pink regions: Increase prediction confidence\n"
                    f"Threshold: ±{significance_threshold:.1e}"
                )
                axes[2].text(0.02, 0.98, legend_text, transform=axes[2].transAxes, 
                            fontsize=9, verticalalignment='top', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as viz_error:
                print(f"Error creating detailed visualization: {viz_error}")
                
                # Simple fallback visualization without using SHAP's image_plot
                try:
                    # Just show the original image with success message
                    for ax in axes:
                        ax.imshow(img)
                        ax.set_title(f"SHAP analysis completed\nClass: {class_name}, Confidence: {confidence:.2f}")
                        ax.axis('off')
                        
                except Exception as fallback_error:
                    print(f"Fallback visualization also failed: {fallback_error}")
                    # Show original image only
                    for ax in axes:
                        ax.imshow(img)
                        ax.set_title(f"SHAP completed but visualization failed\nClass: {class_name}, Confidence: {confidence:.2f}")
                        ax.axis('off')
            
            plt.tight_layout()
            return fig
        
        except Exception as e:
            print(f"Error in SHAP visualization: {e}")
        
            # Create a simple visualization with just the original image
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img)
            ax.set_title(f"Visualization failed: {str(e)}\nClass: {class_name}, Confidence: {confidence:.2f}")
            ax.axis('off')
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
        Apply SHAP to an image and save or display the result.
        
        Args:
            img_path (str): Path to the image file
            model: The model to explain
            background_data: Background data for SHAP explainer
            pred_index: Optional class index to explain
            output_path: Optional path to save the result

        Returns:
            str or matplotlib.figure.Figure or None: Path to the saved result if output_path is provided,
                                                Figure object if output_path is None
        """
        print(f"Applying SHAP to image: {os.path.basename(img_path)}")
        print(f"Model: {model.name if hasattr(model, 'name') else 'unnamed model'}")
        
        try:
            # Preprocess the image
            img, display_img = self.preprocess_image_for_shap(img_path)
            print(f"Preprocessed image shape: {img.shape}")
            
            # Get background data if not provided
            if background_data is None:
                background_data = self.get_background_data(num_samples=5)  # Fewer samples for stability
            
            print(f"Background data shape: {background_data.shape}")
            
            # Generate SHAP explanation
            shap_values, predicted_class, class_name, confidence = self.explain_image_shap(
                model, img, background_data, pred_index=pred_index, num_samples=30
            )
            
            # Visualize the explanation
            fig = self.visualize_shap_explanation(img, shap_values, predicted_class, class_name, confidence)
            
            # Save the figure if output path is provided
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"SHAP result saved to: {output_path}")
                return output_path
            else:
                # Return the figure object without closing it
                return fig
                
        except Exception as e:
            print(f"Error applying SHAP to image: {e}")
            
            try:
                # Fallback: try to at least get the prediction
                img, display_img = self.preprocess_image_for_shap(img_path)
                preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                predicted_class = np.argmax(preds)
                confidence = preds[predicted_class]
                class_name = self.CLASS_NAMES[predicted_class]
                
                # Create a simple visualization with just the prediction
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(display_img)
                ax.set_title(f'Model: {model.name if hasattr(model, "name") else "Unknown"}\n'
                            f'Prediction: {class_name}\n'
                            f'Confidence: {confidence:.2f}\n\n'
                            f'SHAP explanation failed: {str(e)}', 
                        fontsize=12, color='red')
                ax.axis('off')
                
                if output_path:
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Fallback visualization saved to: {output_path}")
                    return output_path
                else:
                    return fig
                    
            except Exception as inner_e:
                print(f"Fallback visualization also failed: {inner_e}")
                
                # Final fallback: create an error message figure
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.text(0.5, 0.5, f"Error applying SHAP: {str(e)}\n\nFallback also failed: {str(inner_e)}", 
                        ha='center', va='center', color='red', fontsize=12, wrap=True)
                ax.axis('off')
                
                if output_path:
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    return output_path
                else:
                    return fig
    
    def compare_models_on_image(self, models, img_path, true_class_idx, background_data=None, output_path=None):
        """
        Updated compare_models_on_image method with pink-blue colormap.
        
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
        
        # Create a figure with subplots - one row per model, 3 columns
        fig, axes = plt.subplots(num_models, 3, figsize=(18, 6 * num_models))
        
        # If only one model, make axes 2D
        if num_models == 1:
            axes = np.array([axes])
        
        # Create the pink-blue colormap
        colors_blue_to_pink = [
            '#0066CC',  # Dark blue (strong negative)
            '#3399FF',  # Medium blue
            '#66CCFF',  # Light blue
            '#FFFFFF',  # White (neutral)
            '#FFB3D9',  # Light pink
            '#FF66B3',  # Medium pink
            '#FF0080'   # Dark pink/magenta (strong positive)
        ]
        
        from matplotlib.colors import LinearSegmentedColormap
        pink_blue_cmap = LinearSegmentedColormap.from_list(
            'pink_blue', colors_blue_to_pink, N=256
        )
        
        # Add true class label to the figure if known
        if true_class_idx >= 0 and true_class_idx < len(self.CLASS_NAMES):
            true_class_name = self.CLASS_NAMES[true_class_idx]
            fig.suptitle(f'True Class: {true_class_name}', fontsize=16, y=0.98)
        
        # Process each model
        for i, (model_name, model) in enumerate(models.items()):
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
                        # Extract SHAP values (similar to the main visualization function)
                        if hasattr(shap_values, 'values'):
                            if isinstance(shap_values.values, list):
                                attribution = shap_values.values[0]
                            else:
                                attribution = shap_values.values
                        else:
                            attribution = shap_values[0] if isinstance(shap_values, list) else shap_values
                        
                        # Process attribution to get 2D map
                        if len(attribution.shape) == 5:
                            attribution = attribution[0]
                            attribution = np.sum(attribution, axis=-1)
                        elif len(attribution.shape) == 4:
                            if attribution.shape[0] == 1:
                                attribution = attribution[0]
                            elif attribution.shape[-1] > 3:
                                attribution = np.sum(attribution, axis=-1)
                        
                        # Create attribution map
                        if len(attribution.shape) == 3:
                            attribution_map = attribution.sum(axis=-1)
                        elif len(attribution.shape) == 2:
                            attribution_map = attribution
                        else:
                            if len(attribution.shape) > 2:
                                attribution_map = attribution.sum(axis=tuple(range(2, len(attribution.shape))))
                            else:
                                attribution_map = attribution
                        
                        # Resize if needed
                        if attribution_map.shape != img.shape[:2]:
                            from skimage.transform import resize
                            attribution_map = resize(attribution_map, img.shape[:2], anti_aliasing=True)
                        
                        # Normalize for colormap
                        abs_max = np.max(np.abs(attribution_map))
                        if abs_max > 0:
                            vmin, vmax = -abs_max, abs_max
                        else:
                            vmin, vmax = -1, 1
                        
                        # Display SHAP attribution map
                        im = axes[i, 1].imshow(attribution_map, cmap=pink_blue_cmap, vmin=vmin, vmax=vmax)
                        axes[i, 1].set_title(f'SHAP Attribution\n(Blue: -, Pink: +)', fontsize=12)
                        axes[i, 1].axis('off')
                        
                        # Create overlay
                        img_normalized = img.copy()
                        if np.max(img_normalized) > 1:
                            img_normalized = img_normalized / 255.0
                        
                        # Create colored attribution overlay
                        norm_attribution = (attribution_map - vmin) / (vmax - vmin)
                        colored_attribution = pink_blue_cmap(norm_attribution)[:, :, :3]
                        
                        # Create overlay with transparency
                        significance_threshold = 0.1 * abs_max
                        significant_mask = np.abs(attribution_map) > significance_threshold
                        
                        overlay = img_normalized.copy()
                        alpha = 0.6
                        
                        for j in range(3):
                            attribution_channel = colored_attribution[:, :, j]
                            overlay[:, :, j] = np.where(
                                significant_mask,
                                img_normalized[:, :, j] * (1 - alpha) + attribution_channel * alpha,
                                img_normalized[:, :, j]
                            )
                        
                        axes[i, 2].imshow(overlay)
                        
                        # Add prediction status
                        if true_class_idx >= 0:
                            if predicted_class == true_class_idx:
                                prediction_status = "✓ Correct"
                                color = 'green'
                                # Add green border
                                for spine in axes[i, 0].spines.values():
                                    spine.set_edgecolor('green')
                                    spine.set_linewidth(3)
                            else:
                                prediction_status = "✗ Incorrect"
                                color = 'red'
                                # Add red border
                                for spine in axes[i, 0].spines.values():
                                    spine.set_edgecolor('red')
                                    spine.set_linewidth(3)
                            
                            axes[i, 2].set_title(f'Overlay\nPred: {class_name} ({confidence:.2f})\n{prediction_status}', 
                                                fontsize=12, color=color)
                        else:
                            axes[i, 2].set_title(f'Overlay\nPrediction: {class_name} ({confidence:.2f})', 
                                                fontsize=12)
                        
                        axes[i, 2].axis('off')
                        
                    except Exception as e:
                        print(f"Error in model comparison visualization: {e}")
                        # Show error message
                        for j in range(1, 3):
                            axes[i, j].text(0.5, 0.5, f'Visualization failed: {str(e)}', 
                                        ha='center', va='center', color='red', fontsize=10)
                            axes[i, j].axis('off')
                else:
                    # SHAP values not available
                    for j in range(1, 3):
                        axes[i, j].text(0.5, 0.5, "SHAP explanation failed", 
                                    ha='center', va='center', color='red', fontsize=12)
                        axes[i, j].axis('off')
                    
                    # Still show prediction
                    if true_class_idx >= 0:
                        if predicted_class == true_class_idx:
                            prediction_status = "✓ Correct"
                            color = 'green'
                        else:
                            prediction_status = "✗ Incorrect"
                            color = 'red'
                        
                        axes[i, 0].set_title(f'Model: {model_name.replace(".h5", "")}\nPred: {class_name}\n{prediction_status}', 
                                        fontsize=12, color=color)
                    else:
                        axes[i, 0].set_title(f'Model: {model_name.replace(".h5", "")}\nPrediction: {class_name}', 
                                        fontsize=12)
            
            except Exception as e:
                # Error with this model
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, f"Error with {model_name}:\n{str(e)}", 
                                ha='center', va='center', color='red', fontsize=10)
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # Save the figure if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return output_path
        else:
            plt.show()
            plt.close(fig)
            return None
            
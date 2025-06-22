#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General-purpose LIME utilities for visualizing and comparing model explanations.
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
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from scipy import ndimage

class LIMEUtils:
    def __init__(self, class_names, img_size=150, dataset_dir=None, models_dir=None, output_dir=None):
        """
        Initialize LIME utilities with the necessary paths and parameters.
        
        Args:
            class_names (list): List of class names
            img_size (int): Image size for model input
            dataset_dir (str): Path to the dataset directory
            models_dir (str): Path to the directory containing trained models
            output_dir (str): Path to save LIME results
        """
        self.CLASS_NAMES = class_names
        self.IMG_SIZE = img_size
        
        # Set default paths if not provided
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_DIR = dataset_dir or os.path.join(project_dir, 'dataset_organized')
        self.MODELS_DIR = models_dir or os.path.join(project_dir, 'trained_models')
        self.OUTPUT_DIR = output_dir or os.path.join(project_dir, 'lime_results')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Initialize LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
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
        Create a prediction function for LIME that processes images in batches.
        
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
                
                # Ensure the output has the right shape for LIME
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
    
    def explain_image_lime(self, model, img, pred_index=None, num_features=100, num_samples=1000):
        """
        Generate LIME explanation for a single image.
        
        Args:
            model: The model to explain
            img: Image array (already preprocessed and normalized)
            pred_index: Optional class index to explain, if None uses the predicted class
            num_features: Number of features (superpixels) to use in explanation
            num_samples: Number of samples to use in explanation
        
        Returns:
            tuple: (explanation, segments, predicted_class, class_name, confidence)
        """
        # Create prediction function for LIME
        predict_fn = self.model_predict_fn(model)
        
        # Get prediction for this image
        preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        predicted_class = np.argmax(preds)
        
        # Use either predicted class or provided index for explanation
        explanation_class_idx = predicted_class if pred_index is None else pred_index
        
        # Get confidence for the predicted class (not necessarily the explanation class)
        confidence = preds[predicted_class]
        class_name = self.CLASS_NAMES[predicted_class]
        
        # Print prediction details for debugging
        print(f"Model prediction: Class {predicted_class} ({self.CLASS_NAMES[predicted_class]}) with confidence {confidence:.4f}")
        if explanation_class_idx != predicted_class:
            print(f"Explaining class {explanation_class_idx} ({self.CLASS_NAMES[explanation_class_idx]}) instead of predicted class")
        
        # Generate segmentation for the image
        segments = slic(img, n_segments=20, compactness=10, sigma=1, start_label=1)
        
        # Explicitly include the target class in top_labels
        # Also include the top predicted class(es)
        top_indices = np.argsort(preds)[-5:]  # Get top 5 predicted classes
        top_labels = list(top_indices)
        if explanation_class_idx not in top_labels:
            top_labels.append(explanation_class_idx)
        
        # Make sure we pass the specific class of interest and a few other top classes
        try:
            # First attempt - with explicit labels
            explanation = self.explainer.explain_instance(
                img, 
                predict_fn,
                labels=top_labels,  # Explicitly specify the labels we want to explain
                hide_color=0, 
                num_features=num_features,
                num_samples=num_samples,
                segmentation_fn=lambda x: segments
            )
            
            # Test if the explanation works for the class index
            # This will throw an error if the label isn't in the explanation
            try:
                explanation.get_image_and_mask(explanation_class_idx, positive_only=True, num_features=1, hide_rest=False)
            except:
                # If that didn't work, we'll try an alternative approach below
                raise Exception("Label not in explanation")
            
        except Exception as e:
            print(f"First LIME approach failed: {e}. Trying alternative approach.")
            
            # # Second attempt - let LIME determine top labels
            # try:
            #     explanation = self.explainer.explain_instance(
            #         img, 
            #         predict_fn,
            #         top_labels=5,  # Let LIME determine the top 5 labels
            #         hide_color=0, 
            #         num_features=num_features,
            #         num_samples=num_samples * 2,  # Use more samples for better results
            #         segmentation_fn=lambda x: segments
            #     )
                
            #     # Store the top_labels in the explanation object for later use
            #     # Check if any of the top labels match our class of interest
            #     if hasattr(explanation, 'top_labels'):
            #         if explanation_class_idx not in explanation.top_labels:
            #             # If our class isn't in top_labels, we might need to change the explanation class
            #             # to one that LIME can explain
            #             if len(explanation.top_labels) > 0:
            #                 # Use the first available label instead
            #                 print(f"Target class {explanation_class_idx} not in LIME's top_labels. Using {explanation.top_labels[0]} instead.")
            #                 explanation_class_idx = explanation.top_labels[0]
            #                 # Note: We do NOT change predicted_class, only the class we're explaining
            
            # except Exception as e2:
            #     print(f"Second LIME approach also failed: {e2}. Creating a minimal explanation.")
                
            #     # Create a minimal explanation with just basic information
            #     # This won't have proper explanations but will allow the code to continue
            #     from lime.lime_base import LimeBase
                
            #     class MinimalExplanation:
            #         def __init__(self, segments, class_index):
            #             self.segments = segments
            #             self.class_index = class_index
                    
            #         def get_image_and_mask(self, label, positive_only=True, num_features=1, hide_rest=False):
            #             # Return a simple mask that highlights a few random segments
            #             mask = np.zeros_like(segments, dtype=bool)
            #             # Select a few random segments to highlight
            #             random_segments = np.random.choice(np.unique(segments), 
            #                                              size=min(5, len(np.unique(segments))), 
            #                                              replace=False)
            #             for seg in random_segments:
            #                 mask = np.logical_or(mask, segments == seg)
            #             return img, mask
                
            #     explanation = MinimalExplanation(segments, explanation_class_idx)
            #     print("Created a minimal explanation with random segments.")
        
        # Always return the predicted class (not explanation class) for correctness checks
        return explanation, segments, predicted_class, class_name, confidence
    
    def visualize_lime_explanation(self, img, explanation, segments, class_index, class_name, confidence, 
                                  positive_only=True, num_features=5, hide_rest=False):
        """
        Visualize LIME explanation for a single image.
        
        Args:
            img: Original image (already preprocessed)
            explanation: LIME explanation object
            segments: Image segments used for explanation
            class_index: Class index to explain
            class_name: Name of the class
            confidence: Model confidence for the class
            positive_only: Whether to show only positive features
            num_features: Number of top features to show
            hide_rest: Whether to hide the rest of the image (now defaults to False)
            
        Returns:
            tuple: (highlighted_image, marked_boundaries_image)
        """
        try:
            # Get the explanation for the specified class
            # We want the mask but don't want to hide the rest
            temp, mask = explanation.get_image_and_mask(
                class_index, 
                positive_only=positive_only, 
                num_features=num_features,
                hide_rest=False
            )
            
            # Create a highlighted version of the image
            highlighted_image = img.copy()
            
            # Convert important parts mask to binary (0 or 1)
            binary_mask = mask.astype(bool)
            
            # Create a grayscale version of the image
            grayscale = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
            grayscale = np.stack([grayscale] * 3, axis=-1)
            
            # Create the visualization:
            # 1. Important parts are shown in normal colors
            # 2. Non-important parts are grayscale
            # 3. Add red contour around important parts
            
            # Apply grayscale to non-important parts
            result = grayscale.copy()
            result[binary_mask] = img[binary_mask]
            
            # Add red contour around important parts
            # First, find the boundaries of the mask
            boundaries = ndimage.binary_dilation(binary_mask).astype(int) - binary_mask
            
            # Apply red color to boundaries
            result[boundaries == 1, 0] = 1.0  # Red channel
            result[boundaries == 1, 1] = 0.0  # Green channel
            result[boundaries == 1, 2] = 0.0  # Blue channel
            
            # Convert to uint8 for display
            highlighted_image = (result * 255).astype(np.uint8)
            
            # Mark the boundaries of segments on the original image
            marked_boundaries_image = mark_boundaries(img, segments)
            
            return highlighted_image, marked_boundaries_image
            
        except Exception as e:
            print(f"Error visualizing explanation: {e}")
            
            # Try to find available labels in a safer way
            available_labels = []
            try:
                # Try to access available_labels method if it exists
                if hasattr(explanation, 'available_labels'):
                    available_labels = explanation.available_labels()
                else:
                    # If method doesn't exist, try to find the top predicted classes
                    # This assumes the explanation has a 'top_labels' attribute
                    # If not, we'll catch the exception and leave available_labels empty
                    if hasattr(explanation, 'top_labels'):
                        available_labels = explanation.top_labels
                    # If we still don't have labels, try the first few class indices
                    else:
                        # Try the first few class indices to see if any work
                        for i in range(min(len(self.CLASS_NAMES), 3)):
                            try:
                                # Check if we can get a mask for this class
                                explanation.get_image_and_mask(i, positive_only=True, num_features=1, hide_rest=False)
                                available_labels.append(i)
                            except:
                                pass
            except:
                # If all else fails, just try class index 0
                available_labels = [0]
            
            # If we found at least one available label, use it
            if available_labels:
                alt_label = available_labels[0]
                print(f"Using alternative label {alt_label} instead of {class_index}")
                try:
                    temp, mask = explanation.get_image_and_mask(
                        alt_label, 
                        positive_only=positive_only, 
                        num_features=num_features,
                        hide_rest=False
                    )
                    
                    # Create a highlighted version with the alternative label
                    highlighted_image = img.copy()
                    binary_mask = mask.astype(bool)
                    
                    # Create a grayscale version of the image
                    grayscale = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                    grayscale = np.stack([grayscale] * 3, axis=-1)
                    
                    # Apply grayscale to non-important parts
                    result = grayscale.copy()
                    result[binary_mask] = img[binary_mask]
                    
                    # Add red contour around important parts
                    boundaries = ndimage.binary_dilation(binary_mask).astype(int) - binary_mask
                    result[boundaries == 1, 0] = 1.0  # Red channel
                    result[boundaries == 1, 1] = 0.0  # Green channel
                    result[boundaries == 1, 2] = 0.0  # Blue channel
                    
                    # Convert to uint8 for display
                    highlighted_image = (result * 255).astype(np.uint8)
                except Exception as inner_e:
                    print(f"Failed to use alternative label: {inner_e}")
                    # If even the alternative label fails, return a simple colored version
                    highlighted_image = (img * 255).astype(np.uint8)
            else:
                # If no labels are available, return the original image
                print("No labels available in explanation. Returning original image.")
                highlighted_image = (img * 255).astype(np.uint8)
        
        # Mark the boundaries of segments on the original image
        marked_boundaries_image = mark_boundaries(img, segments)
        
        return highlighted_image, marked_boundaries_image
    
    def preprocess_image_for_lime(self, img_path):
        """
        Load and preprocess a single image for LIME explanation.
        
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
    
    def apply_lime_to_image(self, img_path, model, pred_index=None, output_path=None, num_features=5, true_class_idx=-1):
        """
        Apply LIME to an image and save or display the result.
        
        Args:
            img_path (str): Path to the image file
            model: The model to explain
            pred_index (int): Optional class index to explain
            output_path (str): Optional path to save the result
            num_features (int): Number of top features to show
            true_class_idx (int): True class index for correctness visualization, -1 if unknown
        
        Returns:
            str or matplotlib.figure.Figure or None: Path to the saved result if output_path is provided,
                                                   Figure object if output_path is None,
                                                   None if an error occurs
        """
        # Preprocess the image
        img, display_img = self.preprocess_image_for_lime(img_path)
        
        try:
            # Generate LIME explanation
            explanation, segments, predicted_class, class_name, confidence = self.explain_image_lime(
                model, img, pred_index=pred_index, num_features=num_features
            )
            
            # Log information about the prediction for debugging
            print(f"True class index: {true_class_idx}, Predicted class: {predicted_class} ({class_name}), Confidence: {confidence:.4f}")
            
            # Determine if prediction is correct based on actual prediction, not explanation class
            is_correct = (predicted_class == true_class_idx) if true_class_idx >= 0 else None
            
            # Ensure we're explaining the class we're interested in
            # If pred_index is provided, use that; otherwise use the predicted class
            explanation_class_idx = predicted_class if pred_index is None else pred_index
            
            # Visualize the explanation for the class we want to explain
            highlighted_img, marked_img = self.visualize_lime_explanation(
                img, explanation, segments, 
                class_index=explanation_class_idx,
                class_name=class_name, confidence=confidence,
                num_features=num_features
            )
            
            # Create the visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Display original image
            ax1.imshow(display_img)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Display segments
            ax2.imshow(marked_img)
            ax2.set_title('Segmentation')
            ax2.axis('off')
            
            # Display LIME explanation with highlighted regions
            ax3.imshow(highlighted_img)
            
            # Add correctness information if true class is known
            if true_class_idx >= 0:
                prediction_text = f'LIME Explanation\nColor: Important Regions, Gray: Less Important\n'
                
                if is_correct:
                    for spine in ax1.spines.values():
                        spine.set_edgecolor('green')
                        spine.set_linewidth(3)
                    prediction_status = "✓ Correct"
                    color = 'green'
                    prediction_text += f'Class: {self.CLASS_NAMES[predicted_class]} ({confidence:.2f})\n{prediction_status}'
                else:
                    for spine in ax1.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                    true_class_name = self.CLASS_NAMES[true_class_idx]
                    prediction_status = "✗ Incorrect"
                    color = 'red'
                    prediction_text += f'Predicted: {self.CLASS_NAMES[predicted_class]} ({confidence:.2f})\nTrue: {true_class_name}\n{prediction_status}'
                
                ax3.set_title(prediction_text, color=color)
            else:
                ax3.set_title(f'LIME Explanation\nColor: Important Regions, Gray: Less Important\nClass: {class_name}\nConfidence: {confidence:.2f}')
            
            ax3.axis('off')
            
            plt.tight_layout()
            
            # Save the figure if output path is provided
            if output_path:
                plt.savefig(output_path)
                plt.close(fig)
                return output_path
            else:
                # Return the figure object without closing it or showing it
                return fig
                
        except Exception as e:
            print(f"Error applying LIME to image {os.path.basename(img_path)}: {e}")
            
            # Try with an alternative approach for DenseNet models
            try:
                print("Trying alternative approach for potentially problematic models...")
                
                # Get prediction directly
                preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                predicted_class = np.argmax(preds)
                class_index = predicted_class if pred_index is None else pred_index
                confidence = preds[class_index]
                class_name = self.CLASS_NAMES[class_index]
                
                # Log information about the prediction for debugging
                print(f"Fallback - True class index: {true_class_idx}, Predicted class: {predicted_class} ({class_name}), Confidence: {confidence:.4f}")
                
                # Determine if prediction is correct
                is_correct = (predicted_class == true_class_idx) if true_class_idx >= 0 else None
                
                # Create a simple visualization with prediction but without LIME
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Display original image
                ax1.imshow(display_img)
                ax1.set_title('Original Image')
                ax1.axis('off')
                
                # Display prediction information
                ax2.imshow(display_img)
                
                # Add correctness information if true class is known
                if true_class_idx >= 0:
                    if is_correct:
                        for spine in ax1.spines.values():
                            spine.set_edgecolor('green')
                            spine.set_linewidth(3)
                        prediction_status = "✓ Correct"
                        color = 'green'
                        ax2.set_title(f'Prediction: {self.CLASS_NAMES[predicted_class]}\nConfidence: {confidence:.2f}\n{prediction_status}', color=color)
                    else:
                        for spine in ax1.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(3)
                        true_class_name = self.CLASS_NAMES[true_class_idx]
                        prediction_status = "✗ Incorrect"
                        color = 'red'
                        ax2.set_title(f'Predicted: {self.CLASS_NAMES[predicted_class]}\nTrue: {true_class_name}\n{prediction_status}', color=color)
                else:
                    ax2.set_title(f'Prediction: {class_name}\nConfidence: {confidence:.2f}')
                
                ax2.text(10, 20, "LIME explanation failed", color='red', fontsize=12)
                ax2.text(10, 40, f"Error: {str(e)}", color='red', fontsize=10)
                ax2.axis('off')
                
                plt.tight_layout()
                
                if output_path:
                    # Modify output path to indicate this is a fallback
                    base, ext = os.path.splitext(output_path)
                    fallback_path = f"{base}_fallback{ext}"
                    plt.savefig(fallback_path)
                    plt.close(fig)
                    return fallback_path
                else:
                    # Return the figure object without closing it or showing it
                    return fig
                    
            except Exception as inner_e:
                print(f"Fallback approach also failed: {inner_e}")
                # In case the fallback also fails, return a simple error image
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                ax.text(0.5, 0.5, f"Error applying LIME: {str(e)}\nFallback error: {str(inner_e)}", 
                        ha='center', va='center', color='red')
                ax.axis('off')
                
                if output_path:
                    base, ext = os.path.splitext(output_path)
                    error_path = f"{base}_error{ext}"
                    plt.savefig(error_path)
                    plt.close(fig)
                    return error_path
                else:
                    # Return the figure object without closing it or showing it
                    return fig
    
    def analyze_correct_predictions(self, model, model_name, X_test, y_test, file_paths, samples_per_class=2, specific_indices=None):
        """
        Analyze correctly predicted images with LIME.
        
        Args:
            model: The model to analyze
            model_name (str): Name of the model
            X_test: Test data
            y_test: Test labels (one-hot encoded)
            file_paths (list): List of image file paths
            samples_per_class (int): Number of samples to analyze per class
            specific_indices (dict): Optional dictionary mapping class indices to lists of specific image indices to use
        
        Returns:
            tuple: (results, selected_indices) where results is a list of analysis results and
                  selected_indices is a dict mapping class indices to selected image indices
        """
        print(f"\nAnalyzing correct predictions for {model_name}...")
        
        # Create model output directory
        model_output_dir = os.path.join(self.OUTPUT_DIR, model_name.replace('.h5', ''))
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Get model predictions
        predictions = model.predict(X_test, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Find correct predictions for each class
        correct_indices = np.where(y_pred == y_true)[0]
        
        results = []
        selected_indices = {}
        
        # Process samples from each class
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_correct_indices = [idx for idx in correct_indices if y_true[idx] == class_idx]
            
            # Skip if no correct predictions for this class
            if not class_correct_indices:
                print(f"No correct predictions for class {class_name}")
                continue
            
            # Use specific indices if provided, otherwise sample randomly
            if specific_indices and class_idx in specific_indices:
                # Filter to make sure the specific indices are actually correct predictions
                valid_specific_indices = [idx for idx in specific_indices[class_idx] if idx in class_correct_indices]
                if valid_specific_indices:
                    sampled_indices = valid_specific_indices
                else:
                    print(f"Warning: None of the specific indices for class {class_name} are correct predictions. Sampling randomly.")
                    sampled_indices = random.sample(class_correct_indices, min(samples_per_class, len(class_correct_indices)))
            else:
                sampled_indices = random.sample(class_correct_indices, min(samples_per_class, len(class_correct_indices)))
            
            # Store the selected indices for this class
            selected_indices[class_idx] = sampled_indices
            
            for idx in sampled_indices:
                img_path = file_paths[idx]
                
                # Define output path
                img_filename = os.path.basename(img_path)
                output_filename = f"lime_correct_{class_name}_{os.path.splitext(img_filename)[0]}.png"
                output_path = os.path.join(model_output_dir, output_filename)
                
                # Apply LIME
                result_path = self.apply_lime_to_image(img_path, model, class_idx, output_path)
                results.append({
                    'model': model_name,
                    'class': class_name,
                    'image': img_filename,
                    'result_path': result_path,
                    'prediction': 'correct'
                })
        
        print(f"LIME analysis for correct predictions completed. Results saved in {model_output_dir}")
        return results, selected_indices
    
    def analyze_misclassifications(self, model, model_name, X_test, y_test, file_paths, num_samples=5, specific_indices=None):
        """
        Analyze misclassified images with LIME.
        
        Args:
            model: The model to analyze
            model_name (str): Name of the model
            X_test: Test data
            y_test: Test labels (one-hot encoded)
            file_paths (list): List of image file paths
            num_samples (int): Number of misclassified samples to analyze
            specific_indices (list): Optional list of specific image indices to use
        
        Returns:
            tuple: (results, selected_indices) where results is a list of analysis results and
                  selected_indices is a list of selected image indices
        """
        print(f"\nAnalyzing misclassifications for {model_name}...")
        
        # Create model output directory
        model_output_dir = os.path.join(self.OUTPUT_DIR, model_name.replace('.h5', ''))
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Get model predictions
        predictions = model.predict(X_test, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Find misclassified predictions
        misclassified_indices = np.where(y_pred != y_true)[0]
        
        results = []
        
        # Sample from misclassifications
        if len(misclassified_indices) > 0:
            # Use specific indices if provided, otherwise sample randomly
            if specific_indices:
                # Filter to make sure the specific indices are actually misclassifications for this model
                valid_specific_indices = [idx for idx in specific_indices if idx in misclassified_indices]
                if valid_specific_indices:
                    sampled_indices = valid_specific_indices[:num_samples]
                else:
                    print(f"Warning: None of the specific indices are misclassifications for {model_name}. Sampling randomly.")
                    sampled_indices = random.sample(list(misclassified_indices), min(num_samples, len(misclassified_indices)))
            else:
                sampled_indices = random.sample(list(misclassified_indices), min(num_samples, len(misclassified_indices)))
            
            for idx in sampled_indices:
                img_path = file_paths[idx]
                true_class = self.CLASS_NAMES[y_true[idx]]
                pred_class = self.CLASS_NAMES[y_pred[idx]]
                
                # Define output path
                img_filename = os.path.basename(img_path)
                output_filename = f"lime_misclassified_{true_class}_as_{pred_class}_{os.path.splitext(img_filename)[0]}.png"
                output_path = os.path.join(model_output_dir, output_filename)
                
                # Apply LIME to the predicted class (to see why the model thought it was this class)
                result_path = self.apply_lime_to_image(img_path, model, y_pred[idx], output_path)
                results.append({
                    'model': model_name,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'image': img_filename,
                    'result_path': result_path,
                    'prediction': 'incorrect'
                })
        else:
            print("No misclassifications found!")
            sampled_indices = []
        
        print(f"LIME analysis for misclassifications completed. Results saved in {model_output_dir}")
        return results, sampled_indices
    
    def compare_models_on_image(self, models, img_path, true_class_idx, output_path=None, num_features=5):
        """
        Compare multiple models' LIME explanations on the same image.
        
        Args:
            models (dict): Dictionary of models {model_name: model}
            img_path (str): Path to the image file
            true_class_idx (int): Index of the true class, or -1 if unknown
            output_path (str): Optional path to save the result
            num_features (int): Number of top features to show in LIME explanation
        
        Returns:
            str or matplotlib.figure.Figure or None: Path to the saved result if output_path is provided,
                                                   Figure object if output_path is None
        """
        # Preprocess the image
        img, display_img = self.preprocess_image_for_lime(img_path)
        
        # Get the number of models
        num_models = len(models)
        
        # Create a figure with subplots - one row per model plus the original image
        fig, axes = plt.subplots(num_models + 1, 3, figsize=(15, 5 * (num_models + 1)))
        
        # Display original image in the first row
        axes[0, 0].imshow(display_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Clear the other two plots in the first row
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
        
        # Add true class label to the first row if known
        if true_class_idx >= 0 and true_class_idx < len(self.CLASS_NAMES):
            true_class_name = self.CLASS_NAMES[true_class_idx]
            fig.text(0.5, 1 - 0.5/(num_models + 1), f'True Class: {true_class_name}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            fig.text(0.5, 1 - 0.5/(num_models + 1), 'True Class: Unknown', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Process each model
        for i, (model_name, model) in enumerate(models.items(), 1):
            # Apply LIME
            try:
                # Generate LIME explanation
                explanation, segments, predicted_class, class_name, confidence = self.explain_image_lime(
                    model, img, num_features=num_features
                )
                
                # Visualize explanation for predicted class with highlighting
                highlighted_img, marked_img = self.visualize_lime_explanation(
                    img, explanation, segments, 
                    class_index=predicted_class,
                    class_name=class_name, 
                    confidence=confidence,
                    num_features=num_features
                )
                
                # Display results
                axes[i, 0].imshow(display_img)
                axes[i, 0].set_title(f'Model: {model_name.replace(".h5", "")}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(marked_img)
                axes[i, 1].set_title('Segmentation')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(highlighted_img)
                
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
                    
                    axes[i, 2].set_title(f'LIME Explanation\nColor: Important, Gray: Less Important\nPrediction: {class_name} ({confidence:.2f})\n{prediction_status}', color=color)
                else:
                    # If true class is unknown, just show the prediction without correctness indication
                    axes[i, 2].set_title(f'LIME Explanation\nColor: Important, Gray: Less Important\nPrediction: {class_name} ({confidence:.2f})')
                
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
            # Return the figure object without closing it or showing it
            return fig
    
    def select_comparison_images(self, X_test, y_test_raw, file_paths, num_per_class=1):
        """
        Select representative images from each class for model comparison.
        
        Args:
            X_test: Test data
            y_test_raw: Raw test labels (class indices)
            file_paths (list): List of image file paths
            num_per_class (int): Number of samples to select per class
        
        Returns:
            list: Selected image information
        """
        comparison_images = []
        
        # Get unique classes
        unique_classes = np.unique(y_test_raw)
        
        for class_idx in unique_classes:
            # Get indices of all images in this class
            class_indices = np.where(y_test_raw == class_idx)[0]
            
            # Skip if no images for this class
            if len(class_indices) == 0:
                continue
            
            # Randomly select images from this class
            selected_indices = random.sample(list(class_indices), min(num_per_class, len(class_indices)))
            
            for idx in selected_indices:
                comparison_images.append({
                    'index': idx,
                    'class_idx': class_idx,
                    'class_name': self.CLASS_NAMES[class_idx],
                    'file_path': file_paths[idx]
                })
        
        return comparison_images
    
    def select_model_disagreement_images(self, predictions_dict, y_test_raw, file_paths, num_samples=5):
        """
        Select images where models disagree on the prediction.
        
        Args:
            predictions_dict (dict): Dictionary of model predictions {model_name: predictions}
            y_test_raw: Raw test labels (class indices)
            file_paths (list): List of image file paths
            num_samples (int): Number of disagreement samples to select
        
        Returns:
            list: Selected image information
        """
        disagreement_images = []
        
        # We need at least 2 models to find disagreements
        if len(predictions_dict) < 2:
            print("Need at least 2 models to find disagreements.")
            return disagreement_images
        
        # Create a DataFrame with predicted classes from each model
        model_predictions = {}
        for model_name, preds in predictions_dict.items():
            model_predictions[model_name] = np.argmax(preds, axis=1)
        
        df = pd.DataFrame(model_predictions)
        
        # Add true class
        df['true_class'] = y_test_raw
        
        # Find rows where there's disagreement between models
        disagreement_indices = []
        for i in range(len(df)):
            # Get predictions for this image from all models
            row = df.iloc[i, :-1]  # Exclude the true_class column
            
            # Check if there's disagreement
            if len(row.unique()) > 1:
                disagreement_indices.append(i)
        
        # Sample from disagreement images
        if disagreement_indices:
            selected_indices = random.sample(disagreement_indices, min(num_samples, len(disagreement_indices)))
            
            for idx in selected_indices:
                disagreement_images.append({
                    'index': idx,
                    'class_idx': y_test_raw[idx],
                    'class_name': self.CLASS_NAMES[y_test_raw[idx]],
                    'file_path': file_paths[idx]
                })
        
        return disagreement_images 
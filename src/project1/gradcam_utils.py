#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General-purpose GradCAM utilities for visualizing and comparing model explanations.
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

class GradCAMUtils:
    def __init__(self, class_names, img_size=150, dataset_dir=None, models_dir=None, output_dir=None):
        """
        Initialize GradCAM utilities with the necessary paths and parameters.
        
        Args:
            class_names (list): List of class names
            img_size (int): Image size for model input
            dataset_dir (str): Path to the dataset directory
            models_dir (str): Path to the directory containing trained models
            output_dir (str): Path to save GradCAM results
        """
        self.CLASS_NAMES = class_names
        self.IMG_SIZE = img_size
        
        # Set default paths if not provided
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_DIR = dataset_dir or os.path.join(project_dir, 'dataset_organized')
        self.MODELS_DIR = models_dir or os.path.join(project_dir, 'trained_models')
        self.OUTPUT_DIR = output_dir or os.path.join(project_dir, 'gradcam_results')
        
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
    
    def get_last_conv_layer(self, model):
        """
        Get the last convolutional layer in the model.
        
        Args:
            model: The model to analyze
        
        Returns:
            str: Name of the last convolutional layer
        """
        for layer in reversed(model.layers):
            # Check if the layer is a convolutional layer
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        
        # If no convolutional layer found, return the last layer before the dense layers
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Check for feature map output (batch, height, width, channels)
                return layer.name
        
        raise ValueError("Could not find a suitable convolutional layer for GradCAM.")
    
    def compute_gradcam(self, model, img_array, pred_index, layer_name):
        """
        Compute GradCAM for the given image and prediction index.
        
        Args:
            model: The model to analyze
            img_array: Input image array (batch, height, width, channels)
            pred_index: Index of the predicted class
            layer_name: Name of the layer to use for GradCAM
        
        Returns:
            numpy.ndarray: Heatmap array
        """
        # Create a model to get the feature maps and predictions
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs], 
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass through the model to get the feature maps and predictions
            conv_outputs, predictions = grad_model(img_array)
            class_prediction = predictions[:, pred_index]
        
        # Compute gradients of the predicted class with respect to the feature maps
        grads = tape.gradient(class_prediction, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map by the gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def preprocess_image_for_gradcam(self, img_path):
        """
        Load and preprocess a single image for GradCAM.
        
        Args:
            img_path (str): Path to the image file
        
        Returns:
            tuple: (img_array, display_img) where img_array is normalized and
                   display_img is the original RGB image
        """
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        
        # Create a copy for display
        display_img = img.copy()
        
        # Normalize the image
        img = img / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img, axis=0)
        
        return img_array, display_img
    
    def apply_gradcam_to_image(self, img_path, model, pred_index=None, output_path=None):
        """
        Apply GradCAM to an image and save or display the result.
        
        Args:
            img_path (str): Path to the image file
            model: The model to use
            pred_index (int): Optional class index to use for GradCAM
                             If None, will use the model's predicted class
            output_path (str): Optional path to save the result
        
        Returns:
            matplotlib.figure.Figure or str or None: Figure object if output_path is None,
                                                    path to the saved result if output_path is provided
        """
        # Preprocess the image
        img_array, display_img = self.preprocess_image_for_gradcam(img_path)
        
        # Get the last convolutional layer
        last_conv_layer = self.get_last_conv_layer(model)
        
        # Get model prediction
        preds = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(preds[0])
        
        # Use either the predicted class or provided index
        class_index = predicted_class if pred_index is None else pred_index
        class_confidence = preds[0][class_index]
        class_name = self.CLASS_NAMES[class_index]
        
        # Compute GradCAM
        heatmap = self.compute_gradcam(model, img_array, class_index, last_conv_layer)
        
        # Convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        
        # Resize heatmap to match the image size
        heatmap = cv2.resize(heatmap, (self.IMG_SIZE, self.IMG_SIZE))
        
        # Apply colormap to the heatmap
        jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert to RGB for matplotlib
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
        
        # Create a superimposed image (weighted blend of original image and heatmap)
        superimposed_img = (jet_heatmap * 0.4 + display_img * 0.6).astype(np.uint8)
        
        # Prepare the visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original image
        ax1.imshow(display_img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display heatmap
        ax2.imshow(jet_heatmap)
        ax2.set_title('GradCAM Heatmap')
        ax2.axis('off')
        
        # Display superimposed image
        ax3.imshow(superimposed_img)
        ax3.set_title(f'Class: {class_name}\nConfidence: {class_confidence:.2f}')
        ax3.axis('off')
        
        plt.tight_layout()
        
        # Save the figure if output path is provided
        if output_path:
            plt.savefig(output_path)
            plt.close(fig)
            return output_path
        else:
            # Don't show the plot or close the figure, return it for notebook display
            return fig
    
    def apply_gradcam(self, model, img_array, display_img, pred_index=None):
        """
        Apply GradCAM to an image array.
        
        Args:
            model: The model to use
            img_array: Preprocessed image array
            display_img: Original image for display
            pred_index: Optional class index to use for GradCAM
        
        Returns:
            tuple: (heatmap, superimposed_img, class_name, class_confidence, predicted_class)
        """
        # Get the last convolutional layer
        last_conv_layer = self.get_last_conv_layer(model)
        
        # Get the model prediction
        preds = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(preds[0])
        
        # Use either the predicted class or provided index
        class_index = predicted_class if pred_index is None else pred_index
        class_confidence = preds[0][class_index]
        class_name = self.CLASS_NAMES[class_index]
        
        # Compute GradCAM
        heatmap = self.compute_gradcam(model, img_array, class_index, last_conv_layer)
        
        # Convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        
        # Resize heatmap to match the image size
        heatmap = cv2.resize(heatmap, (self.IMG_SIZE, self.IMG_SIZE))
        
        # Apply colormap to the heatmap
        jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert to RGB for matplotlib
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
        
        # Create a superimposed image (weighted blend of original image and heatmap)
        superimposed_img = (jet_heatmap * 0.4 + display_img * 0.6).astype(np.uint8)
        
        return jet_heatmap, superimposed_img, class_name, class_confidence, predicted_class
    
    def compare_models_on_image(self, models, img_path, true_class_idx, output_path=None):
        """
        Compare multiple models' GradCAM results on the same image.
        
        Args:
            models (dict): Dictionary of models {model_name: model}
            img_path (str): Path to the image file
            true_class_idx (int): Index of the true class
            output_path (str): Optional path to save the result
        
        Returns:
            matplotlib.figure.Figure or str or None: Figure object if output_path is None,
                                                   path to the saved result if output_path is provided
        """
        # Preprocess the image
        img_array, display_img = self.preprocess_image_for_gradcam(img_path)
        
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
        
        # Add true class label to the first row
        true_class_name = self.CLASS_NAMES[true_class_idx]
        fig.text(0.5, 1 - 0.5/(num_models + 1), f'True Class: {true_class_name}', 
                 ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Process each model
        for i, (model_name, model) in enumerate(models.items(), 1):
            # Apply GradCAM
            try:
                heatmap, superimposed, class_name, confidence, predicted_class = self.apply_gradcam(
                    model, img_array, display_img)
                
                # Display results
                axes[i, 0].imshow(display_img)
                axes[i, 0].set_title(f'Model: {model_name.replace(".h5", "")}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(heatmap)
                axes[i, 1].set_title('GradCAM Heatmap')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(superimposed)
                
                # Add a green border if the prediction is correct, red if incorrect
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
                
                axes[i, 2].set_title(f'Prediction: {class_name} ({confidence:.2f})\n{prediction_status}', color=color)
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
            # Don't show the plot or close the figure, return it for notebook display
            return fig
    
    def analyze_correct_predictions(self, model, model_name, X_test, y_test, file_paths, samples_per_class=2, specific_indices=None):
        """
        Analyze correctly predicted images with GradCAM.
        
        Args:
            model: The model to analyze
            model_name (str): Name of the model
            X_test: Test data
            y_test: Test labels (one-hot encoded)
            file_paths (list): List of image file paths
            samples_per_class (int): Number of samples to analyze per class
            specific_indices (dict): Optional dictionary mapping class indices to lists of specific image indices to use
                                    If provided, will use these specific images instead of random sampling
        
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
                output_filename = f"correct_{class_name}_{os.path.splitext(img_filename)[0]}.png"
                output_path = os.path.join(model_output_dir, output_filename)
                
                # Apply GradCAM
                result_path = self.apply_gradcam_to_image(img_path, model, class_idx, output_path)
                results.append({
                    'model': model_name,
                    'class': class_name,
                    'image': img_filename,
                    'result_path': result_path,
                    'prediction': 'correct'
                })
        
        print(f"GradCAM analysis for correct predictions completed. Results saved in {model_output_dir}")
        return results, selected_indices
    
    def analyze_misclassifications(self, model, model_name, X_test, y_test, file_paths, num_samples=5, specific_indices=None):
        """
        Analyze misclassified images with GradCAM.
        
        Args:
            model: The model to analyze
            model_name (str): Name of the model
            X_test: Test data
            y_test: Test labels (one-hot encoded)
            file_paths (list): List of image file paths
            num_samples (int): Number of misclassified samples to analyze
            specific_indices (list): Optional list of specific image indices to use
                                    If provided, will use these specific images instead of random sampling
        
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
                output_filename = f"misclassified_{true_class}_as_{pred_class}_{os.path.splitext(img_filename)[0]}.png"
                output_path = os.path.join(model_output_dir, output_filename)
                
                # Apply GradCAM
                result_path = self.apply_gradcam_to_image(img_path, model, y_pred[idx], output_path)
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
        
        print(f"GradCAM analysis for misclassifications completed. Results saved in {model_output_dir}")
        return results, sampled_indices
    
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
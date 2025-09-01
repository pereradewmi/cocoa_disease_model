"""
Robust inference pipeline for cocoa disease segmentation.
Provides comprehensive prediction, post-processing, and analysis.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Union
import tensorflow as tf
import os


class CocoaDiseasePredictor:
    """Robust predictor for cocoa disease segmentation."""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the trained model
            input_size (tuple): Input size for the model (height, width)
        """
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.class_names = ['Background', 'Healthy', 'Black Pod Rot', 'Pod Borer']
        self.class_colors = {
            0: [0, 0, 0],        # Background - black
            1: [0, 255, 0],      # Healthy - green
            2: [255, 0, 0],      # Black pod rot - red
            3: [0, 0, 255]       # Pod borer - blue
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for model inference.
        
        Args:
            image (np.ndarray): Input image in BGR or RGB format
            
        Returns:
            tuple: (processed_image_batch, original_image_rgb, original_shape)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume input is BGR (OpenCV format), convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.copy()
        
        original_shape = image_rgb.shape[:2]
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image_rgb, original_shape
    
    def postprocess_prediction(self, prediction: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Post-process model prediction.
        
        Args:
            prediction (np.ndarray): Raw model output
            original_shape (tuple): Original image shape (height, width)
            
        Returns:
            np.ndarray: Processed segmentation mask
        """
        # Get class predictions
        predicted_mask = np.argmax(prediction[0], axis=-1)
        
        # Resize back to original image size
        predicted_mask_resized = cv2.resize(
            predicted_mask.astype(np.uint8), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Apply morphological operations to clean up the mask
        predicted_mask_cleaned = self._clean_mask(predicted_mask_resized)
        
        return predicted_mask_cleaned
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean up the prediction mask using morphological operations.
        
        Args:
            mask (np.ndarray): Raw prediction mask
            
        Returns:
            np.ndarray: Cleaned mask
        """
        cleaned_mask = mask.copy()
        
        # Define morphological kernels
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Process each class separately (except background)
        for class_id in range(1, 4):  # Skip background (0)
            class_mask = (mask == class_id).astype(np.uint8)
            
            # Remove small noise
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, small_kernel)
            
            # Fill small holes
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, medium_kernel)
            
            # Update the cleaned mask
            cleaned_mask[class_mask == 1] = class_id
            cleaned_mask[(cleaned_mask != class_id) & (class_mask == 0) & (mask == class_id)] = 0
        
        return cleaned_mask
    
    def calculate_confidence_scores(self, prediction: np.ndarray) -> Dict[str, float]:
        """
        Calculate confidence scores for the prediction.
        
        Args:
            prediction (np.ndarray): Raw model output (probabilities)
            
        Returns:
            dict: Confidence scores for each class
        """
        # Calculate mean confidence for each class
        confidences = {}
        
        for class_id, class_name in enumerate(self.class_names):
            class_probs = prediction[0, :, :, class_id]
            mean_confidence = np.mean(class_probs)
            max_confidence = np.max(class_probs)
            
            confidences[class_name] = {
                'mean_confidence': float(mean_confidence),
                'max_confidence': float(max_confidence)
            }
        
        # Overall prediction confidence (entropy-based)
        entropy = -np.sum(prediction[0] * np.log(prediction[0] + 1e-8), axis=-1)
        normalized_entropy = entropy / np.log(len(self.class_names))
        overall_confidence = 1.0 - np.mean(normalized_entropy)
        
        confidences['overall_confidence'] = float(overall_confidence)
        
        return confidences
    
    def analyze_disease_severity(self, mask: np.ndarray) -> Dict[str, Union[float, str]]:
        """
        Analyze disease severity from the segmentation mask.
        
        Args:
            mask (np.ndarray): Segmentation mask
            
        Returns:
            dict: Disease analysis results
        """
        total_pixels = mask.size
        
        # Count pixels for each class
        class_counts = {}
        for class_id, class_name in enumerate(self.class_names):
            class_counts[class_name] = int(np.sum(mask == class_id))
        
        # Calculate percentages
        background_pixels = class_counts['Background']
        pod_pixels = total_pixels - background_pixels
        
        if pod_pixels > 0:
            healthy_pct = (class_counts['Healthy'] / pod_pixels) * 100
            black_pod_rot_pct = (class_counts['Black Pod Rot'] / pod_pixels) * 100
            pod_borer_pct = (class_counts['Pod Borer'] / pod_pixels) * 100
        else:
            healthy_pct = black_pod_rot_pct = pod_borer_pct = 0.0
        
        # Calculate overall disease percentage
        disease_pixels = class_counts['Black Pod Rot'] + class_counts['Pod Borer']
        disease_percentage = (disease_pixels / pod_pixels) * 100 if pod_pixels > 0 else 0.0
        
        # Determine severity level
        if disease_percentage < 5:
            severity = "Healthy"
            severity_level = 0
            color = "green"
        elif disease_percentage < 15:
            severity = "Mild Disease"
            severity_level = 1
            color = "yellow"
        elif disease_percentage < 30:
            severity = "Moderate Disease"
            severity_level = 2
            color = "orange"
        else:
            severity = "Severe Disease"
            severity_level = 3
            color = "red"
        
        # Determine dominant disease type
        if class_counts['Black Pod Rot'] > class_counts['Pod Borer']:
            dominant_disease = "Black Pod Rot"
        elif class_counts['Pod Borer'] > class_counts['Black Pod Rot']:
            dominant_disease = "Pod Borer"
        else:
            dominant_disease = "Mixed Diseases" if disease_pixels > 0 else "No Disease"
        
        # Generate recommendation
        if severity_level == 0:
            recommendation = "Pod appears healthy. Continue regular monitoring and preventive measures."
        elif severity_level == 1:
            recommendation = "Early signs of disease detected. Increase monitoring frequency and consider preventive treatments."
        elif severity_level == 2:
            recommendation = "Moderate disease present. Implement targeted treatment based on dominant disease type."
        else:
            recommendation = "Severe disease detected. Immediate intervention required. Consider pod removal if treatment is not feasible."
        
        return {
            'severity': severity,
            'severity_level': severity_level,
            'severity_color': color,
            'disease_percentage': disease_percentage,
            'healthy_percentage': healthy_pct,
            'black_pod_rot_percentage': black_pod_rot_pct,
            'pod_borer_percentage': pod_borer_pct,
            'dominant_disease': dominant_disease,
            'total_pod_pixels': pod_pixels,
            'class_counts': class_counts,
            'recommendation': recommendation
        }
    
    def create_visualization(self, 
                           original_image: np.ndarray, 
                           prediction_mask: np.ndarray,
                           analysis_results: Dict,
                           confidence_scores: Dict,
                           save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of the prediction results.
        
        Args:
            original_image (np.ndarray): Original image
            prediction_mask (np.ndarray): Predicted segmentation mask
            analysis_results (dict): Disease analysis results
            confidence_scores (dict): Confidence scores
            save_path (str, optional): Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction mask (colored)
        colored_mask = self._create_colored_mask(prediction_mask)
        axes[0, 1].imshow(colored_mask)
        axes[0, 1].set_title('Segmentation Result', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Overlay
        overlay = self._create_overlay(original_image, prediction_mask, alpha=0.4)
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Class distribution pie chart
        class_counts = analysis_results['class_counts']
        pod_classes = {k: v for k, v in class_counts.items() if k != 'Background' and v > 0}
        
        if pod_classes:
            colors = ['green', 'red', 'blue'][:len(pod_classes)]
            axes[1, 0].pie(pod_classes.values(), labels=pod_classes.keys(), 
                          colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Pod Area Distribution', fontsize=14, fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No pod detected', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Pod Area Distribution', fontsize=14, fontweight='bold')
        
        # Confidence scores bar chart
        conf_names = []
        conf_values = []
        for class_name, conf_data in confidence_scores.items():
            if class_name != 'overall_confidence':
                conf_names.append(class_name)
                conf_values.append(conf_data['mean_confidence'])
        
        axes[1, 1].bar(conf_names, conf_values, color=['black', 'green', 'red', 'blue'])
        axes[1, 1].set_title('Mean Confidence Scores', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Analysis summary
        axes[1, 2].axis('off')
        summary_text = f"""
ANALYSIS SUMMARY

Pod Status: {analysis_results['severity']}
Overall Disease: {analysis_results['disease_percentage']:.1f}%
Dominant Issue: {analysis_results['dominant_disease']}

AREA BREAKDOWN:
• Healthy: {analysis_results['healthy_percentage']:.1f}%
• Black Pod Rot: {analysis_results['black_pod_rot_percentage']:.1f}%
• Pod Borer: {analysis_results['pod_borer_percentage']:.1f}%

CONFIDENCE:
Overall: {confidence_scores['overall_confidence']:.3f}

RECOMMENDATION:
{analysis_results['recommendation'][:100]}...
"""
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
    
    def _create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create colored visualization of the segmentation mask."""
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            colored_mask[mask == class_id] = color
        
        return colored_mask
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Create overlay of image and colored mask."""
        colored_mask = self._create_colored_mask(mask)
        
        # Resize image to match mask if needed
        if image.shape[:2] != mask.shape:
            image_resized = cv2.resize(image, (mask.shape[1], mask.shape[0]))
        else:
            image_resized = image.copy()
        
        # Create overlay
        overlay = cv2.addWeighted(image_resized, 1-alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def predict(self, 
                image: Union[str, np.ndarray], 
                visualize: bool = True,
                save_visualization: Optional[str] = None) -> Dict:
        """
        Complete prediction pipeline for a single image.
        
        Args:
            image (str or np.ndarray): Image path or image array
            visualize (bool): Whether to create visualization
            save_visualization (str, optional): Path to save visualization
            
        Returns:
            dict: Complete prediction results
        """
        # Load image if path is provided
        if isinstance(image, str):
            image_array = cv2.imread(image)
            if image_array is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            image_array = image.copy()
        
        # Preprocess
        image_batch, image_rgb, original_shape = self.preprocess_image(image_array)
        
        # Predict
        print("🔍 Making prediction...")
        raw_prediction = self.model.predict(image_batch, verbose=0)
        
        # Post-process
        prediction_mask = self.postprocess_prediction(raw_prediction, original_shape)
        
        # Calculate confidence scores
        confidence_scores = self.calculate_confidence_scores(raw_prediction)
        
        # Analyze disease severity
        analysis_results = self.analyze_disease_severity(prediction_mask)
        
        # Create visualization if requested
        if visualize:
            self.create_visualization(
                image_rgb, 
                prediction_mask, 
                analysis_results, 
                confidence_scores,
                save_visualization
            )
        
        # Compile complete results
        results = {
            'prediction_mask': prediction_mask,
            'raw_prediction': raw_prediction,
            'confidence_scores': confidence_scores,
            'analysis_results': analysis_results,
            'original_image': image_rgb,
            'original_shape': original_shape
        }
        
        return results
    
    def batch_predict(self, image_paths: list, output_dir: Optional[str] = None) -> Dict:
        """
        Predict on multiple images.
        
        Args:
            image_paths (list): List of image paths
            output_dir (str, optional): Directory to save results
            
        Returns:
            dict: Batch prediction results
        """
        print(f"🔍 Processing {len(image_paths)} images...")
        
        batch_results = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # Set visualization path
                viz_path = None
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    viz_path = os.path.join(output_dir, f"{base_name}_analysis.png")
                
                # Predict
                results = self.predict(image_path, visualize=False, save_visualization=viz_path)
                batch_results[image_path] = results
                
                # Print quick summary
                analysis = results['analysis_results']
                print(f"  Status: {analysis['severity']}")
                print(f"  Disease Level: {analysis['disease_percentage']:.1f}%")
                print(f"  Dominant Issue: {analysis['dominant_disease']}")
                
            except Exception as e:
                print(f"  ❌ Error processing {image_path}: {e}")
                batch_results[image_path] = {'error': str(e)}
        
        return batch_results


def print_prediction_summary(results: Dict):
    """Print a formatted summary of prediction results."""
    
    analysis = results['analysis_results']
    confidence = results['confidence_scores']
    
    print("\n" + "="*60)
    print("🔍 COCOA POD DISEASE ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Pod Status: {analysis['severity']}")
    print(f"Overall Disease Level: {analysis['disease_percentage']:.1f}%")
    print(f"Dominant Disease: {analysis['dominant_disease']}")
    
    print(f"\n📊 Area Distribution:")
    print(f"  • Healthy Areas: {analysis['healthy_percentage']:.1f}%")
    print(f"  • Black Pod Rot: {analysis['black_pod_rot_percentage']:.1f}%")
    print(f"  • Pod Borer: {analysis['pod_borer_percentage']:.1f}%")
    
    print(f"\n🎯 Confidence Scores:")
    print(f"  • Overall Confidence: {confidence['overall_confidence']:.3f}")
    for class_name, conf_data in confidence.items():
        if class_name != 'overall_confidence':
            print(f"  • {class_name}: {conf_data['mean_confidence']:.3f}")
    
    print(f"\n💡 Recommendation:")
    print(f"  {analysis['recommendation']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("🔍 Cocoa Disease Inference Pipeline")
    print("This module provides robust inference for cocoa disease segmentation.")
    print("\nExample usage:")
    print("  from inference import CocoaDiseasePredictor")
    print("  predictor = CocoaDiseasePredictor('model.keras')")
    print("  results = predictor.predict('image.jpg')")
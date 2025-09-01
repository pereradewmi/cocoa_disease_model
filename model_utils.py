"""
Model utilities for cocoa disease segmentation.
Includes metrics, class balancing, and training utilities.
"""

import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationMetrics:
    """Custom metrics for multi-class segmentation."""
    
    def __init__(self, num_classes=4, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ['Background', 'Healthy', 'Black Pod Rot', 'Pod Borer']
    
    def intersection_over_union(self, y_true, y_pred, class_id):
        """Calculate IoU for a specific class."""
        y_true_class = tf.cast(tf.equal(y_true, class_id), tf.float32)
        y_pred_class = tf.cast(tf.equal(y_pred, class_id), tf.float32)
        
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        union = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class) - intersection
        
        # Avoid division by zero
        iou = tf.where(union > 0, intersection / union, 1.0)
        return iou
    
    def mean_iou(self, y_true, y_pred):
        """Calculate mean IoU across all classes."""
        ious = []
        for class_id in range(self.num_classes):
            iou = self.intersection_over_union(y_true, y_pred, class_id)
            ious.append(iou)
        return tf.reduce_mean(ious)
    
    def dice_coefficient(self, y_true, y_pred, class_id):
        """Calculate Dice coefficient for a specific class."""
        y_true_class = tf.cast(tf.equal(y_true, class_id), tf.float32)
        y_pred_class = tf.cast(tf.equal(y_pred, class_id), tf.float32)
        
        intersection = tf.reduce_sum(y_true_class * y_pred_class)
        total = tf.reduce_sum(y_true_class) + tf.reduce_sum(y_pred_class)
        
        # Avoid division by zero
        dice = tf.where(total > 0, 2.0 * intersection / total, 1.0)
        return dice
    
    def mean_dice(self, y_true, y_pred):
        """Calculate mean Dice coefficient across all classes."""
        dice_scores = []
        for class_id in range(self.num_classes):
            dice = self.dice_coefficient(y_true, y_pred, class_id)
            dice_scores.append(dice)
        return tf.reduce_mean(dice_scores)


def create_custom_metrics(num_classes=4):
    """Create custom metric functions for training."""
    metrics_calculator = SegmentationMetrics(num_classes)
    
    def mean_iou_metric(y_true, y_pred):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        return metrics_calculator.mean_iou(y_true, y_pred_classes)
    
    def mean_dice_metric(y_true, y_pred):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        return metrics_calculator.mean_dice(y_true, y_pred_classes)
    
    def healthy_iou(y_true, y_pred):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        return metrics_calculator.intersection_over_union(y_true, y_pred_classes, 1)
    
    def disease_iou(y_true, y_pred):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        # Average IoU for disease classes (2 and 3)
        iou_2 = metrics_calculator.intersection_over_union(y_true, y_pred_classes, 2)
        iou_3 = metrics_calculator.intersection_over_union(y_true, y_pred_classes, 3)
        return (iou_2 + iou_3) / 2.0
    
    return {
        'mean_iou': mean_iou_metric,
        'mean_dice': mean_dice_metric,
        'healthy_iou': healthy_iou,
        'disease_iou': disease_iou
    }


def calculate_class_weights(mask_directory, num_classes=4, method='balanced'):
    """
    Calculate class weights from mask files to handle imbalanced data.
    
    Args:
        mask_directory (str): Directory containing mask files
        num_classes (int): Number of classes
        method (str): 'balanced' or 'frequency'
        
    Returns:
        dict: Class weights dictionary
    """
    import glob
    import cv2
    from collections import Counter
    
    print(f"📊 Calculating class weights from masks in: {mask_directory}")
    
    # Find all mask files
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        mask_files.extend(glob.glob(os.path.join(mask_directory, ext)))
    
    if not mask_files:
        print("No mask files found. Using equal weights.")
        return {i: 1.0 for i in range(num_classes)}
    
    # Count pixels for each class
    class_counts = Counter()
    total_pixels = 0
    
    print(f"Processing {len(mask_files)} mask files...")
    
    for i, mask_path in enumerate(mask_files):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(mask_files)} masks")
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Count pixels for each class
        unique, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique, counts):
            if 0 <= class_id < num_classes:
                class_counts[class_id] += count
                total_pixels += count
    
    print(f"Total pixels analyzed: {total_pixels:,}")
    
    # Calculate weights
    if method == 'balanced':
        # Use sklearn's balanced class weight calculation
        classes = list(range(num_classes))
        y = []
        for class_id in classes:
            count = class_counts.get(class_id, 1)  # Avoid zero counts
            y.extend([class_id] * min(count, 10000))  # Sample to avoid memory issues
        
        if len(y) > 0:
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
        else:
            weights_dict = {i: 1.0 for i in range(num_classes)}
            
    elif method == 'frequency':
        # Inverse frequency weighting
        weights_dict = {}
        for class_id in range(num_classes):
            count = class_counts.get(class_id, 1)
            frequency = count / total_pixels
            weights_dict[class_id] = 1.0 / frequency if frequency > 0 else 1.0
            
        # Normalize weights
        max_weight = max(weights_dict.values())
        weights_dict = {k: v / max_weight for k, v in weights_dict.items()}
    
    # Print class distribution and weights
    print("\n📈 Class Distribution and Weights:")
    class_names = ['Background', 'Healthy', 'Black Pod Rot', 'Pod Borer']
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        weight = weights_dict.get(class_id, 1.0)
        name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        print(f"  {name}: {count:,} pixels ({percentage:.2f}%) -> weight: {weight:.3f}")
    
    return weights_dict


def create_weighted_loss(class_weights):
    """
    Create a weighted sparse categorical crossentropy loss function.
    
    Args:
        class_weights (dict): Dictionary of class weights
        
    Returns:
        function: Weighted loss function
    """
    def weighted_sparse_categorical_crossentropy(y_true, y_pred):
        # Convert class weights to tensor
        weights = tf.constant([class_weights.get(i, 1.0) for i in range(len(class_weights))])
        
        # Get the class weights for each pixel
        y_true_int = tf.cast(y_true, tf.int32)
        sample_weights = tf.gather(weights, y_true_int)
        
        # Calculate the standard sparse categorical crossentropy
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Apply weights
        weighted_loss = loss * sample_weights
        
        return tf.reduce_mean(weighted_loss)
    
    return weighted_sparse_categorical_crossentropy


def evaluate_model_detailed(model, test_dataset, class_names=None):
    """
    Perform detailed evaluation of the segmentation model.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        class_names: List of class names
        
    Returns:
        dict: Detailed evaluation results
    """
    if class_names is None:
        class_names = ['Background', 'Healthy', 'Black Pod Rot', 'Pod Borer']
    
    print("🔍 Performing detailed model evaluation...")
    
    # Collect predictions and ground truth
    all_y_true = []
    all_y_pred = []
    
    for batch_images, batch_masks in test_dataset:
        predictions = model.predict(batch_images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=-1)
        
        # Flatten for easier analysis
        y_true_flat = batch_masks.numpy().flatten()
        y_pred_flat = predicted_classes.flatten()
        
        all_y_true.extend(y_true_flat)
        all_y_pred.extend(y_pred_flat)
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Calculate metrics
    metrics_calculator = SegmentationMetrics(len(class_names), class_names)
    
    # Per-class IoU and Dice
    per_class_iou = []
    per_class_dice = []
    
    for class_id in range(len(class_names)):
        # Calculate IoU
        true_class = (all_y_true == class_id)
        pred_class = (all_y_pred == class_id)
        
        intersection = np.sum(true_class & pred_class)
        union = np.sum(true_class | pred_class)
        iou = intersection / union if union > 0 else 0.0
        per_class_iou.append(iou)
        
        # Calculate Dice
        dice = 2.0 * intersection / (np.sum(true_class) + np.sum(pred_class)) if (np.sum(true_class) + np.sum(pred_class)) > 0 else 0.0
        per_class_dice.append(dice)
    
    # Overall metrics
    mean_iou = np.mean(per_class_iou)
    mean_dice = np.mean(per_class_dice)
    
    # Accuracy
    accuracy = np.mean(all_y_true == all_y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=range(len(class_names)))
    
    # Classification report
    report = classification_report(all_y_true, all_y_pred, target_names=class_names, output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'per_class_iou': {class_names[i]: per_class_iou[i] for i in range(len(class_names))},
        'per_class_dice': {class_names[i]: per_class_dice[i] for i in range(len(class_names))},
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results


def print_evaluation_results(results, class_names=None):
    """Print formatted evaluation results."""
    
    if class_names is None:
        class_names = ['Background', 'Healthy', 'Black Pod Rot', 'Pod Borer']
    
    print("\n" + "="*60)
    print("📊 DETAILED MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    
    print(f"\n📈 Per-Class IoU Scores:")
    for class_name, iou in results['per_class_iou'].items():
        print(f"  {class_name}: {iou:.4f}")
    
    print(f"\n📈 Per-Class Dice Scores:")
    for class_name, dice in results['per_class_dice'].items():
        print(f"  {class_name}: {dice:.4f}")
    
    print(f"\n📈 Per-Class Precision/Recall/F1:")
    for class_name in class_names:
        if class_name in results['classification_report']:
            metrics = results['classification_report'][class_name]
            print(f"  {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print("\n" + "="*60)


def plot_confusion_matrix(confusion_matrix, class_names=None, figsize=(8, 6)):
    """Plot confusion matrix with proper formatting."""
    
    if class_names is None:
        class_names = ['Background', 'Healthy', 'Black Pod Rot', 'Pod Borer']
    
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.3f', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()
    
    # Also show raw counts
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()


def create_training_callbacks(model_save_path, patience=10):
    """
    Create comprehensive training callbacks.
    
    Args:
        model_save_path (str): Path to save the best model
        patience (int): Early stopping patience
        
    Returns:
        list: List of callbacks
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_mean_iou',
            mode='max',
            save_best_only=True,
            verbose=1,
            save_format='keras'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mean_iou',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mean_iou',
            mode='max',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            model_save_path.replace('.keras', '_training_log.csv'),
            append=False
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path=None):
    """Plot training history with metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    if 'accuracy' in history.history:
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
    
    # Mean IoU
    if 'mean_iou' in history.history:
        axes[1, 0].plot(history.history['mean_iou'], label='Training IoU')
        if 'val_mean_iou' in history.history:
            axes[1, 0].plot(history.history['val_mean_iou'], label='Validation IoU')
        axes[1, 0].set_title('Mean IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("🔧 Model Utilities for Cocoa Disease Segmentation")
    print("This module provides training utilities, metrics, and evaluation tools.")
    print("\nExample usage:")
    print("  from model_utils import create_custom_metrics, calculate_class_weights")
    print("  metrics = create_custom_metrics()")
    print("  weights = calculate_class_weights('/path/to/masks')")
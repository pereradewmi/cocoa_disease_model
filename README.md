# Enhanced Cocoa Disease Segmentation Model

A comprehensive deep learning solution for automated detection and segmentation of cocoa pod diseases using U-Net architecture with significant improvements to address critical issues in the original implementation.

## 🚀 Key Improvements

### ✅ Fixed Critical Issues

1. **Binary Output Problem Resolved**
   - Original model only output 0 and 1 pixel values
   - **Fixed**: Now properly outputs all 4 classes (0=background, 1=healthy, 2=black_pod_rot, 3=pod_borer)

2. **Enhanced Mask Creation Logic**
   - Original `combine_masks()` had overlap handling issues
   - **Fixed**: Proper overlap handling with configurable priority system
   - Validation of mask pixel value ranges

3. **Improved Training Configuration**
   - Original model showed unrealistic 100% accuracy after epoch 1
   - **Fixed**: Class balancing, proper loss weighting, better metrics
   - More realistic training progression

4. **Robust Data Pipeline**
   - Added comprehensive data validation
   - Improved image-mask pairing verification
   - Data augmentation for better generalization

5. **Enhanced Inference**
   - Post-processing with morphological operations
   - Confidence scoring and uncertainty estimation
   - Disease severity assessment with recommendations

## 📁 Project Structure

```
cocoa_disease_model/
├── Enhanced_Segmentation_model.ipynb  # Main enhanced notebook
├── data_validation.py                 # Data validation utilities
├── model_utils.py                     # Model utilities and metrics
├── inference.py                       # Robust inference pipeline
├── README.md                          # This documentation
└── Segmentation_model_final.ipynb     # Original notebook (reference)
```

## 🛠️ Installation and Setup

### Prerequisites

```bash
# Required packages
pip install tensorflow>=2.8.0
pip install opencv-python>=4.5.0
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.0.0
pip install numpy>=1.19.0
```

### Data Structure

Organize your data in the following structure:

```
data/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── masks/
    ├── train/
    │   ├── healthy/        # Healthy class masks
    │   ├── black_pod_rot/  # Black pod rot class masks
    │   ├── pod_borer/      # Pod borer class masks
    │   └── Multiclass/     # Combined multiclass masks (auto-generated)
    └── val/
        ├── healthy/
        ├── black_pod_rot/
        ├── pod_borer/
        └── Multiclass/     # Combined multiclass masks (auto-generated)
```

## 🚀 Quick Start

### 1. Training the Enhanced Model

```python
from data_validation import analyze_mask_directory, print_analysis_summary
from model_utils import calculate_class_weights, create_custom_metrics
from inference import CocoaDiseasePredictor

# Validate your data first
results = analyze_mask_directory('/path/to/masks')
print_analysis_summary(results)

# Calculate class weights for balanced training
class_weights = calculate_class_weights('/path/to/training/masks')

# Train using the enhanced notebook
# Open Enhanced_Segmentation_model.ipynb and follow the steps
```

### 2. Making Predictions

```python
# Load the trained model
predictor = CocoaDiseasePredictor('enhanced_model.keras')

# Single image prediction
results = predictor.predict('path/to/image.jpg', visualize=True)

# Batch prediction
batch_results = predictor.batch_predict([
    'image1.jpg', 'image2.jpg', 'image3.jpg'
])

# Print detailed analysis
from inference import print_prediction_summary
print_prediction_summary(results)
```

### 3. Data Validation

```python
from data_validation import (
    validate_mask_content, 
    analyze_mask_directory,
    validate_image_mask_pairs,
    visualize_mask_samples
)

# Validate individual masks
result = validate_mask_content('path/to/mask.png')

# Analyze entire directory
analysis = analyze_mask_directory('/path/to/masks')
print_analysis_summary(analysis)

# Validate image-mask pairs
pairs = validate_image_mask_pairs('/path/to/images', '/path/to/masks')

# Visualize samples
visualize_mask_samples('/path/to/masks', num_samples=4)
```

## 🎯 Model Performance

### Class Definitions
- **Class 0 (Background)**: Non-pod regions
- **Class 1 (Healthy)**: Healthy cocoa pod tissue
- **Class 2 (Black Pod Rot)**: Areas affected by black pod rot disease
- **Class 3 (Pod Borer)**: Areas affected by pod borer damage

### Enhanced Metrics
- **Mean IoU**: Intersection over Union averaged across all classes
- **Mean Dice**: Dice coefficient averaged across all classes
- **Per-class IoU/Dice**: Individual performance for each class
- **Confusion Matrix**: Detailed classification breakdown
- **Confidence Scores**: Model certainty for predictions

### Expected Performance Improvements
- ✅ Proper 4-class segmentation (vs. binary output)
- ✅ Realistic training accuracy progression (vs. immediate 100%)
- ✅ Better disease detection and classification
- ✅ Robust performance on unseen data

## 🔧 Advanced Usage

### Custom Training Configuration

```python
from model_utils import create_training_callbacks, create_weighted_loss

# Custom class weights
class_weights = {0: 0.1, 1: 1.0, 2: 5.0, 3: 5.0}

# Custom loss function
weighted_loss = create_weighted_loss(class_weights)

# Enhanced callbacks
callbacks = create_training_callbacks(
    model_save_path='my_model.keras',
    patience=15
)
```

### Batch Processing

```python
# Process multiple images
predictor = CocoaDiseasePredictor('model.keras')

image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.batch_predict(
    image_paths, 
    output_dir='analysis_results'
)

# Generate summary report
for path, result in results.items():
    if 'error' not in result:
        analysis = result['analysis_results']
        print(f"{path}: {analysis['severity']} - {analysis['disease_percentage']:.1f}% disease")
```

### Model Evaluation

```python
from model_utils import evaluate_model_detailed, plot_confusion_matrix

# Detailed evaluation
results = evaluate_model_detailed(model, test_dataset)

# Plot results
plot_confusion_matrix(results['confusion_matrix'])
print_evaluation_results(results)
```

## 🔍 Data Validation Tools

### Mask Quality Assessment

```python
# Comprehensive mask analysis
from data_validation import analyze_mask_directory

analysis = analyze_mask_directory('/path/to/masks', sample_size=100)

# Key metrics:
# - Valid masks (classes 0-3 only)
# - Masks with all 4 classes
# - Class distribution
# - Problematic masks identification
```

### Data Quality Issues Detection

The validation tools automatically detect:
- Invalid pixel values (outside 0-3 range)
- Missing class representations
- Image-mask size mismatches
- Corrupted files

## 🎨 Visualization Features

### Comprehensive Prediction Visualization

The enhanced inference pipeline provides:
1. **Original Image**: Input image
2. **Segmentation Result**: Colored class predictions
3. **Overlay**: Transparent overlay on original image
4. **Class Distribution**: Pie chart of area percentages
5. **Confidence Scores**: Model certainty for each class
6. **Analysis Summary**: Disease severity and recommendations

### Training Progress Visualization

```python
from model_utils import plot_training_history

# Automatically plots:
# - Loss curves (training/validation)
# - Accuracy progression
# - IoU metrics
# - Learning rate schedule
plot_training_history(history)
```

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **"No matching image-mask pairs found"**
   - Check file naming consistency
   - Ensure same base names for images and masks
   - Verify file extensions (.jpg, .png)

2. **"Invalid pixel values in masks"**
   - Use `data_validation.py` to identify problematic masks
   - Values should be 0, 1, 2, or 3 only
   - Use the mask fixing utilities

3. **"Model predicting only background"**
   - Check class weights calculation
   - Verify mask creation with overlap handling
   - Ensure proper data augmentation

4. **Poor model performance**
   - Increase training data
   - Adjust class weights
   - Use data augmentation
   - Check data quality with validation tools

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate data thoroughly before training
from data_validation import validate_image_mask_pairs
pairs = validate_image_mask_pairs(image_dir, mask_dir, sample_size=20)
```

## 📊 Model Architecture

### Enhanced U-Net Features
- **Batch Normalization**: Stable training and faster convergence
- **Dropout Regularization**: Prevents overfitting
- **Skip Connections**: Better gradient flow and feature preservation
- **Softmax Output**: Proper multi-class probability distribution

### Training Enhancements
- **Class Balancing**: Automatic weight calculation for imbalanced data
- **Advanced Callbacks**: Model checkpointing, early stopping, learning rate scheduling
- **Comprehensive Metrics**: IoU, Dice, per-class metrics
- **Data Augmentation**: Rotation, flipping, brightness/contrast adjustment

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests and documentation
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/your-repo/cocoa_disease_model.git
cd cocoa_disease_model
pip install -r requirements.txt
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original model inspiration from cocoa disease detection research
- U-Net architecture based on Ronneberger et al.
- Community contributions for data validation and testing

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the comprehensive validation tools
3. Open an issue with detailed error information
4. Include sample data and configuration for reproducibility

---

**Note**: This enhanced version addresses all critical issues identified in the original implementation, providing a robust and production-ready solution for cocoa disease segmentation.
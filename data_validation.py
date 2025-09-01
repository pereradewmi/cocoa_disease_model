"""
Data validation utilities for cocoa disease segmentation model.
Provides comprehensive validation of images, masks, and class distributions.
"""

import cv2
import numpy as np
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt


def validate_mask_content(mask_path, verbose=True):
    """
    Validate mask content and return class distribution.
    
    Args:
        mask_path (str): Path to mask file
        verbose (bool): Whether to print detailed info
        
    Returns:
        dict: Class distribution and validation results
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"error": f"Could not read mask: {mask_path}"}
    
    unique_values = np.unique(mask)
    class_counts = Counter(mask.flatten())
    
    # Expected classes: 0=background, 1=healthy, 2=black_pod_rot, 3=pod_borer
    expected_classes = {0, 1, 2, 3}
    found_classes = set(unique_values)
    
    results = {
        "mask_path": mask_path,
        "unique_values": unique_values.tolist(),
        "class_counts": dict(class_counts),
        "found_classes": found_classes,
        "expected_classes": expected_classes,
        "valid_classes": found_classes.issubset(expected_classes),
        "has_all_classes": expected_classes.issubset(found_classes),
        "mask_shape": mask.shape,
        "total_pixels": mask.size
    }
    
    if verbose:
        print(f"Mask: {os.path.basename(mask_path)}")
        print(f"  Shape: {mask.shape}")
        print(f"  Unique values: {unique_values}")
        print(f"  Class distribution: {dict(class_counts)}")
        print(f"  Valid classes only: {results['valid_classes']}")
        print(f"  Has all 4 classes: {results['has_all_classes']}")
        
        if not results['valid_classes']:
            unexpected = found_classes - expected_classes
            print(f"  ⚠️  Unexpected values: {unexpected}")
            
    return results


def analyze_mask_directory(mask_dir, sample_size=None):
    """
    Analyze all masks in a directory for validation issues.
    
    Args:
        mask_dir (str): Directory containing mask files
        sample_size (int): Number of masks to analyze (None for all)
        
    Returns:
        dict: Comprehensive analysis results
    """
    print(f"🔍 Analyzing masks in: {mask_dir}")
    
    # Find all mask files
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext)))
    
    if not mask_files:
        return {"error": f"No mask files found in {mask_dir}"}
    
    # Sample if requested
    if sample_size and len(mask_files) > sample_size:
        import random
        mask_files = random.sample(mask_files, sample_size)
    
    print(f"Found {len(mask_files)} mask files to analyze")
    
    # Analyze each mask
    all_results = []
    overall_class_counts = Counter()
    invalid_masks = []
    masks_with_unexpected_values = []
    
    for i, mask_path in enumerate(mask_files):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(mask_files)}...")
            
        result = validate_mask_content(mask_path, verbose=False)
        
        if "error" in result:
            invalid_masks.append(result)
            continue
            
        all_results.append(result)
        
        # Accumulate class counts
        for class_id, count in result["class_counts"].items():
            overall_class_counts[class_id] += count
            
        # Track problematic masks
        if not result["valid_classes"]:
            masks_with_unexpected_values.append(result)
    
    # Calculate statistics
    total_masks = len(all_results)
    valid_masks = sum(1 for r in all_results if r["valid_classes"])
    masks_with_all_classes = sum(1 for r in all_results if r["has_all_classes"])
    
    # Class distribution analysis
    total_pixels = sum(overall_class_counts.values())
    class_percentages = {
        class_id: (count / total_pixels) * 100 
        for class_id, count in overall_class_counts.items()
    }
    
    summary = {
        "total_masks_found": len(mask_files),
        "successfully_analyzed": total_masks,
        "invalid_masks": len(invalid_masks),
        "valid_masks": valid_masks,
        "masks_with_all_classes": masks_with_all_classes,
        "masks_with_unexpected_values": len(masks_with_unexpected_values),
        "overall_class_counts": dict(overall_class_counts),
        "class_percentages": class_percentages,
        "problematic_masks": masks_with_unexpected_values[:10],  # Show first 10
        "sample_mask_results": all_results[:5]  # Show first 5 detailed results
    }
    
    return summary


def print_analysis_summary(analysis_results):
    """Print a formatted summary of mask analysis results."""
    
    print("\n" + "="*60)
    print("📊 MASK ANALYSIS SUMMARY")
    print("="*60)
    
    if "error" in analysis_results:
        print(f"❌ Error: {analysis_results['error']}")
        return
    
    # Basic statistics
    print(f"Total masks found: {analysis_results['total_masks_found']}")
    print(f"Successfully analyzed: {analysis_results['successfully_analyzed']}")
    print(f"Invalid masks: {analysis_results['invalid_masks']}")
    print(f"Valid masks (classes 0-3 only): {analysis_results['valid_masks']}")
    print(f"Masks with all 4 classes: {analysis_results['masks_with_all_classes']}")
    print(f"Masks with unexpected values: {analysis_results['masks_with_unexpected_values']}")
    
    # Class distribution
    print(f"\n📈 CLASS DISTRIBUTION:")
    for class_id in sorted(analysis_results['class_percentages'].keys()):
        count = analysis_results['overall_class_counts'][class_id]
        percentage = analysis_results['class_percentages'][class_id]
        class_name = {0: "Background", 1: "Healthy", 2: "Black Pod Rot", 3: "Pod Borer"}.get(class_id, f"Class {class_id}")
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")
    
    # Issues found
    if analysis_results['masks_with_unexpected_values'] > 0:
        print(f"\n⚠️  ISSUES DETECTED:")
        print(f"Found {analysis_results['masks_with_unexpected_values']} masks with unexpected pixel values")
        print("Sample problematic masks:")
        for mask_result in analysis_results['problematic_masks']:
            unexpected = set(mask_result['found_classes']) - set(mask_result['expected_classes'])
            print(f"  - {os.path.basename(mask_result['mask_path'])}: unexpected values {unexpected}")
    
    print("\n" + "="*60)


def fix_mask_values(mask_path, output_path=None):
    """
    Fix mask by clamping values to valid range [0, 3].
    
    Args:
        mask_path (str): Input mask path
        output_path (str): Output path (if None, overwrites input)
        
    Returns:
        dict: Fix results
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"error": f"Could not read mask: {mask_path}"}
    
    original_unique = np.unique(mask)
    
    # Clamp values to valid range [0, 3]
    fixed_mask = np.clip(mask, 0, 3)
    
    # If values > 3 exist, map them intelligently
    if np.max(mask) > 3:
        # Map high values to background (0)
        fixed_mask[mask > 3] = 0
    
    new_unique = np.unique(fixed_mask)
    
    # Save fixed mask
    if output_path is None:
        output_path = mask_path
    
    cv2.imwrite(output_path, fixed_mask)
    
    return {
        "original_path": mask_path,
        "output_path": output_path,
        "original_unique_values": original_unique.tolist(),
        "fixed_unique_values": new_unique.tolist(),
        "changes_made": not np.array_equal(original_unique, new_unique)
    }


def visualize_mask_samples(mask_dir, num_samples=4):
    """
    Visualize sample masks to understand the data.
    
    Args:
        mask_dir (str): Directory containing masks
        num_samples (int): Number of samples to show
    """
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        print(f"No PNG mask files found in {mask_dir}")
        return
    
    # Select sample files
    import random
    sample_files = random.sample(mask_files, min(num_samples, len(mask_files)))
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, mask_path in enumerate(sample_files):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Original mask
        axes[0, i].imshow(mask, cmap='tab10', vmin=0, vmax=3)
        axes[0, i].set_title(f"Original\n{os.path.basename(mask_path)}")
        axes[0, i].axis('off')
        
        # Colored mask for better visualization
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask == 0] = [0, 0, 0]        # Background - black
        colored_mask[mask == 1] = [0, 255, 0]      # Healthy - green
        colored_mask[mask == 2] = [255, 0, 0]      # Black pod rot - red
        colored_mask[mask == 3] = [0, 0, 255]      # Pod borer - blue
        
        axes[1, i].imshow(colored_mask)
        axes[1, i].set_title(f"Colored\nBG:Black, H:Green, BPR:Red, PB:Blue")
        axes[1, i].axis('off')
        
        # Print class distribution
        unique, counts = np.unique(mask, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"\n{os.path.basename(mask_path)}:")
        print(f"  Class distribution: {class_dist}")
    
    plt.tight_layout()
    plt.show()


def validate_image_mask_pairs(image_dir, mask_dir, sample_size=10):
    """
    Validate that images and masks are properly paired and compatible.
    
    Args:
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        sample_size (int): Number of pairs to validate
        
    Returns:
        dict: Validation results
    """
    print(f"🔍 Validating image-mask pairs...")
    
    # Get image and mask files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    mask_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext)))
    
    # Create dictionaries for matching
    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
    mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_files}
    
    # Find common keys
    common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
    
    if not common_keys:
        return {
            "error": "No matching image-mask pairs found",
            "image_count": len(image_files),
            "mask_count": len(mask_files),
            "sample_image_keys": list(image_dict.keys())[:5],
            "sample_mask_keys": list(mask_dict.keys())[:5]
        }
    
    # Sample pairs to validate
    sample_keys = common_keys[:sample_size] if len(common_keys) > sample_size else common_keys
    
    validation_results = []
    
    for key in sample_keys:
        image_path = image_dict[key]
        mask_path = mask_dict[key]
        
        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            validation_results.append({
                "key": key,
                "error": f"Could not load image or mask",
                "image_path": image_path,
                "mask_path": mask_path
            })
            continue
        
        # Validate shapes and contents
        result = {
            "key": key,
            "image_path": image_path,
            "mask_path": mask_path,
            "image_shape": image.shape,
            "mask_shape": mask.shape,
            "shapes_compatible": image.shape[:2] == mask.shape,
            "mask_unique_values": np.unique(mask).tolist(),
            "mask_valid": set(np.unique(mask)).issubset({0, 1, 2, 3})
        }
        
        validation_results.append(result)
    
    # Summary statistics
    total_pairs = len(common_keys)
    validated_pairs = len(validation_results)
    valid_pairs = sum(1 for r in validation_results if r.get("shapes_compatible", False) and r.get("mask_valid", False))
    
    summary = {
        "total_possible_pairs": total_pairs,
        "validated_pairs": validated_pairs,
        "valid_pairs": valid_pairs,
        "validation_results": validation_results,
        "image_count": len(image_files),
        "mask_count": len(mask_files)
    }
    
    print(f"  Total possible pairs: {total_pairs}")
    print(f"  Validated: {validated_pairs}")
    print(f"  Valid pairs: {valid_pairs}")
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("🔍 Data Validation Utilities for Cocoa Disease Segmentation")
    print("This module provides tools to validate and analyze mask data quality.")
    print("\nExample usage:")
    print("  from data_validation import analyze_mask_directory, print_analysis_summary")
    print("  results = analyze_mask_directory('/path/to/masks')")
    print("  print_analysis_summary(results)")
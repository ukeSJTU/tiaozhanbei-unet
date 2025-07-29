#!/usr/bin/env python3
"""
Testing script for UNet-based anomaly detection on MVTec dataset.
"""

import os
import argparse
import json
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

from model import AnomalyUNet, UNet
from dataset import get_dataloaders, get_available_categories
from utils import (load_checkpoint, calculate_metrics, calculate_pixel_metrics, 
                   visualize_results, plot_confusion_matrix, print_metrics,
                   get_optimal_threshold, compute_anomaly_score)


def parse_args():
    parser = argparse.ArgumentParser(description='Test UNet for MVTec anomaly detection')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='../datasets/mvtec_anomaly_detection',
                        help='Path to MVTec dataset root directory')
    parser.add_argument('--category', type=str, default='bottle',
                        help='Object category to test on')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='anomaly_unet', choices=['unet', 'anomaly_unet'],
                        help='Model architecture')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transposed convolution')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    # Evaluation arguments
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold for anomaly detection (if None, will be optimized)')
    parser.add_argument('--pixel_thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                        help='Thresholds for pixel-level evaluation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='../test_results',
                        help='Directory to save test results')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--max_vis_samples', type=int, default=20,
                        help='Maximum number of samples to visualize')
    
    return parser.parse_args()


def test_model(model, test_loader, device, threshold=None):
    """Test the model and collect predictions."""
    model.eval()
    
    all_images = []
    all_reconstructions = []
    all_anomaly_maps = []
    all_masks_true = []
    all_labels = []
    all_anomaly_types = []
    all_image_paths = []
    all_anomaly_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask']
            labels = batch['label'].numpy()
            anomaly_types = batch['anomaly_type']
            image_paths = batch['image_path']
            
            # Forward pass
            if isinstance(model, AnomalyUNet):
                reconstruction, anomaly_map = model(images)
            else:
                # Standard UNet - use output as anomaly map, create dummy reconstruction
                anomaly_map = torch.sigmoid(model(images))
                reconstruction = images  # Dummy reconstruction
            
            # Calculate anomaly scores
            anomaly_scores = compute_anomaly_score(reconstruction, images).cpu().numpy()
            
            # Store results
            all_images.extend(images.cpu())
            all_reconstructions.extend(reconstruction.cpu())
            all_anomaly_maps.extend(anomaly_map.cpu())
            all_masks_true.extend(masks)
            all_labels.extend(labels)
            all_anomaly_types.extend(anomaly_types)
            all_image_paths.extend(image_paths)
            all_anomaly_scores.extend(anomaly_scores)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)
    all_masks_true = np.array([mask.numpy() for mask in all_masks_true])
    all_anomaly_maps = np.array([amap.numpy() for amap in all_anomaly_maps])
    
    # Determine threshold if not provided
    if threshold is None:
        threshold, _ = get_optimal_threshold(all_labels, all_anomaly_scores)
        print(f"Optimal threshold: {threshold:.4f}")
    
    # Make predictions
    predictions = (all_anomaly_scores > threshold).astype(int)
    
    return {
        'images': all_images,
        'reconstructions': all_reconstructions,
        'anomaly_maps': all_anomaly_maps,
        'masks_true': all_masks_true,
        'labels': all_labels,
        'predictions': predictions,
        'anomaly_scores': all_anomaly_scores,
        'anomaly_types': all_anomaly_types,
        'image_paths': all_image_paths,
        'threshold': threshold
    }


def evaluate_results(results, pixel_thresholds):
    """Evaluate the test results."""
    labels = results['labels']
    predictions = results['predictions']
    anomaly_scores = results['anomaly_scores']
    masks_true = results['masks_true']
    anomaly_maps = results['anomaly_maps']
    anomaly_types = results['anomaly_types']
    
    # Image-level metrics
    image_metrics = calculate_metrics(labels, predictions, anomaly_scores)
    
    # Per-anomaly-type metrics
    type_metrics = defaultdict(dict)
    unique_types = list(set(anomaly_types))
    
    for atype in unique_types:
        type_indices = np.array([i for i, t in enumerate(anomaly_types) if t == atype])
        if len(type_indices) > 0:
            type_labels = labels[type_indices]
            type_predictions = predictions[type_indices]
            type_scores = anomaly_scores[type_indices]
            
            if len(np.unique(type_labels)) > 1:  # Only if both classes present
                type_metrics[atype] = calculate_metrics(type_labels, type_predictions, type_scores)
            else:
                # All same class
                type_metrics[atype] = {
                    'accuracy': 1.0 if type_predictions[0] == type_labels[0] else 0.0,
                    'count': len(type_indices)
                }
    
    # Pixel-level metrics (only for anomalous images)
    pixel_metrics = {}
    anomaly_indices = labels == 1
    
    if np.sum(anomaly_indices) > 0:
        anomaly_masks_true = masks_true[anomaly_indices]
        anomaly_maps_pred = anomaly_maps[anomaly_indices]
        
        for threshold in pixel_thresholds:
            metrics = calculate_pixel_metrics(anomaly_masks_true, anomaly_maps_pred, threshold)
            pixel_metrics[f'threshold_{threshold}'] = metrics
    
    return {
        'image_metrics': image_metrics,
        'type_metrics': dict(type_metrics),
        'pixel_metrics': pixel_metrics
    }


def save_results(results, evaluation, output_dir, args):
    """Save test results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save evaluation metrics
    metrics_file = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        evaluation_json = convert_numpy(evaluation)
        evaluation_json['threshold'] = float(results['threshold'])
        evaluation_json['args'] = vars(args)
        
        json.dump(evaluation_json, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")
    
    # Save confusion matrix
    cm_file = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['labels'], results['predictions'], save_path=cm_file)
    
    # Save detailed results
    detailed_results = {
        'labels': results['labels'].tolist(),
        'predictions': results['predictions'].tolist(),
        'anomaly_scores': results['anomaly_scores'].tolist(),
        'anomaly_types': results['anomaly_types'],
        'image_paths': results['image_paths'],
        'threshold': float(results['threshold'])
    }
    
    detailed_file = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Detailed results saved to: {detailed_file}")


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Testing category: {args.category}")
    
    # Check if category exists
    available_categories = get_available_categories(args.data_root)
    if args.category not in available_categories:
        print(f"Category '{args.category}' not found!")
        print(f"Available categories: {available_categories}")
        return
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.category}_test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loader (only test set)
    print("Creating data loader...")
    _, test_loader = get_dataloaders(
        root_dir=args.data_root,
        category=args.category,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    if args.model == 'anomaly_unet':
        model = AnomalyUNet(n_channels=3, bilinear=args.bilinear)
    else:
        model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear)
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, None, args.checkpoint, device)
    
    # Test model
    print("Testing model...")
    results = test_model(model, test_loader, device, args.threshold)
    
    # Evaluate results
    print("Evaluating results...")
    evaluation = evaluate_results(results, args.pixel_thresholds)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    print_metrics(evaluation['image_metrics'], "Image-level")
    
    if evaluation['pixel_metrics']:
        print("\nPixel-level Metrics:")
        for threshold, metrics in evaluation['pixel_metrics'].items():
            print(f"\n{threshold}:")
            print_metrics(metrics, f"  ")
    
    if evaluation['type_metrics']:
        print("\nPer-anomaly-type Metrics:")
        for atype, metrics in evaluation['type_metrics'].items():
            print(f"\n{atype}:")
            print_metrics(metrics, f"  ")
    
    # Save results
    save_results(results, evaluation, output_dir, args)
    
    # Save visualizations
    if args.save_visualizations:
        print("Saving visualizations...")
        
        # Select samples for visualization
        n_samples = min(args.max_vis_samples, len(results['images']))
        indices = np.random.choice(len(results['images']), n_samples, replace=False)
        
        vis_images = [results['images'][i] for i in indices]
        vis_masks_true = [torch.tensor(results['masks_true'][i]) for i in indices]
        vis_masks_pred = [torch.tensor(results['anomaly_maps'][i]) for i in indices]
        vis_reconstructions = [results['reconstructions'][i] for i in indices] if isinstance(model, AnomalyUNet) else None
        
        vis_path = os.path.join(output_dir, 'visualizations.png')
        visualize_results(
            vis_images, vis_masks_true, vis_masks_pred, 
            vis_reconstructions, vis_path, n_samples
        )
    
    print(f"\nTesting completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

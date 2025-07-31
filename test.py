#!/usr/bin/env python3
"""
Test script for evaluating trained models on Gear dataset.
"""

import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.gear_dataset import get_gear_dataloaders
from src.model import SegmentationUNet, UNet
from src.metrics import SegmentationMetrics, compute_metrics_from_predictions
from src.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Test UNet on Gear dataset')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='datasets/Gear',
                        help='Path to Gear dataset root directory')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                        help='Dataset split to evaluate on')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='seg_unet',
                        choices=['unet', 'seg_unet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for segmentation UNet')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction visualizations')
    parser.add_argument('--save_confusion_matrix', action='store_true',
                        help='Save confusion matrix plot')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with limited samples')
    parser.add_argument('--debug_samples', type=int, default=50,
                        help='Number of samples to use in debug mode')

    return parser.parse_args()


def evaluate_model(model, dataloader, device, num_classes, save_predictions=False, 
                  save_dir=None, class_names=None):
    """
    Evaluate model on dataset
    
    Returns:
        Dictionary containing evaluation metrics and results
    """
    model.eval()
    
    metrics = SegmentationMetrics(num_classes)
    all_predictions = []
    all_targets = []
    all_image_paths = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (images, masks, image_paths) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(outputs, masks)
            
            # Store for later analysis
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
            all_image_paths.extend(image_paths)
            
            # Save some prediction visualizations
            if save_predictions and batch_idx < 5 and save_dir:
                save_batch_predictions(
                    images.cpu(), masks.cpu(), predictions.cpu(), 
                    image_paths, save_dir, batch_idx, class_names
                )
    
    # Compute final metrics
    computed_metrics = metrics.compute_all_metrics()
    
    # Print detailed metrics
    metrics.print_metrics(class_names)
    
    # Plot confusion matrix
    if save_dir:
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        metrics.plot_confusion_matrix(class_names, save_path=cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
    
    return {
        'metrics': computed_metrics,
        'predictions': torch.cat(all_predictions, dim=0),
        'targets': torch.cat(all_targets, dim=0),
        'image_paths': all_image_paths
    }


def save_batch_predictions(images, masks, predictions, image_paths, save_dir, 
                          batch_idx, class_names=None):
    """Save prediction visualizations for a batch"""
    batch_size = images.size(0)
    
    # Create colormap for visualization
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(masks.max().item() + 1)]
    
    # Create a colormap
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i in range(min(batch_size, 4)):  # Save up to 4 images per batch
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        gt_mask = masks[i].numpy()
        axes[1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction mask
        pred_mask = predictions[i].numpy()
        axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        img_name = os.path.basename(image_paths[i]).split('.')[0]
        save_path = os.path.join(save_dir, f'prediction_batch{batch_idx}_img{i}_{img_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def save_results_summary(results, save_path, args):
    """Save evaluation results summary"""
    metrics = results['metrics']
    
    summary = {
        'evaluation_args': vars(args),
        'overall_metrics': {
            'pixel_accuracy': float(metrics['pixel_accuracy']),
            'mean_accuracy': float(metrics['mean_accuracy']),
            'mean_iou': float(metrics['mean_iou']),
            'mean_dice': float(metrics['mean_dice']),
            'mean_precision': float(metrics['mean_precision']),
            'mean_recall': float(metrics['mean_recall']),
            'mean_f1': float(metrics['mean_f1'])
        },
        'per_class_metrics': {
            'iou': metrics['iou_per_class'].tolist(),
            'dice': metrics['dice_per_class'].tolist(),
            'precision': metrics['precision_per_class'].tolist(),
            'recall': metrics['recall_per_class'].tolist(),
            'f1': metrics['f1_per_class'].tolist()
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: {save_path}")


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create data loader
    print("Creating data loader...")
    train_loader, val_loader, test_loader, num_classes = get_gear_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers
    )
    
    # Select the appropriate dataloader
    if args.split == 'test':
        dataloader = test_loader
        print(f"Evaluating on test set ({len(test_loader.dataset)} samples)")
    else:
        dataloader = val_loader
        print(f"Evaluating on validation set ({len(val_loader.dataset)} samples)")
    
    # Debug mode: limit dataset size
    if args.debug:
        print(f"DEBUG MODE: Limiting evaluation to {args.debug_samples} samples")
        import random
        from torch.utils.data import Subset
        
        indices = random.sample(
            range(len(dataloader.dataset)),
            min(args.debug_samples, len(dataloader.dataset))
        )
        subset = Subset(dataloader.dataset, indices)
        dataloader = torch.utils.data.DataLoader(
            subset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers, 
            pin_memory=True
        )
    
    # Get class names from dataset
    dataset = dataloader.dataset
    if hasattr(dataset, 'class_names'):
        class_names = ['background'] + list(dataset.class_names)
    else:
        class_names = ['background', 'pitting', 'spalling', 'scrape'][:num_classes]
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create model
    print("Creating model...")
    if args.model == 'seg_unet':
        model = SegmentationUNet(
            n_channels=3, 
            n_classes=num_classes, 
            bilinear=args.bilinear,
            dropout=args.dropout
        )
    else:
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=args.bilinear)
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    epoch, loss = load_checkpoint(model, None, args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=num_classes,
        save_predictions=args.save_predictions,
        save_dir=args.save_dir if args.save_predictions else None,
        class_names=class_names
    )
    
    # Save results summary
    summary_path = os.path.join(args.save_dir, 'evaluation_results.json')
    save_results_summary(results, summary_path, args)
    
    # Print final summary
    metrics = results['metrics']
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {args.split}")
    print(f"Samples evaluated: {len(results['targets'])}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
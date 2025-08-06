#!/usr/bin/env python3
"""
Visualization script for displaying model predictions on Gear dataset.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from src.gear_dataset import get_gear_dataloaders
from src.model import SegmentationUNet, UNet
from src.utils import load_checkpoint, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize UNet predictions on Gear dataset')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='datasets/Gear',
                        help='Path to Gear dataset root directory')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'],
                        help='Dataset split to visualize')
    
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
    
    # Visualization settings
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    
    # Output settings
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save individual prediction images')
    parser.add_argument('--save_grid', action='store_true',
                        help='Save grid of multiple predictions')
    parser.add_argument('--show_confidence', action='store_true',
                        help='Show prediction confidence maps')
    
    # Display settings
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 5],
                        help='Figure size for individual visualizations')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[2, 5],
                        help='Grid size for multi-image visualization')
    parser.add_argument('--always_save', action='store_true', default=True,
                        help='Always save visualizations (default: True)')

    return parser.parse_args()


def create_colormap(num_classes):
    """Create a colormap for visualization"""
    if num_classes <= 10:
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i) for i in range(num_classes)]
    else:
        cmap = plt.cm.get_cmap('tab20')
        colors = [cmap(i) for i in range(num_classes)]
    return colors


def denormalize_image(image_tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    if image_tensor.dim() == 4:  # Batch dimension
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:  # Single image
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    
    image = image_tensor * std + mean
    return torch.clamp(image, 0, 1)


def create_overlay_mask(mask, num_classes, alpha=0.4):
    """Create a transparent colored mask overlay"""
    # Create colormap
    colors = create_colormap(num_classes)
    
    # Convert mask to RGB
    mask_rgb = np.zeros((*mask.shape, 4))  # RGBA
    
    for class_id in range(num_classes):
        class_mask = mask == class_id
        if class_id == 0:  # Background - no overlay
            continue
        mask_rgb[class_mask] = colors[class_id]
        mask_rgb[class_mask, 3] = alpha  # Set alpha channel
    
    return mask_rgb


def visualize_single_prediction(image, gt_mask, pred_mask, pred_logits, 
                               class_names, save_path=None, show_confidence=False):
    """Visualize a single prediction with transparent overlays"""
    num_plots = 3 if show_confidence else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    # Denormalize image
    img_denorm = denormalize_image(image)
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    
    # Convert to numpy
    gt_np = gt_mask.cpu().numpy()
    pred_np = pred_mask.cpu().numpy()
    
    # Create overlay masks
    gt_overlay = create_overlay_mask(gt_np, len(class_names), alpha=0.4)
    pred_overlay = create_overlay_mask(pred_np, len(class_names), alpha=0.4)
    
    # Ground truth overlay
    axes[0].imshow(img_np)
    axes[0].imshow(gt_overlay)
    axes[0].set_title('Original + Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    # Prediction overlay
    axes[1].imshow(img_np)
    axes[1].imshow(pred_overlay)
    axes[1].set_title('Original + Prediction', fontsize=14)
    axes[1].axis('off')
    
    # Confidence map
    if show_confidence:
        # Use max probability across classes as confidence
        confidence = torch.softmax(pred_logits, dim=0).max(dim=0)[0]
        im3 = axes[2].imshow(confidence.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Confidence', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Create a legend for class colors
    colors = create_colormap(len(class_names))
    legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) 
                      for i in range(1, len(class_names))]  # Skip background
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    return fig


def visualize_prediction_grid(images, gt_masks, pred_masks, pred_logits, 
                             class_names, grid_size, save_path=None):
    """Visualize multiple predictions in individual files with transparent overlays"""
    rows, cols = grid_size
    
    # Generate individual visualizations for each sample
    for i in range(min(rows * cols, len(images))):
        # Create individual figure for each sample  
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Denormalize image
        img_denorm = denormalize_image(images[i])
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        
        # Convert to numpy
        gt_np = gt_masks[i].cpu().numpy()
        pred_np = pred_masks[i].cpu().numpy()
        
        # Create overlay masks
        gt_overlay = create_overlay_mask(gt_np, len(class_names), alpha=0.4)
        pred_overlay = create_overlay_mask(pred_np, len(class_names), alpha=0.4)
        
        # Ground truth overlay
        axes[0].imshow(img_np)
        axes[0].imshow(gt_overlay)
        axes[0].set_title(f'Sample {i+1}: Original + Ground Truth', fontsize=14)
        axes[0].axis('off')
        
        # Prediction overlay
        axes[1].imshow(img_np)
        axes[1].imshow(pred_overlay)
        axes[1].set_title(f'Sample {i+1}: Original + Prediction', fontsize=14)
        axes[1].axis('off')
        
        # Create a legend for class colors
        colors = create_colormap(len(class_names))
        legend_elements = [Patch(facecolor=colors[j], label=class_names[j]) 
                          for j in range(1, len(class_names))]  # Skip background
        if legend_elements:
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        # Save individual sample if requested
        if save_path:
            sample_path = save_path.replace('.png', f'_sample_{i+1}.png')
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            print(f"Saved sample {i+1} visualization to: {sample_path}")
        
        plt.close(fig)
    
    # Create a single composite figure
    if save_path:
        print(f"Saved {min(rows * cols, len(images))} individual sample visualizations")
    
    return None  # Return None since we're saving individual files


def compute_prediction_stats(pred_logits, gt_mask, class_names):
    """Compute prediction statistics"""
    pred_probs = torch.softmax(pred_logits, dim=0)
    pred_mask = torch.argmax(pred_logits, dim=0)
    
    stats = {
        'accuracy': (pred_mask == gt_mask).float().mean().item(),
        'confidence_mean': pred_probs.max(dim=0)[0].mean().item(),
        'confidence_std': pred_probs.max(dim=0)[0].std().item(),
    }
    
    # Per-class statistics
    for i, class_name in enumerate(class_names):
        class_mask = gt_mask == i
        if class_mask.sum() > 0:
            class_acc = (pred_mask[class_mask] == i).float().mean().item()
            stats[f'accuracy_{class_name}'] = class_acc
    
    return stats


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.save_dir, "visualization")
    
    logger.info(f"Using device: {device}")
    
    # Create data loader
    logger.info("Creating data loader...")
    train_loader, val_loader, test_loader, num_classes = get_gear_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers
    )
    
    # Select the appropriate dataloader
    if args.split == 'test':
        dataloader = test_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = train_loader
    
    logger.info(f"Visualizing {args.split} set")
    
    # Get class names from dataset
    dataset = dataloader.dataset
    try:
        class_names = ['background'] + list(dataset.class_names)
    except AttributeError:
        class_names = ['background', 'pitting', 'spalling', 'scrape'][:num_classes]
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Create model
    logger.info("Creating model...")
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
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    epoch, loss = load_checkpoint(model, None, args.checkpoint, device)
    logger.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect samples for visualization
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_pred_logits = []
    all_image_paths = []
    
    logger.info("Generating predictions...")
    with torch.no_grad():
        samples_collected = 0
        for images, masks, image_paths in dataloader:
            if samples_collected >= args.num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            batch_size = images.size(0)
            for i in range(min(batch_size, args.num_samples - samples_collected)):
                all_images.append(images[i].cpu())
                all_gt_masks.append(masks[i].cpu())
                all_pred_masks.append(predictions[i].cpu())
                all_pred_logits.append(outputs[i].cpu())
                all_image_paths.append(image_paths[i])
                samples_collected += 1
    
    logger.info(f"Collected {len(all_images)} samples for visualization")
    
    # Generate individual visualizations (always save by default)
    if args.save_individual or args.always_save:
        logger.info("Generating individual visualizations...")
        for i, (img, gt, pred, logits, path) in enumerate(zip(
            all_images, all_gt_masks, all_pred_masks, all_pred_logits, all_image_paths
        )):
            # Compute stats
            stats = compute_prediction_stats(logits, gt, class_names)
            
            # Create filename
            img_name = os.path.basename(path).split('.')[0]
            save_path = os.path.join(args.save_dir, f'prediction_{i:03d}_{img_name}.png')
            
            # Visualize
            fig = visualize_single_prediction(
                img, gt, pred, logits, class_names, 
                save_path=save_path, show_confidence=args.show_confidence
            )
            plt.close(fig)
            
            # Print stats
            logger.info(f"Sample {i+1}: Accuracy={stats['accuracy']:.3f}, "
                       f"Confidence={stats['confidence_mean']:.3f}±{stats['confidence_std']:.3f}")
    
    # Generate grid visualization (always save by default)
    if args.save_grid or args.always_save:
        logger.info("Generating grid visualization...")
        grid_save_path = os.path.join(args.save_dir, 'predictions_grid.png')
        fig = visualize_prediction_grid(
            all_images, all_gt_masks, all_pred_masks, all_pred_logits,
            class_names, args.grid_size, save_path=grid_save_path
        )
        plt.close(fig)
    
    # Generate class distribution plot
    logger.info("Generating class distribution plot...")
    gt_classes = torch.cat(all_gt_masks).flatten()
    pred_classes = torch.cat(all_pred_masks).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth distribution
    gt_counts = torch.bincount(gt_classes, minlength=num_classes)
    axes[0].bar(range(num_classes), gt_counts.numpy())
    axes[0].set_title('Ground Truth Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Pixel Count')
    axes[0].set_xticks(range(num_classes))
    axes[0].set_xticklabels(class_names, rotation=45)
    
    # Prediction distribution
    pred_counts = torch.bincount(pred_classes, minlength=num_classes)
    axes[1].bar(range(num_classes), pred_counts.numpy())
    axes[1].set_title('Prediction Class Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Pixel Count')
    axes[1].set_xticks(range(num_classes))
    axes[1].set_xticklabels(class_names, rotation=45)
    
    plt.tight_layout()
    dist_save_path = os.path.join(args.save_dir, 'class_distribution.png')
    plt.savefig(dist_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Class distribution plot saved to: {dist_save_path}")
    logger.info(f"All visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Visualization script for displaying model predictions on Gear dataset.
"""

import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image

from src.gear_dataset import get_gear_dataloaders
from src.model import SegmentationUNet, UNet
from src.utils import load_checkpoint


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

    return parser.parse_args()


def create_colormap(num_classes):
    """Create a colormap for visualization"""
    if num_classes <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
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


def visualize_single_prediction(image, gt_mask, pred_mask, pred_logits, 
                               class_names, save_path=None, show_confidence=False):
    """Visualize a single prediction"""
    num_plots = 4 if show_confidence else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    # Denormalize image
    img_denorm = denormalize_image(image)
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth mask
    gt_np = gt_mask.cpu().numpy()
    im1 = axes[1].imshow(gt_np, cmap='tab10', vmin=0, vmax=len(class_names)-1)
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # Prediction mask
    pred_np = pred_mask.cpu().numpy()
    im2 = axes[2].imshow(pred_np, cmap='tab10', vmin=0, vmax=len(class_names)-1)
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    # Confidence map
    if show_confidence:
        # Use max probability across classes as confidence
        confidence = torch.softmax(pred_logits, dim=0).max(dim=0)[0]
        im3 = axes[3].imshow(confidence.cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title('Confidence', fontsize=14)
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Add colorbar for masks
    cbar = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(class_names)))
    cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    return fig


def visualize_prediction_grid(images, gt_masks, pred_masks, pred_logits, 
                             class_names, grid_size, save_path=None):
    """Visualize multiple predictions in a grid"""
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 15, rows * 5))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(rows * cols, len(images))):
        row = i // cols
        col = i % cols
        
        # Denormalize image
        img_denorm = denormalize_image(images[i])
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        
        # Original image
        axes[row, col*3].imshow(img_np)
        axes[row, col*3].set_title(f'Sample {i+1}: Original', fontsize=12)
        axes[row, col*3].axis('off')
        
        # Ground truth
        gt_np = gt_masks[i].cpu().numpy()
        axes[row, col*3+1].imshow(gt_np, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[row, col*3+1].set_title(f'Sample {i+1}: Ground Truth', fontsize=12)
        axes[row, col*3+1].axis('off')
        
        # Prediction
        pred_np = pred_masks[i].cpu().numpy()
        im = axes[row, col*3+2].imshow(pred_np, cmap='tab10', vmin=0, vmax=len(class_names)-1)
        axes[row, col*3+2].set_title(f'Sample {i+1}: Prediction', fontsize=12)
        axes[row, col*3+2].axis('off')
    
    # Hide unused subplots
    for i in range(rows * cols, rows * cols):
        for j in range(3):
            if i < len(axes) and j*3+2 < len(axes[i//cols]):
                axes[i//cols, (i%cols)*3+j].axis('off')
    
    # Add a single colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(class_names)))
    cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid visualization to: {save_path}")
    
    return fig


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
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = train_loader
    
    print(f"Visualizing {args.split} set ({len(dataloader.dataset)} samples)")
    
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
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect samples for visualization
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_pred_logits = []
    all_image_paths = []
    
    print("Generating predictions...")
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
    
    print(f"Collected {len(all_images)} samples for visualization")
    
    # Generate individual visualizations
    if args.save_individual:
        print("Generating individual visualizations...")
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
            print(f"Sample {i+1}: Accuracy={stats['accuracy']:.3f}, "
                  f"Confidence={stats['confidence_mean']:.3f}±{stats['confidence_std']:.3f}")
    
    # Generate grid visualization
    if args.save_grid:
        print("Generating grid visualization...")
        grid_save_path = os.path.join(args.save_dir, 'predictions_grid.png')
        fig = visualize_prediction_grid(
            all_images, all_gt_masks, all_pred_masks, all_pred_logits,
            class_names, args.grid_size, save_path=grid_save_path
        )
        plt.close(fig)
    
    # Generate class distribution plot
    print("Generating class distribution plot...")
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
    
    print(f"Class distribution plot saved to: {dist_save_path}")
    print(f"All visualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
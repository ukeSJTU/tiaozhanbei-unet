#!/usr/bin/env python3
"""
Training script for UNet-based anomaly detection on MVTec dataset.
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

from dataset import get_available_categories, get_dataloaders
from model import AnomalyUNet
from train_utils import (
    CombinedLoss,
    SSIMLoss,
    get_optimizer,
    get_scheduler,
    train_epoch,
    validate_epoch,
)
from utils import (
    create_output_dirs,
    load_checkpoint,
    plot_training_curves,
    print_metrics,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet for MVTec anomaly detection')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='../datasets/mvtec_anomaly_detection',
                        help='Path to MVTec dataset root directory')
    parser.add_argument('--category', type=str, default='bottle',
                        help='Object category to train on')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='anomaly_unet', choices=['unet', 'anomaly_unet'],
                        help='Model architecture')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transposed convolution')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Loss arguments
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight for segmentation loss')
    parser.add_argument('--use_ssim', action='store_true',
                        help='Use SSIM loss for reconstruction')
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='../outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Validation
    parser.add_argument('--val_freq', type=int, default=5,
                        help='Validate every N epochs')

    # Debug arguments
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with limited data')
    parser.add_argument('--debug_samples', type=int, default=20,
                        help='Number of samples to use in debug mode')

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Training category: {args.category}")
    
    # Check if category exists
    available_categories = get_available_categories(args.data_root)
    if args.category not in available_categories:
        print(f"Category '{args.category}' not found!")
        print(f"Available categories: {available_categories}")
        return
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.category}_{args.model}_{timestamp}"
    experiment_dir = os.path.join(args.save_dir, experiment_name)
    output_dirs = create_output_dirs(experiment_dir)
    
    print(f"Experiment directory: {experiment_dir}")
    
    # Save arguments
    with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_root,
        category=args.category,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    # Debug mode: limit dataset size
    if args.debug:
        print(f"DEBUG MODE: Limiting dataset to {args.debug_samples} samples")
        import random

        from torch.utils.data import Subset

        # Limit training data
        train_indices = random.sample(range(len(train_loader.dataset)),
                                    min(args.debug_samples, len(train_loader.dataset)))
        train_subset = Subset(train_loader.dataset, train_indices)
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )

        # Limit validation data
        val_indices = random.sample(range(len(val_loader.dataset)),
                                  min(args.debug_samples, len(val_loader.dataset)))
        val_subset = Subset(val_loader.dataset, val_indices)
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    if args.model == 'anomaly_unet':
        model = AnomalyUNet(n_channels=3, bilinear=args.bilinear)
    else:
        from model import UNet
        model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    if args.use_ssim:
        recon_criterion = SSIMLoss()
    else:
        recon_criterion = nn.MSELoss()
    
    criterion = CombinedLoss(
        recon_weight=args.recon_weight,
        seg_weight=args.seg_weight
    ).to(device)
    
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        num_epochs=args.epochs
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_metrics['total_loss'])
        
        # Update scheduler
        if scheduler and args.scheduler != 'plateau':
            scheduler.step()
        
        # Validate
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_metrics = validate_epoch(model, val_loader, criterion, device)
            val_losses.append(val_metrics['total_loss'])
            
            # Update scheduler for plateau
            if scheduler and args.scheduler == 'plateau':
                scheduler.step(val_metrics['total_loss'])
            
            # Print metrics
            print(f"\nEpoch {epoch}/{args.epochs-1}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Recon: {train_metrics['recon_loss']:.4f}, "
                  f"Seg: {train_metrics['seg_loss']:.4f})")
            print(f"Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(Recon: {val_metrics['recon_loss']:.4f}, "
                  f"Seg: {val_metrics['seg_loss']:.4f})")
            
            print_metrics(val_metrics['image_metrics'], "Image-level")
            if val_metrics['pixel_metrics']:
                print_metrics(val_metrics['pixel_metrics'], "Pixel-level")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_checkpoint_path = os.path.join(output_dirs['checkpoints'], 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_metrics['total_loss'], best_checkpoint_path)
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dirs['checkpoints'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, train_metrics['total_loss'], checkpoint_path)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.2f}s")
    
    # Save training curves
    curves_path = os.path.join(output_dirs['results'], 'training_curves.png')
    plot_training_curves(train_losses, val_losses, curves_path)
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_epochs': args.epochs,
        'total_params': total_params,
        'args': vars(args)
    }
    
    results_path = os.path.join(output_dirs['results'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()

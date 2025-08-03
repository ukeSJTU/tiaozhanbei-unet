#!/usr/bin/env python3
"""
Training script for multi-class semantic segmentation on Gear dataset.
"""

import argparse
import json
import os
import time
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import numpy as np
from tqdm import tqdm

from src.gear_dataset import get_gear_dataloaders
from src.model import SegmentationUNet, UNet
from src.metrics import SegmentationMetrics, CombinedSegmentationLoss
from src.utils import create_output_dirs, save_checkpoint, load_checkpoint, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet for Gear multi-class segmentation')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='datasets/Gear',
                        help='Path to Gear dataset root directory')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size (will be used for both height and width)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='seg_unet', 
                        choices=['unet', 'seg_unet'],
                        help='Model architecture')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transposed convolution')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for segmentation UNet')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', 
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    
    # Loss arguments
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='Weight for cross entropy loss')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Weight for dice loss')
    parser.add_argument('--focal_weight', type=float, default=0.0,
                        help='Weight for focal loss')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='Class weights as comma-separated values (e.g., "1.0,2.0,1.5")')
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='outputs',
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


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, optimizer_name, lr, weight_decay):
    """Create optimizer"""
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_classes):
    """Train for one epoch"""
    model.train()
    
    metrics = SegmentationMetrics(num_classes)
    total_loss = 0
    num_batches = len(dataloader)
    
    # Create progress bar for batches
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch:3d} [Train]', 
                       leave=False, unit='batch')
    
    for batch_idx, (images, masks, _) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        metrics.update(outputs, masks)
        
        # Update progress bar
        avg_loss_so_far = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss_so_far:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    computed_metrics = metrics.compute_all_metrics()
    
    return {
        'loss': avg_loss,
        'metrics': computed_metrics
    }


def validate_epoch(model, dataloader, criterion, device, num_classes):
    """Validate for one epoch"""
    model.eval()
    
    metrics = SegmentationMetrics(num_classes)
    total_loss = 0
    num_batches = len(dataloader)
    
    # Create progress bar for validation
    progress_bar = tqdm(dataloader, desc='Validating', 
                       leave=False, unit='batch')
    
    with torch.no_grad():
        for batch_idx, (images, masks, _) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs, masks)
            
            # Update progress bar
            avg_loss_so_far = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss_so_far:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    computed_metrics = metrics.compute_all_metrics()
    
    return {
        'loss': avg_loss,
        'metrics': computed_metrics
    }


def print_epoch_results(epoch, train_results, val_results=None, epoch_time=None, logger=None):
    """Print training and validation results"""
    log_func = logger.info if logger else print
    
    log_func(f"{'='*60}")
    log_func(f"EPOCH {epoch:3d} RESULTS")
    log_func(f"{'='*60}")
    
    # Training results
    log_func(f"🚀 TRAINING:")
    log_func(f"   Loss:     {train_results['loss']:.4f}")
    log_func(f"   mIoU:     {train_results['metrics']['mean_iou']:.4f}")
    log_func(f"   mDice:    {train_results['metrics']['mean_dice']:.4f}")
    log_func(f"   Accuracy: {train_results['metrics']['pixel_accuracy']:.4f}")
    
    # Validation results
    if val_results:
        log_func(f"📊 VALIDATION:")
        log_func(f"   Loss:     {val_results['loss']:.4f}")
        log_func(f"   mIoU:     {val_results['metrics']['mean_iou']:.4f}")
        log_func(f"   mDice:    {val_results['metrics']['mean_dice']:.4f}")
        log_func(f"   Accuracy: {val_results['metrics']['pixel_accuracy']:.4f}")
    
    # Timing info
    if epoch_time:
        log_func(f"⏱️  Epoch Time: {epoch_time:.2f}s")
    
    log_func(f"{'='*60}")


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gear_seg_{args.model}_{timestamp}"
    experiment_dir = os.path.join(args.save_dir, experiment_name)
    output_dirs = create_output_dirs(experiment_dir)
    
    # Setup logging
    logger = setup_logging(output_dirs['logs'], experiment_name)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Parse class weights
    class_weights = None
    if args.class_weights:
        class_weights = [float(w) for w in args.class_weights.split(',')]
        logger.info(f"Using class weights: {class_weights}")
    
    # Save arguments
    with open(os.path.join(experiment_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, num_classes = get_gear_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers
    )

    # Debug mode: limit dataset size
    if args.debug:
        logger.info(f"DEBUG MODE: Limiting dataset to {args.debug_samples} samples")
        
        # Limit training data
        train_indices = random.sample(
            range(len(train_loader.dataset)),
            min(args.debug_samples, len(train_loader.dataset))
        )
        train_subset = Subset(train_loader.dataset, train_indices)
        train_loader = torch.utils.data.DataLoader(
            train_subset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers, 
            pin_memory=True
        )

        # Limit validation data
        val_indices = random.sample(
            range(len(val_loader.dataset)),
            min(args.debug_samples, len(val_loader.dataset))
        )
        val_subset = Subset(val_loader.dataset, val_indices)
        val_loader = torch.utils.data.DataLoader(
            val_subset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers, 
            pin_memory=True
        )

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = CombinedSegmentationLoss(
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        class_weights=class_weights
    ).to(device)
    
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("🚀 Starting training...")
    logger.info(f"📊 Total epochs: {args.epochs}")
    logger.info(f"💾 Checkpoints will be saved every {args.save_freq} epochs")
    logger.info(f"🔍 Validation will run every {args.val_freq} epochs")
    
    train_losses = []
    val_losses = []
    best_val_miou = 0.0
    
    # Create main progress bar for epochs
    epoch_progress = tqdm(range(start_epoch, args.epochs), desc='Training Progress', 
                         unit='epoch', position=0)
    
    for epoch in epoch_progress:
        epoch_start_time = time.time()
        
        # Train
        train_results = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_classes)
        train_losses.append(train_results['loss'])
        
        # Validate
        val_results = None
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            val_results = validate_epoch(model, val_loader, criterion, device, num_classes)
            val_losses.append(val_results['loss'])
            
            # Save best model based on mIoU
            val_miou = val_results['metrics']['mean_iou']
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_checkpoint_path = os.path.join(output_dirs['checkpoints'], 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_results['loss'], best_checkpoint_path)
                logger.info(f"🏆 New best model saved with mIoU: {best_val_miou:.4f}")
                tqdm.write(f"🏆 New best model saved with mIoU: {best_val_miou:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        
        # Update main progress bar
        postfix = {
            'Train Loss': f'{train_results["loss"]:.4f}',
            'Train mIoU': f'{train_results["metrics"]["mean_iou"]:.4f}'
        }
        if val_results:
            postfix.update({
                'Val Loss': f'{val_results["loss"]:.4f}',
                'Val mIoU': f'{val_results["metrics"]["mean_iou"]:.4f}'
            })
        postfix['Time'] = f'{epoch_time:.1f}s'
        epoch_progress.set_postfix(postfix)
        
        # Print detailed results
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            tqdm.write("")
            print_epoch_results(epoch, train_results, val_results, epoch_time, logger)
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(output_dirs['checkpoints'], f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, train_results['loss'], checkpoint_path)
    
    # Close progress bar
    epoch_progress.close()
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_miou': best_val_miou,
        'total_epochs': args.epochs,
        'total_params': total_params,
        'num_classes': num_classes,
        'args': vars(args)
    }
    
    results_path = os.path.join(output_dirs['results'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"🎉 Training completed!")
    logger.info(f"🏆 Best validation mIoU: {best_val_miou:.4f}")
    logger.info(f"📁 Results saved to: {experiment_dir}")
    logger.info(f"📊 Training Summary:")
    logger.info(f"   Total epochs trained: {args.epochs}")
    logger.info(f"   Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"   Best validation mIoU: {best_val_miou:.4f}")
    logger.info(f"   Model parameters: {total_params:,}")


if __name__ == "__main__":
    main()
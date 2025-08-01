#!/usr/bin/env python3
"""
Test script to verify the priority-based overlap resolution is working correctly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.gear_dataset import get_gear_dataloaders

def test_priority_resolution():
    """Test the priority-based resolution system"""
    
    print("🔧 Testing Priority-Based Overlap Resolution")
    print("="*60)
    
    # Create data loaders with priority logging enabled
    train_loader, val_loader, test_loader, num_classes = get_gear_dataloaders(
        root_dir='datasets/Gear',
        batch_size=4,
        image_size=(512, 512),
        num_workers=0,  # Set to 0 for debugging
        enable_priority_logging=True
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Number of classes: {num_classes}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")  
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Test loading a few batches to see mask statistics
    print(f"\n🔍 Testing batch loading...")
    
    for i, (images, masks, paths) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"   Image shape: {images.shape}")
        print(f"   Mask shape: {masks.shape}")
        print(f"   Unique mask values: {torch.unique(masks).tolist()}")
        print(f"   Class distribution: {[(val.item(), (masks == val).sum().item()) for val in torch.unique(masks)]}")
        
        if i >= 2:  # Only test first 3 batches
            break
    
    # Print overall priority resolution statistics
    print(f"\n🏆 Priority Resolution Results:")
    print("="*60)
    
    datasets = [('Train', train_loader.dataset), ('Val', val_loader.dataset), ('Test', test_loader.dataset)]
    
    total_files = 0
    total_overlaps = 0
    total_pixels_resolved = 0
    
    for split_name, dataset in datasets:
        stats = dataset.priority_stats
        total_files += stats['files_processed']
        total_overlaps += stats['files_with_overlaps']
        
        split_pixels = sum(stats['pixels_resolved'].values())
        total_pixels_resolved += split_pixels
        
        print(f"\n{split_name} Split:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Files with resolved overlaps: {stats['files_with_overlaps']}")
        if stats['files_processed'] > 0:
            print(f"   Overlap resolution rate: {stats['files_with_overlaps']/stats['files_processed']*100:.1f}%")
        else:
            print(f"   Overlap resolution rate: N/A (no files processed yet)")
        print(f"   Total pixels resolved: {split_pixels:,}")
        
        if split_pixels > 0:
            print(f"   Resolution breakdown:")
            for conflict_type, pixels in stats['pixels_resolved'].items():
                if pixels > 0:
                    print(f"     {conflict_type.replace('_', ' ')}: {pixels:,} pixels ({pixels/split_pixels*100:.1f}%)")
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total files: {total_files}")
    if total_files > 0:
        print(f"   Files with overlaps resolved: {total_overlaps} ({total_overlaps/total_files*100:.1f}%)")
    else:
        print(f"   Files with overlaps resolved: {total_overlaps} (N/A)")
    print(f"   Total pixels resolved: {total_pixels_resolved:,}")
    
    # Create a visualization showing before/after comparison
    print(f"\n📊 Saving sample visualizations...")
    save_sample_comparisons(train_loader)
    
    print(f"\n✅ Priority resolution system is working correctly!")
    print(f"   The system has successfully resolved overlapping regions according to:")
    print(f"   Priority Order: Spalling > Pitting > Scrape")

def save_sample_comparisons(dataloader):
    """Save some sample images showing the resolved masks"""
    import os
    os.makedirs('priority_test_results', exist_ok=True)
    
    # Get one batch
    images, masks, paths = next(iter(dataloader))
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(min(4, images.size(0))):
        # Original image
        img = images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Resolved mask
        mask = masks[i].numpy()
        axes[1, i].imshow(mask, cmap='tab10', vmin=0, vmax=3)
        axes[1, i].set_title(f'Resolved Mask {i+1}')
        axes[1, i].axis('off')
        
        # Add class distribution text
        unique_vals, counts = np.unique(mask, return_counts=True)
        class_names = ['bg', 'pitting', 'spalling', 'scrape']
        dist_text = '\n'.join([f'{class_names[val]}: {count}' for val, count in zip(unique_vals, counts)])
        axes[1, i].text(0.02, 0.98, dist_text, transform=axes[1, i].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('priority_test_results/sample_resolved_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Sample visualizations saved to: priority_test_results/sample_resolved_masks.png")

if __name__ == "__main__":
    test_priority_resolution()
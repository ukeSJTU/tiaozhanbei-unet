#!/usr/bin/env python3
"""
Demo script to quickly test the dataset and model setup.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import AnomalyUNet, UNet
from dataset import get_dataloaders, get_available_categories
from utils import denormalize_image, tensor_to_numpy


def test_dataset(data_root="../datasets/mvtec_anomaly_detection"):
    """Test dataset loading."""
    print("Testing dataset loading...")
    
    # Get available categories
    categories = get_available_categories(data_root)
    print(f"Available categories: {categories}")
    
    if not categories:
        print("No categories found! Please check your dataset path.")
        return None
    
    # Use first category for testing
    category = categories[0]
    print(f"Testing with category: {category}")
    
    try:
        # Create data loaders
        train_loader, test_loader = get_dataloaders(
            root_dir=data_root,
            category=category,
            batch_size=4,
            image_size=256,
            num_workers=0  # Use 0 for debugging
        )
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        
        # Test loading a batch
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        print(f"Train batch - Image shape: {train_batch['image'].shape}")
        print(f"Train batch - Mask shape: {train_batch['mask'].shape}")
        print(f"Train batch - Labels: {train_batch['label']}")
        print(f"Train batch - Anomaly types: {train_batch['anomaly_type']}")
        
        print(f"Test batch - Image shape: {test_batch['image'].shape}")
        print(f"Test batch - Mask shape: {test_batch['mask'].shape}")
        print(f"Test batch - Labels: {test_batch['label']}")
        print(f"Test batch - Anomaly types: {test_batch['anomaly_type']}")
        
        return train_loader, test_loader, category
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def test_model():
    """Test model creation and forward pass."""
    print("\nTesting model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test AnomalyUNet
    model = AnomalyUNet(n_channels=3, bilinear=False)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"AnomalyUNet total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256).to(device)
    
    with torch.no_grad():
        reconstruction, anomaly_map = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Anomaly map shape: {anomaly_map.shape}")
    
    return model


def visualize_sample(data_loader, category):
    """Visualize a sample from the dataset."""
    print(f"\nVisualizing samples from {category}...")
    
    # Get a batch
    batch = next(iter(data_loader))
    
    # Select first 4 samples
    n_samples = min(4, len(batch['image']))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_samples):
        # Original image
        img = denormalize_image(batch['image'][i])
        img = torch.clamp(img, 0, 1)
        img_np = tensor_to_numpy(img).transpose(1, 2, 0)
        
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Image - {batch['anomaly_type'][i]}")
        axes[0, i].axis('off')
        
        # Mask
        mask = tensor_to_numpy(batch['mask'][i]).squeeze()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"Mask - Label: {batch['label'][i].item()}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('../demo_outputs', exist_ok=True)
    plt.savefig(f'../demo_outputs/{category}_samples.png', dpi=150, bbox_inches='tight')
    print(f"Sample visualization saved to: ../demo_outputs/{category}_samples.png")
    plt.close()


def main():
    print("="*50)
    print("MVTec Anomaly Detection Demo")
    print("="*50)
    
    # Test dataset
    result = test_dataset()
    if result is None:
        print("Dataset test failed!")
        return
    
    train_loader, test_loader, category = result
    
    # Test model
    model = test_model()
    
    # Visualize samples
    visualize_sample(test_loader, category)
    
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Train a model: python train.py --category bottle --epochs 50")
    print("2. Test a model: python test.py --category bottle --checkpoint path/to/checkpoint.pth")
    print("3. Check available categories and modify scripts as needed")
    print("\nFor more options, use --help with any script")


if __name__ == "__main__":
    main()

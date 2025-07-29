#!/usr/bin/env python3
"""
Interactive visualization script for MVTec anomaly detection results.
Allows loading different model checkpoints and visualizing predictions.
"""

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch

from model import AnomalyUNet, UNet
from dataset import get_dataloaders, get_available_categories
from utils import load_checkpoint, denormalize_image, tensor_to_numpy


class AnomalyVisualizer:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.current_idx = 0
        
        # Load all test data
        self.load_test_data()
        
        # Setup matplotlib
        self.setup_plot()
    
    def load_test_data(self):
        """Load all test data into memory for easy navigation."""
        print("Loading test data...")
        self.test_data = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask']
                labels = batch['label']
                anomaly_types = batch['anomaly_type']
                image_paths = batch['image_path']
                
                # Forward pass
                if isinstance(self.model, AnomalyUNet):
                    reconstruction, anomaly_map = self.model(images)
                else:
                    # Standard UNet
                    anomaly_map = torch.sigmoid(self.model(images))
                    reconstruction = images  # No reconstruction for standard UNet
                
                # Store results
                for i in range(len(images)):
                    self.test_data.append({
                        'image': images[i].cpu(),
                        'reconstruction': reconstruction[i].cpu() if isinstance(self.model, AnomalyUNet) else None,
                        'anomaly_map': anomaly_map[i].cpu(),
                        'mask_true': masks[i],
                        'label': labels[i].item(),
                        'anomaly_type': anomaly_types[i],
                        'image_path': image_paths[i]
                    })
        
        print(f"Loaded {len(self.test_data)} test samples")
    
    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        # Determine number of subplots
        has_reconstruction = isinstance(self.model, AnomalyUNet)
        n_cols = 5 if has_reconstruction else 4
        
        self.fig, self.axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
        self.fig.suptitle('MVTec Anomaly Detection Visualization', fontsize=16)
        
        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.25, 0.02, 0.1, 0.05])
        ax_info = plt.axes([0.4, 0.02, 0.2, 0.05])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_info = Button(ax_info, 'Show Info')
        
        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)
        self.btn_info.on_clicked(self.show_info)
        
        # Initial display
        self.update_display()
    
    def prev_sample(self, event):
        """Go to previous sample."""
        self.current_idx = (self.current_idx - 1) % len(self.test_data)
        self.update_display()
    
    def next_sample(self, event):
        """Go to next sample."""
        self.current_idx = (self.current_idx + 1) % len(self.test_data)
        self.update_display()
    
    def show_info(self, event):
        """Show detailed information about current sample."""
        data = self.test_data[self.current_idx]
        info = f"""
Sample {self.current_idx + 1}/{len(self.test_data)}
Label: {'Anomaly' if data['label'] == 1 else 'Normal'}
Anomaly Type: {data['anomaly_type']}
Image Path: {os.path.basename(data['image_path'])}
Anomaly Score: {self.compute_anomaly_score(data):.4f}
        """
        print(info)
    
    def compute_anomaly_score(self, data):
        """Compute anomaly score for current sample."""
        if data['reconstruction'] is not None:
            # Use reconstruction error
            mse = torch.nn.functional.mse_loss(data['reconstruction'], data['image'])
            return mse.item()
        else:
            # Use max anomaly map value
            return data['anomaly_map'].max().item()
    
    def update_display(self):
        """Update the display with current sample."""
        data = self.test_data[self.current_idx]
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
            ax.axis('off')
        
        col_idx = 0
        
        # Original image
        img = denormalize_image(data['image'])
        img = torch.clamp(img, 0, 1)
        img_np = tensor_to_numpy(img).transpose(1, 2, 0)
        self.axes[col_idx].imshow(img_np)
        self.axes[col_idx].set_title('Original Image')
        col_idx += 1
        
        # Reconstruction (if available)
        if data['reconstruction'] is not None:
            recon = torch.clamp(data['reconstruction'], 0, 1)
            recon_np = tensor_to_numpy(recon).transpose(1, 2, 0)
            self.axes[col_idx].imshow(recon_np)
            self.axes[col_idx].set_title('Reconstruction')
            col_idx += 1
        
        # Predicted anomaly map
        anomaly_map = tensor_to_numpy(data['anomaly_map']).squeeze()
        im1 = self.axes[col_idx].imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
        self.axes[col_idx].set_title('Predicted Anomaly Map')
        plt.colorbar(im1, ax=self.axes[col_idx], fraction=0.046, pad=0.04)
        col_idx += 1
        
        # True mask
        mask_true = tensor_to_numpy(data['mask_true']).squeeze()
        self.axes[col_idx].imshow(mask_true, cmap='gray', vmin=0, vmax=1)
        self.axes[col_idx].set_title('Ground Truth Mask')
        col_idx += 1
        
        # Overlay
        overlay = img_np.copy()
        if mask_true.max() > 0:
            # Add red overlay for true anomalies
            overlay[:, :, 0] = np.where(mask_true > 0.5, 1.0, overlay[:, :, 0])
        
        # Add yellow overlay for predicted anomalies
        pred_mask = anomaly_map > 0.5
        overlay[:, :, 1] = np.where(pred_mask, 1.0, overlay[:, :, 1])
        overlay[:, :, 0] = np.where(pred_mask, 1.0, overlay[:, :, 0])
        
        self.axes[col_idx].imshow(overlay)
        self.axes[col_idx].set_title('Overlay\n(Red: GT, Yellow: Pred)')
        
        # Update figure title with sample info
        label_str = 'Anomaly' if data['label'] == 1 else 'Normal'
        anomaly_score = self.compute_anomaly_score(data)
        title = f"Sample {self.current_idx + 1}/{len(self.test_data)} - {label_str} ({data['anomaly_type']}) - Score: {anomaly_score:.4f}"
        self.fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        plt.draw()
    
    def show(self):
        """Show the interactive visualization."""
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MVTec anomaly detection results')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='../datasets/mvtec_anomaly_detection',
                        help='Path to MVTec dataset root directory')
    parser.add_argument('--category', type=str, default='bottle',
                        help='Object category to visualize')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='anomaly_unet', choices=['unet', 'anomaly_unet'],
                        help='Model architecture')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transposed convolution')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (if None, will list available checkpoints)')
    
    # Visualization arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    return parser.parse_args()


def list_available_checkpoints(base_dir='../outputs'):
    """List available model checkpoints."""
    if not os.path.exists(base_dir):
        print(f"Output directory {base_dir} not found!")
        return []
    
    checkpoints = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pth'):
                checkpoints.append(os.path.join(root, file))
    
    return sorted(checkpoints)


def select_checkpoint():
    """Interactive checkpoint selection."""
    checkpoints = list_available_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found in ../outputs directory!")
        print("Please train a model first using: python train.py --category <category>")
        return None
    
    print("\nAvailable checkpoints:")
    for i, checkpoint in enumerate(checkpoints):
        rel_path = os.path.relpath(checkpoint)
        print(f"{i + 1}: {rel_path}")
    
    while True:
        try:
            choice = input(f"\nSelect checkpoint (1-{len(checkpoints)}) or 'q' to quit: ")
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            else:
                print("Invalid selection!")
        except ValueError:
            print("Please enter a valid number!")


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if category exists
    available_categories = get_available_categories(args.data_root)
    if args.category not in available_categories:
        print(f"Category '{args.category}' not found!")
        print(f"Available categories: {available_categories}")
        return
    
    # Select checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = select_checkpoint()
        if checkpoint_path is None:
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create data loader (test only)
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
    load_checkpoint(model, None, checkpoint_path, device)
    model.eval()
    
    # Create and show visualizer
    print("Starting visualization...")
    print("Use 'Previous' and 'Next' buttons to navigate samples")
    print("Use 'Show Info' button to print detailed information")
    print("Close the window to exit")
    
    visualizer = AnomalyVisualizer(model, test_loader, device)
    visualizer.show()


if __name__ == "__main__":
    main()

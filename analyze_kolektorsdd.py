#!/usr/bin/env python3
"""
Analysis script for KolektorSDD dataset to understand image and mask properties.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def analyze_kolektorsdd_dataset(dataset_root):
    """Analyze KolektorSDD dataset structure and properties"""
    
    print("=" * 60)
    print("🔍 KOLEKTORSDD DATASET ANALYSIS")
    print("=" * 60)
    
    # Find all image and mask files
    image_files = []
    mask_files = []
    
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
            elif file.endswith('.bmp'):
                mask_files.append(os.path.join(root, file))
    
    print(f"📂 Dataset root: {dataset_root}")
    print(f"📊 Total images: {len(image_files)}")
    print(f"📊 Total masks: {len(mask_files)}")
    print(f"📁 Number of folders: {len([d for d in os.listdir(dataset_root) if d.startswith('kos')])}")
    
    # Analyze sample images and masks
    sample_size = min(10, len(image_files))
    print(f"\n🔍 Analyzing {sample_size} sample images and masks...")
    
    image_sizes = []
    mask_sizes = []
    mask_values = []
    
    for i in range(sample_size):
        # Analyze image
        img_path = image_files[i]
        img = Image.open(img_path)
        image_sizes.append(img.size)  # (width, height)
        
        # Find corresponding mask
        base_name = os.path.basename(img_path).replace('.jpg', '')
        mask_path = img_path.replace('.jpg', '_label.bmp')
        
        if os.path.exists(mask_path):
            # Analyze mask using different methods
            
            # Method 1: PIL
            try:
                mask_pil = Image.open(mask_path)
                mask_sizes.append(mask_pil.size)
                mask_array_pil = np.array(mask_pil)
                print(f"\n📄 {base_name}:")
                print(f"   Image size: {img.size} (W×H)")
                print(f"   Mask size (PIL): {mask_pil.size} (W×H)")
                print(f"   Mask shape (PIL): {mask_array_pil.shape}")
                print(f"   Mask dtype (PIL): {mask_array_pil.dtype}")
                print(f"   Mask unique values (PIL): {np.unique(mask_array_pil)}")
                mask_values.extend(mask_array_pil.flatten())
            except Exception as e:
                print(f"   PIL error: {e}")
            
            # Additional analysis
            if mask_array_pil.size > 0:
                print(f"   Mask min/max: {mask_array_pil.min()}/{mask_array_pil.max()}")
                print(f"   Mask mean: {mask_array_pil.mean():.2f}")
        else:
            print(f"   ⚠️  Mask not found: {mask_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    
    if image_sizes:
        unique_image_sizes = list(set(image_sizes))
        print(f"🖼️  Unique image sizes: {unique_image_sizes}")
        
        if len(unique_image_sizes) == 1:
            w, h = unique_image_sizes[0]
            print(f"   All images are {w}×{h} pixels")
        
    if mask_sizes:
        unique_mask_sizes = list(set(mask_sizes))
        print(f"🎭 Unique mask sizes: {unique_mask_sizes}")
    
    if mask_values:
        unique_mask_values = np.unique(mask_values)
        print(f"🎨 Unique mask values across all samples: {unique_mask_values}")
        value_counts = Counter(mask_values)
        print(f"🔢 Value distribution in samples:")
        for value, count in sorted(value_counts.items()):
            percentage = (count / len(mask_values)) * 100
            print(f"   Value {value}: {count:,} pixels ({percentage:.2f}%)")
    
    # Check naming pattern
    print(f"\n📝 NAMING PATTERN ANALYSIS")
    print("=" * 30)
    
    # Group by folder
    folder_stats = {}
    for img_path in image_files[:20]:  # Check first 20
        folder = os.path.basename(os.path.dirname(img_path))
        filename = os.path.basename(img_path)
        
        if folder not in folder_stats:
            folder_stats[folder] = []
        folder_stats[folder].append(filename)
    
    for folder, files in folder_stats.items():
        print(f"📁 {folder}: {len(files)} files")
        print(f"   Sample files: {files[:3]}")
    
    return {
        'total_images': len(image_files),
        'total_masks': len(mask_files),
        'image_sizes': unique_image_sizes if image_sizes else [],
        'mask_sizes': unique_mask_sizes if mask_sizes else [],
        'mask_values': unique_mask_values if mask_values else [],
        'folder_count': len([d for d in os.listdir(dataset_root) if d.startswith('kos')])
    }

def visualize_samples(dataset_root, num_samples=3):
    """Visualize sample images and masks"""
    
    print(f"\n🖼️  VISUALIZING {num_samples} SAMPLES")
    print("=" * 40)
    
    # Find samples
    image_files = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
                if len(image_files) >= num_samples:
                    break
        if len(image_files) >= num_samples:
            break
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(image_files[:num_samples]):
        # Load image
        img = Image.open(img_path)
        
        # Load corresponding mask
        mask_path = img_path.replace('.jpg', '_label.bmp')
        
        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image: {os.path.basename(img_path)}')
        axes[i, 0].axis('off')
        
        # Display mask
        if os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                
                # Handle different mask formats
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[:, :, 0]  # Take first channel
                
                axes[i, 1].imshow(mask_array, cmap='viridis')
                axes[i, 1].set_title(f'Mask: {os.path.basename(mask_path)}')
                axes[i, 1].axis('off')
                
                print(f"✅ Sample {i+1}: Image {img.size}, Mask {mask_array.shape}, Values {np.unique(mask_array)}")
                
            except Exception as e:
                axes[i, 1].text(0.5, 0.5, f'Error loading mask:\n{str(e)}', 
                               ha='center', va='center', transform=axes[i, 1].transAxes)
                axes[i, 1].set_title(f'Mask Error: {os.path.basename(mask_path)}')
                print(f"❌ Sample {i+1}: Error loading mask - {e}")
        else:
            axes[i, 1].text(0.5, 0.5, 'Mask not found', 
                           ha='center', va='center', transform=axes[i, 1].transAxes)
            axes[i, 1].set_title('Mask not found')
    
    plt.tight_layout()
    plt.savefig('kolektorsdd_samples.png', dpi=150, bbox_inches='tight')
    print(f"💾 Sample visualization saved as 'kolektorsdd_samples.png'")
    plt.show()

if __name__ == "__main__":
    dataset_root = "/Users/uke/Desktop/tiaozhanbei/datasets/KolektorSDD"
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset not found at: {dataset_root}")
        exit(1)
    
    # Analyze dataset
    results = analyze_kolektorsdd_dataset(dataset_root)
    
    # Visualize samples
    try:
        visualize_samples(dataset_root, num_samples=3)
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    print(f"\n🎉 Analysis complete!")
    print(f"📋 Dataset has {results['total_images']} images across {results['folder_count']} folders")
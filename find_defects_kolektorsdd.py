#!/usr/bin/env python3
"""
Find samples with defects in KolektorSDD dataset
"""

import os
import numpy as np
from PIL import Image

def find_defect_samples(dataset_root, max_samples=10):
    """Find samples that contain defects (non-zero mask values)"""
    
    print("🔍 Searching for defect samples...")
    
    defect_samples = []
    no_defect_count = 0
    
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                mask_path = img_path.replace('.jpg', '_label.bmp')
                
                if os.path.exists(mask_path):
                    try:
                        mask = Image.open(mask_path)
                        mask_array = np.array(mask)
                        
                        # Check if mask has defects (non-zero values)
                        unique_values = np.unique(mask_array)
                        if len(unique_values) > 1 or unique_values[0] != 0:
                            defect_info = {
                                'image_path': img_path,
                                'mask_path': mask_path,
                                'image_size': Image.open(img_path).size,
                                'mask_shape': mask_array.shape,
                                'unique_values': unique_values,
                                'defect_pixels': np.sum(mask_array > 0),
                                'total_pixels': mask_array.size,
                                'defect_percentage': (np.sum(mask_array > 0) / mask_array.size) * 100
                            }
                            defect_samples.append(defect_info)
                            
                            if len(defect_samples) >= max_samples:
                                break
                        else:
                            no_defect_count += 1
                            
                    except Exception as e:
                        print(f"Error processing {mask_path}: {e}")
        
        if len(defect_samples) >= max_samples:
            break
    
    print(f"\n📊 DEFECT ANALYSIS RESULTS")
    print("=" * 40)
    print(f"✅ Found {len(defect_samples)} samples with defects")
    print(f"❌ Found {no_defect_count} samples without defects")
    
    if defect_samples:
        print(f"\n🔍 DEFECT SAMPLES DETAILS:")
        print("-" * 60)
        
        for i, sample in enumerate(defect_samples):
            print(f"\n{i+1}. {os.path.basename(sample['image_path'])}")
            print(f"   📁 Folder: {os.path.basename(os.path.dirname(sample['image_path']))}")
            print(f"   📏 Size: {sample['image_size']} (W×H)")
            print(f"   🎨 Mask values: {sample['unique_values']}")
            print(f"   ⚫ Defect pixels: {sample['defect_pixels']:,}")
            print(f"   📊 Defect percentage: {sample['defect_percentage']:.3f}%")
    
    return defect_samples

if __name__ == "__main__":
    dataset_root = "/Users/uke/Desktop/tiaozhanbei/datasets/KolektorSDD"
    
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset not found at: {dataset_root}")
        exit(1)
    
    defect_samples = find_defect_samples(dataset_root, max_samples=20)
    
    print(f"\n🎯 SUMMARY:")
    if defect_samples:
        total_defect_pixels = sum(s['defect_pixels'] for s in defect_samples)
        total_pixels = sum(s['total_pixels'] for s in defect_samples)
        avg_defect_percentage = (total_defect_pixels / total_pixels) * 100
        
        print(f"📊 Average defect percentage in defective samples: {avg_defect_percentage:.3f}%")
        print(f"🎨 All unique mask values found: {sorted(set().union(*[s['unique_values'] for s in defect_samples]))}")
    else:
        print("⚠️  No defect samples found in the analyzed portion")
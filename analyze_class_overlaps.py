#!/usr/bin/env python3
"""
Script to analyze class overlaps in the Gear dataset.
This script calculates how many pixels each pair of classes overlap
to understand the extent of multi-label regions before implementing 
priority-based label resolution.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def parse_label_file_raw(label_path, img_width, img_height):
    """
    Parse label file and create separate masks for each class
    Returns a dictionary with class_id as key and mask as value
    """
    class_masks = {}
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:  # class_id + at least 4 coordinates
                        class_id = int(parts[0])
                        
                        # Extract normalized coordinates
                        coords = [float(x) for x in parts[1:]]
                        
                        # Convert to pixel coordinates
                        pixel_coords = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x = int(coords[i] * img_width)
                                y = int(coords[i + 1] * img_height)
                                pixel_coords.append((x, y))
                        
                        # Create polygon mask for this class
                        if len(pixel_coords) >= 3:  # Need at least 3 points for a polygon
                            if class_id not in class_masks:
                                class_masks[class_id] = np.zeros((img_height, img_width), dtype=bool)
                            
                            # Create PIL image for drawing
                            mask_img = Image.fromarray(class_masks[class_id].astype(np.uint8))
                            draw = ImageDraw.Draw(mask_img)
                            draw.polygon(pixel_coords, fill=1)
                            
                            # Update the class mask
                            class_masks[class_id] = np.array(mask_img).astype(bool)
    
    except Exception as e:
        print(f"Warning: Could not parse label file {label_path}: {e}")
    
    return class_masks

def calculate_overlaps(root_dir, splits=['train', 'val', 'test']):
    """
    Calculate overlaps between all class pairs across all splits
    """
    class_names = {0: 'pitting', 1: 'spalling', 2: 'scrape'}
    
    # Store overlap statistics
    overlap_stats = {
        'total_pixels_per_class': defaultdict(int),
        'overlap_pixels': defaultdict(int),
        'overlap_percentages': defaultdict(float),
        'files_with_overlaps': defaultdict(list),
        'detailed_stats': []
    }
    
    total_files_processed = 0
    files_with_any_overlap = 0
    
    # Process each split
    for split in splits:
        images_dir = os.path.join(root_dir, 'images', split)
        labels_dir = os.path.join(root_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {split} split - directories not found")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Analyzing {split}"):
            # Load image to get dimensions
            img_path = os.path.join(images_dir, img_file)
            image = Image.open(img_path)
            img_width, img_height = image.size
            
            # Corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
            
            # Parse label file to get individual class masks
            class_masks = parse_label_file_raw(label_path, img_width, img_height)
            
            if not class_masks:
                continue
            
            total_files_processed += 1
            file_has_overlap = False
            
            # Calculate total pixels for each class in this image
            for class_id, mask in class_masks.items():
                total_pixels = np.sum(mask)
                overlap_stats['total_pixels_per_class'][class_id] += total_pixels
            
            # Calculate overlaps between all pairs of classes
            class_ids = list(class_masks.keys())
            for i, class_a in enumerate(class_ids):
                for j, class_b in enumerate(class_ids):
                    if i < j:  # Only calculate each pair once
                        mask_a = class_masks[class_a]
                        mask_b = class_masks[class_b]
                        
                        # Calculate overlap
                        overlap_mask = mask_a & mask_b
                        overlap_pixels = np.sum(overlap_mask)
                        
                        if overlap_pixels > 0:
                            file_has_overlap = True
                            pair_key = f"{class_names[class_a]}_vs_{class_names[class_b]}"
                            overlap_stats['overlap_pixels'][pair_key] += overlap_pixels
                            overlap_stats['files_with_overlaps'][pair_key].append(
                                f"{split}/{img_file}"
                            )
                            
                            # Store detailed stats for this file
                            overlap_stats['detailed_stats'].append({
                                'file': f"{split}/{img_file}",
                                'class_a': class_names[class_a],
                                'class_b': class_names[class_b],
                                'overlap_pixels': int(overlap_pixels),
                                'class_a_total': int(np.sum(mask_a)),
                                'class_b_total': int(np.sum(mask_b)),
                                'overlap_ratio_a': float(overlap_pixels / np.sum(mask_a)) if np.sum(mask_a) > 0 else 0,
                                'overlap_ratio_b': float(overlap_pixels / np.sum(mask_b)) if np.sum(mask_b) > 0 else 0
                            })
            
            if file_has_overlap:
                files_with_any_overlap += 1
    
    # Calculate percentages
    for pair_key, overlap_pixels in overlap_stats['overlap_pixels'].items():
        class_a_name, class_b_name = pair_key.split('_vs_')
        class_a_id = {v: k for k, v in class_names.items()}[class_a_name]
        class_b_id = {v: k for k, v in class_names.items()}[class_b_name]
        
        total_a = overlap_stats['total_pixels_per_class'][class_a_id]
        total_b = overlap_stats['total_pixels_per_class'][class_b_id]
        
        if total_a > 0:
            overlap_stats['overlap_percentages'][f"{pair_key}_pct_of_{class_a_name}"] = (overlap_pixels / total_a) * 100
        if total_b > 0:
            overlap_stats['overlap_percentages'][f"{pair_key}_pct_of_{class_b_name}"] = (overlap_pixels / total_b) * 100
    
    # Add summary statistics
    overlap_stats['summary'] = {
        'total_files_processed': total_files_processed,
        'files_with_any_overlap': files_with_any_overlap,
        'percentage_files_with_overlap': (files_with_any_overlap / total_files_processed * 100) if total_files_processed > 0 else 0,
        'class_names': class_names,
        'total_pixels_per_class_name': {class_names[k]: v for k, v in overlap_stats['total_pixels_per_class'].items()}
    }
    
    return overlap_stats

def print_overlap_analysis(overlap_stats):
    """Print detailed overlap analysis"""
    print("\n" + "="*80)
    print("🔍 CLASS OVERLAP ANALYSIS RESULTS")
    print("="*80)
    
    summary = overlap_stats['summary']
    print(f"📁 Total files processed: {summary['total_files_processed']}")
    print(f"⚠️  Files with overlaps: {summary['files_with_any_overlap']}")
    print(f"📊 Percentage with overlaps: {summary['percentage_files_with_overlap']:.2f}%")
    
    print(f"\n📏 Total pixels per class:")
    for class_name, total_pixels in summary['total_pixels_per_class_name'].items():
        print(f"   {class_name:>10}: {total_pixels:>12,} pixels")
    
    print(f"\n🔗 Overlap Statistics:")
    print(f"{'Class Pair':<20} {'Overlap Pixels':<15} {'Files':<8} {'% of Class A':<12} {'% of Class B':<12}")
    print("-" * 80)
    
    for pair_key, overlap_pixels in overlap_stats['overlap_pixels'].items():
        class_a_name, class_b_name = pair_key.split('_vs_')
        num_files = len(overlap_stats['files_with_overlaps'][pair_key])
        
        pct_a_key = f"{pair_key}_pct_of_{class_a_name}"
        pct_b_key = f"{pair_key}_pct_of_{class_b_name}"
        
        pct_a = overlap_stats['overlap_percentages'].get(pct_a_key, 0)
        pct_b = overlap_stats['overlap_percentages'].get(pct_b_key, 0)
        
        print(f"{pair_key:<20} {overlap_pixels:<15,} {num_files:<8} {pct_a:<11.2f}% {pct_b:<11.2f}%")
    
    # Show most problematic overlaps
    if overlap_stats['detailed_stats']:
        print(f"\n🚨 Top 10 largest overlaps:")
        detailed_stats = sorted(overlap_stats['detailed_stats'], 
                              key=lambda x: x['overlap_pixels'], reverse=True)
        
        print(f"{'File':<25} {'Classes':<20} {'Overlap':<10} {'Ratio A':<10} {'Ratio B':<10}")
        print("-" * 80)
        
        for stat in detailed_stats[:10]:
            file_short = stat['file'].split('/')[-1][:20]
            classes = f"{stat['class_a']} vs {stat['class_b']}"
            print(f"{file_short:<25} {classes:<20} {stat['overlap_pixels']:<10,} "
                  f"{stat['overlap_ratio_a']:<9.3f} {stat['overlap_ratio_b']:<9.3f}")

def create_overlap_visualizations(overlap_stats, save_dir):
    """Create visualizations of overlap statistics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Overlap matrix heatmap
    class_names = ['pitting', 'spalling', 'scrape']
    overlap_matrix = np.zeros((len(class_names), len(class_names)))
    
    for pair_key, overlap_pixels in overlap_stats['overlap_pixels'].items():
        class_a_name, class_b_name = pair_key.split('_vs_')
        i = class_names.index(class_a_name)
        j = class_names.index(class_b_name)
        overlap_matrix[i, j] = overlap_pixels
        overlap_matrix[j, i] = overlap_pixels  # Make symmetric
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, 
                xticklabels=class_names, 
                yticklabels=class_names,
                annot=True, 
                fmt='.0f',
                cmap='Reds',
                cbar_kws={'label': 'Overlap Pixels'})
    plt.title('Class Overlap Matrix (Pixel Count)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overlap_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Overlap percentages bar chart
    if overlap_stats['overlap_percentages']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Percentage of each class that overlaps
        pct_data = {}
        for key, value in overlap_stats['overlap_percentages'].items():
            if 'pct_of_' in key:
                pair, class_name = key.split('_pct_of_')
                class_a, class_b = pair.split('_vs_')
                other_class = class_b if class_name == class_a else class_a
                pct_data[f"{class_name}\nvs {other_class}"] = value
        
        if pct_data:
            bars = ax1.bar(range(len(pct_data)), list(pct_data.values()))
            ax1.set_xticks(range(len(pct_data)))
            ax1.set_xticklabels(list(pct_data.keys()), rotation=45, ha='right')
            ax1.set_ylabel('Percentage of Class Pixels')
            ax1.set_title('Percentage of Each Class That Overlaps')
            
            # Color bars by percentage
            for bar, pct in zip(bars, pct_data.values()):
                if pct > 10:
                    bar.set_color('red')
                elif pct > 5:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
        
        # Files with overlaps
        files_data = {pair: len(files) for pair, files in overlap_stats['files_with_overlaps'].items()}
        if files_data:
            ax2.bar(range(len(files_data)), list(files_data.values()))
            ax2.set_xticks(range(len(files_data)))
            ax2.set_xticklabels(list(files_data.keys()), rotation=45, ha='right')
            ax2.set_ylabel('Number of Files')
            ax2.set_title('Number of Files with Each Overlap Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overlap_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Visualizations saved to: {save_dir}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze class overlaps in Gear dataset')
    parser.add_argument('--data_root', type=str, default='datasets/Gear',
                       help='Path to Gear dataset root directory')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to analyze')
    parser.add_argument('--save_dir', type=str, default='overlap_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print("🔍 Starting class overlap analysis...")
    print(f"📁 Dataset root: {args.data_root}")
    print(f"📊 Analyzing splits: {args.splits}")
    
    # Calculate overlaps
    overlap_stats = calculate_overlaps(args.data_root, args.splits)
    
    # Print analysis
    print_overlap_analysis(overlap_stats)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_overlap_visualizations(overlap_stats, args.save_dir)
    
    # Save detailed results to JSON
    results_path = os.path.join(args.save_dir, 'overlap_analysis_detailed.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_stats = {}
    for key, value in overlap_stats.items():
        if key == 'detailed_stats':
            json_stats[key] = value  # Already serializable
        elif isinstance(value, dict):
            json_stats[key] = {}
            for k, v in value.items():
                if hasattr(v, 'tolist'):
                    json_stats[key][str(k)] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    json_stats[key][str(k)] = int(v) if isinstance(v, np.integer) else float(v)
                else:
                    json_stats[key][str(k)] = v
        else:
            if hasattr(value, 'tolist'):
                json_stats[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_stats[key] = int(value) if isinstance(value, np.integer) else float(value)
            else:
                json_stats[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_path}")
    
    # Print recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    
    summary = overlap_stats['summary']
    if summary['percentage_files_with_overlap'] > 10:
        print("⚠️  HIGH OVERLAP DETECTED:")
        print("   - Implementing priority-based resolution is STRONGLY recommended")
        print("   - Priority order: spalling > pitting > scrape")
        print("   - This will resolve conflicting labels systematically")
    elif summary['percentage_files_with_overlap'] > 5:
        print("⚠️  MODERATE OVERLAP DETECTED:")
        print("   - Priority-based resolution is recommended")
        print("   - Monitor training metrics for class imbalance issues")
    else:
        print("✅ LOW OVERLAP DETECTED:")
        print("   - Priority-based resolution may still be beneficial")
        print("   - Current approach might work but can be improved")
    
    print(f"\n🎯 Next steps:")
    print("1. Review the overlap statistics above")
    print("2. Examine the visualizations in the save directory")
    print("3. Implement priority-based label resolution if needed")
    print("4. Re-train models with resolved labels")

if __name__ == "__main__":
    main()
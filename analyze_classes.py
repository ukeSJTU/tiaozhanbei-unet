#!/usr/bin/env python3
"""
Quick script to analyze all classes in the Gear dataset across all splits.
"""

import os
import glob
from collections import Counter


def analyze_gear_classes(data_root):
    """Analyze all classes in the Gear dataset"""
    
    all_classes = set()
    class_counts = Counter()
    file_count = 0
    
    # Check all splits
    splits = ['train', 'val', 'test']
    
    for split in splits:
        labels_dir = os.path.join(data_root, 'labels', split)
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Directory not found: {labels_dir}")
            continue
        
        # Get all .txt files
        txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        print(f"\n{split.upper()} split: {len(txt_files)} label files")
        
        split_classes = set()
        
        for txt_file in txt_files:
            file_count += 1
            
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:  # class_id + at least 4 coordinates
                                class_id = int(parts[0])
                                all_classes.add(class_id)
                                split_classes.add(class_id)
                                class_counts[class_id] += 1
            
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
        
        print(f"Classes found in {split}: {sorted(split_classes)}")
    
    return all_classes, class_counts, file_count


def main():
    data_root = "datasets/Gear"
    
    print("="*60)
    print("GEAR DATASET CLASS ANALYSIS")
    print("="*60)
    
    all_classes, class_counts, file_count = analyze_gear_classes(data_root)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Total label files processed: {file_count}")
    print(f"All classes found: {sorted(all_classes)}")
    print(f"Total number of classes: {len(all_classes)}")
    
    if all_classes:
        print(f"Class range: {min(all_classes)} to {max(all_classes)}")
        
        print(f"\nClass frequency (number of polygon instances):")
        class_names = {0: "pitting", 1: "spalling", 2: "scrape"}
        for class_id in sorted(all_classes):
            class_name = class_names.get(class_id, f"unknown_{class_id}")
            print(f"  Class {class_id} ({class_name}): {class_counts[class_id]} instances")
    
    print(f"\nClass definitions:")
    print(f"  0 = pitting (surface fatigue defect)")
    print(f"  1 = spalling (material removal defect)")
    print(f"  2 = scrape (surface scratch defect)")
    
    print(f"\nFor training, you'll have:")
    print(f"  0 = background (unlabeled areas)")
    print(f"  1 = pitting (remapped from original 0)")
    print(f"  2 = spalling (remapped from original 1)")
    print(f"  3 = scrape (remapped from original 2)")
    print(f"- Total classes (including background): {len(all_classes) + 1}")


if __name__ == "__main__":
    main()
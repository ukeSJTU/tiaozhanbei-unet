# Gear Defect Segmentation

Multi-class semantic segmentation for gear defect detection using UNet architecture.

## Dataset Structure

The Gear dataset contains 1920×1080 images with LabelMe annotations:
```
datasets/Gear/
├── images/
│   ├── train/    # 448 training images
│   ├── val/      # 56 validation images  
│   └── test/     # 56 test images
└── labels/
    ├── train/    # LabelMe .txt annotations
    ├── val/      # LabelMe .txt annotations
    └── test/     # LabelMe .txt annotations
```

**Classes in training:**
- 0: Background (unlabeled areas)
- 1: Pitting (surface fatigue defect)
- 2: Spalling (material removal defect)  
- 3: Scrape (surface scratch defect)

**Original label mapping:**
- Label 0 → Pitting (remapped to class 1)
- Label 1 → Spalling (remapped to class 2)
- Label 2 → Scrape (remapped to class 3)

## Scripts

### 1. Training Script
```bash
python train.py --data_root datasets/Gear --epochs 50 --batch_size 8 --image_size 512
```

**Key arguments:**
- `--debug`: Enable debug mode with limited samples
- `--debug_samples 20`: Number of samples in debug mode
- `--model seg_unet`: Use SegmentationUNet (recommended)
- `--dropout 0.1`: Dropout rate for regularization
- `--ce_weight 1.0 --dice_weight 1.0`: Loss function weights
- `--class_weights "1.0,0.96,2.71,1.5"`: Class weights for imbalanced data (background, pitting, spalling, scrape)

### 2. Test Script
```bash
python test.py --checkpoint outputs/best_model.pth --data_root datasets/Gear --split test
```

**Key arguments:**
- `--save_predictions`: Save prediction visualizations
- `--save_confusion_matrix`: Save confusion matrix plot
- `--debug`: Test on limited samples

### 3. Visualization Script
```bash
python visualize.py --checkpoint outputs/best_model.pth --data_root datasets/Gear --num_samples 10
```

**Key arguments:**
- `--save_individual`: Save individual prediction images
- `--save_grid`: Save grid of multiple predictions
- `--show_confidence`: Display prediction confidence maps
- `--split test`: Dataset split to visualize

## Model Architecture

**SegmentationUNet** (recommended):
- Multi-class UNet with dropout regularization
- Optimized for semantic segmentation
- ~31M parameters

**Standard UNet**:
- Original UNet architecture
- Can be used with `--model unet`

## Metrics

Comprehensive evaluation metrics:
- **IoU (Intersection over Union)**: Per-class and mean
- **Dice Coefficient**: Per-class and mean  
- **F1 Score**: Per-class and mean
- **Precision/Recall**: Per-class and mean
- **Pixel Accuracy**: Overall pixel-wise accuracy
- **Mean Accuracy**: Average per-class accuracy

## 🚀 **Complete Workflow**

### **Step-by-Step Getting Started**
```bash
# 1. First, analyze your dataset
python analyze_classes.py

# 2. Test everything works with debug mode (2-3 minutes)
python train.py --debug --debug_samples 20 --epochs 5

# 3. Start real training with optimized settings
python train.py --epochs 50 --batch_size 8 \
  --class_weights "1.0,1.56,1.0,2.82" \
  --ce_weight 1.0 --dice_weight 1.0

# 4. Evaluate your best model 
python test.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --split test --save_predictions

# 5. Create visualizations for analysis
python visualize.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --num_samples 20 --save_individual --save_grid
```

## Quick Start Commands

### 🔍 **Dataset Analysis**
```bash
# Analyze class distribution across all splits
python analyze_classes.py
```

### 🐛 **Debug Mode (Test Setup)**
```bash
# Quick debug training (20 samples, 5 epochs)
python train.py --debug --debug_samples 20 --epochs 5

# Debug with class weights for imbalanced data
python train.py --debug --debug_samples 20 --epochs 5 \
  --class_weights "1.0,1.56,1.0,2.82"

# Debug specific model architecture
python train.py --debug --debug_samples 20 --epochs 5 \
  --model seg_unet --dropout 0.2
```

### 🚀 **Training Scenarios**

#### **Basic Training**
```bash
# Standard training with default settings
python train.py --epochs 50 --batch_size 8
```

#### **Optimized Training (Recommended)**
```bash
# Training with class weights for gear defect imbalance
python train.py --epochs 50 --batch_size 8 \
  --class_weights "1.0,1.56,1.0,2.82" \
  --ce_weight 1.0 --dice_weight 1.0 \
  --save_freq 10
```

#### **High-Performance Training**
```bash
# Larger batch size with mixed loss functions
python train.py --epochs 100 --batch_size 16 \
  --class_weights "1.0,1.56,1.0,2.82" \
  --ce_weight 0.7 --dice_weight 1.0 --focal_weight 0.3 \
  --learning_rate 1e-3 --weight_decay 1e-4 \
  --optimizer adamw --dropout 0.15
```

#### **Resume Training**
```bash
# Resume from a checkpoint
python train.py --resume outputs/gear_seg_*/checkpoints/checkpoint_epoch_30.pth \
  --epochs 100 --class_weights "1.0,1.56,1.0,2.82"
```

### 📊 **Evaluation Scenarios**

#### **Quick Evaluation**
```bash
# Evaluate on test set with basic metrics
python test.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --split test
```

#### **Comprehensive Evaluation**
```bash
# Full evaluation with visualizations and confusion matrix
python test.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --split test --save_predictions --save_confusion_matrix
```

#### **Debug Evaluation**
```bash
# Quick evaluation on limited samples
python test.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --split test --debug --debug_samples 20
```

#### **Validation Set Evaluation**
```bash
# Evaluate on validation set instead of test
python test.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --split val --save_predictions
```

### 🎨 **Visualization Scenarios**

#### **Basic Visualization**
```bash
# Visualize 10 predictions from test set
python visualize.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --num_samples 10
```

#### **Comprehensive Visualization**
```bash
# Individual images + grid + confidence maps
python visualize.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --num_samples 20 --save_individual --save_grid --show_confidence
```

#### **Training Set Visualization**
```bash
# Visualize model performance on training data
python visualize.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --split train --num_samples 15 --save_individual
```

#### **Custom Grid Visualization**
```bash
# Custom grid layout for presentation
python visualize.py --checkpoint outputs/gear_seg_*/checkpoints/best_model.pth \
  --num_samples 12 --save_grid --grid_size 3 4 --figsize 20 15
```

### 🔧 **Advanced Scenarios**

#### **Hyperparameter Tuning**
```bash
# Experiment with different architectures
python train.py --model unet --epochs 30 --debug  # Standard UNet
python train.py --model seg_unet --dropout 0.2 --epochs 30 --debug  # With dropout

# Different optimizers
python train.py --optimizer adam --learning_rate 1e-3 --epochs 30 --debug
python train.py --optimizer adamw --learning_rate 5e-4 --epochs 30 --debug
python train.py --optimizer sgd --learning_rate 1e-2 --epochs 30 --debug
```

#### **Loss Function Experiments**
```bash
# Cross-entropy only
python train.py --ce_weight 1.0 --dice_weight 0.0 --epochs 30 --debug

# Dice loss only  
python train.py --ce_weight 0.0 --dice_weight 1.0 --epochs 30 --debug

# With focal loss for hard examples
python train.py --ce_weight 0.5 --dice_weight 1.0 --focal_weight 0.5 --epochs 30 --debug
```

#### **Different Image Sizes**
```bash
# Higher resolution (more GPU memory needed)
python train.py --image_size 768 --batch_size 4 --epochs 30 --debug

# Lower resolution (faster training)
python train.py --image_size 256 --batch_size 16 --epochs 30 --debug
```

### 📁 **Output Management**
```bash
# Custom output directory
python train.py --save_dir custom_experiments --epochs 50

# Different checkpoint frequency
python train.py --save_freq 5 --epochs 50  # Save every 5 epochs

# Custom validation frequency
python train.py --val_freq 2 --epochs 50   # Validate every 2 epochs
```

## File Structure
```
├── train.py              # Training script
├── test.py               # Evaluation script  
├── visualize.py          # Visualization script
├── src/
│   ├── gear_dataset.py   # Gear dataset loader
│   ├── model.py          # UNet models
│   ├── metrics.py        # Evaluation metrics
│   └── utils.py          # Utility functions
├── datasets/Gear/        # Dataset directory
└── outputs/              # Training outputs
```

## Notes

- Images are resized from 1920×1080 to 512×512 for training efficiency
- LabelMe polygon annotations are converted to pixel masks
- Model supports multi-class segmentation (background + defect classes)
- Debug mode limits dataset size for quick testing
- All scripts support argument customization for flexibility
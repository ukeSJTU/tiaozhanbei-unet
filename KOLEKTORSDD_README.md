# KolektorSDD Dataset Support

This project now supports the **KolektorSDD** (Surface Defect Detection) dataset in addition to the original Gear dataset.

## 📁 Dataset Structure

The KolektorSDD dataset should be placed in `datasets/KolektorSDD/` with the following structure:
```
datasets/KolektorSDD/
├── kos01/
│   ├── Part0.jpg
│   ├── Part0_label.bmp
│   ├── Part1.jpg
│   ├── Part1_label.bmp
│   └── ...
├── kos02/
│   └── ...
└── kos50/
    └── ...
```

## 🎯 Dataset Properties

- **Total samples**: 399 images with corresponding masks
- **Image format**: JPG (RGB)
- **Mask format**: BMP (grayscale)
- **Image sizes**: ~500×1240-1280 pixels (variable height)
- **Classes**: 3 classes
  - `0`: Background (no defect)
  - `1`: Defect type 1
  - `2`: Defect type 2
- **Task**: Surface defect detection/segmentation
- **Class distribution**: Highly imbalanced (~99.95% background, ~0.05% defects)

## 🚀 Usage

### Training
```bash
# Basic training
python train_kolektorsdd.py --epochs 50 --batch_size 8

# With custom image size
python train_kolektorsdd.py --epochs 50 --batch_size 8 --image_height 1024 --image_width 512

# With custom class weights (recommended for imbalanced data)
python train_kolektorsdd.py --epochs 50 --batch_size 8 --class_weights "1.0,50.0,50.0"

# Debug mode (fast testing)
python train_kolektorsdd.py --epochs 2 --batch_size 4 --debug --debug_samples 16
```

### Testing
```bash
# Test on test split
python test_kolektorsdd.py --checkpoint outputs/kolektorsdd_*/checkpoints/best_model.pth --split test

# Test on validation split
python test_kolektorsdd.py --checkpoint outputs/kolektorsdd_*/checkpoints/best_model.pth --split val

# Save predictions and confusion matrix
python test_kolektorsdd.py --checkpoint outputs/kolektorsdd_*/checkpoints/best_model.pth \
    --save_predictions --save_confusion_matrix
```

### Visualization
```bash
# Visualize dataset samples
python visualize_kolektorsdd.py --split train --num_samples 8
```

## ⚙️ Key Parameters

### Training Parameters
- `--image_height`: Input image height (default: 1024)
- `--image_width`: Input image width (default: 512)
- `--class_weights`: Class weights for imbalanced data (default: "1.0,50.0,50.0")
- `--train_split`: Fraction for training (default: 0.7)
- `--val_split`: Fraction for validation (default: 0.15)

### Recommended Settings
- **Image size**: 1024×512 (good balance of detail and efficiency)
- **Class weights**: "1.0,50.0,50.0" (accounts for class imbalance)
- **Batch size**: 4-8 (depending on GPU memory)
- **Learning rate**: 1e-3 (default)

## 📊 Data Splits

The dataset is automatically split into:
- **Training**: 70% (279 samples)
- **Validation**: 15% (60 samples)  
- **Testing**: 15% (60 samples)

Splits are reproducible (fixed random seed) and folder-aware.

## 🔧 Implementation Details

### Dataset Class
- **File**: `src/kolektorsdd_dataset.py`
- **Class**: `KolektorSDDDataset`
- **Features**:
  - Automatic train/val/test splitting
  - BMP mask loading and processing
  - Consistent image/mask resizing
  - Class value validation (clips to [0,1,2])

### Model Compatibility
- Uses the same UNet architectures as Gear dataset
- Supports both `UNet` and `SegmentationUNet` models
- Automatically configured for 3 classes

### Data Augmentation
- **Training**: Horizontal flip, rotation (5°), color jitter (reduced for industrial images)
- **Validation/Test**: Resize and normalize only

## 🎪 Example Training Command

```bash
# Full training run with recommended settings
python train_kolektorsdd.py \
    --epochs 100 \
    --batch_size 6 \
    --image_height 1024 \
    --image_width 512 \
    --class_weights "1.0,50.0,50.0" \
    --learning_rate 1e-3 \
    --val_freq 5 \
    --save_freq 10
```

## 📁 Output Structure

Training creates organized output directories:
```
outputs/kolektorsdd_seg_unet_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_model.pth
│   ├── checkpoint_epoch_10.pth
│   └── ...
├── results/
│   └── training_results.json
└── args.json
```

## 🔄 Integration with Existing Code

The KolektorSDD implementation is completely separate from the Gear dataset code:
- ✅ **Gear dataset unchanged**: `train.py`, `test.py`, `visualize.py` work as before
- ✅ **New KolektorSDD scripts**: `train_kolektorsdd.py`, `test_kolektorsdd.py`, `visualize_kolektorsdd.py`
- ✅ **Shared models**: Same UNet architectures work for both datasets
- ✅ **Shared utilities**: Metrics, losses, and utils are reused

## 🚨 Notes

1. **Performance**: Training is slow on CPU due to large image size (1024×512). GPU recommended.
2. **Memory**: Large images require significant GPU memory. Reduce batch size if needed.
3. **Class Imbalance**: Use class weights to handle severely imbalanced data.
4. **Defect Types**: Classes 1 and 2 represent different defect types (exact meaning unknown but trainable).
# MVTec Anomaly Detection with UNet

This repository implements UNet-based anomaly detection for the MVTec Anomaly Detection dataset. The implementation includes both standard UNet and a specialized AnomalyUNet architecture with dual decoders for reconstruction and segmentation.

## Features

- **Two Model Architectures**:
  - Standard UNet for anomaly segmentation
  - AnomalyUNet with dual decoders for reconstruction + segmentation
- **Comprehensive Training Pipeline**: Full training loop with validation, checkpointing, and metrics
- **Advanced Loss Functions**: Combined reconstruction and segmentation losses with focal loss for class imbalance
- **Extensive Evaluation**: Image-level and pixel-level metrics with per-anomaly-type analysis
- **Visualization Tools**: Training curves, confusion matrices, and result visualizations
- **Easy-to-use Scripts**: Simple command-line interface for training and testing

## Dataset Structure

The MVTec Anomaly Detection dataset should be organized as follows:
```
datasets/mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/          # Normal training images
│   ├── test/good/           # Normal test images
│   ├── test/broken_large/   # Anomalous test images
│   ├── test/broken_small/   # Anomalous test images
│   ├── test/contamination/  # Anomalous test images
│   └── ground_truth/        # Binary masks for anomalous regions
├── cable/
├── capsule/
└── ... (other categories)
```

## Installation

1. **Clone the repository** (if applicable) or ensure you have the source files
2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install torch torchvision pillow numpy matplotlib seaborn scikit-learn tqdm
   ```

## Quick Start

### 1. Test Dataset and Model Setup
```bash
cd src
python demo.py
```
This will test dataset loading, model creation, and generate sample visualizations.

### 2. Train a Model
```bash
cd src
python train.py --category bottle --epochs 50 --batch_size 16
```

### 3. Test a Trained Model
```bash
cd src
python test.py --category bottle --checkpoint ../outputs/experiment_name/checkpoints/best_model.pth --save_visualizations
```

## Available Categories

The following categories are typically available in the MVTec dataset:
- bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

Check available categories in your dataset:
```python
from dataset import get_available_categories
categories = get_available_categories("../datasets/mvtec_anomaly_detection")
print(categories)
```

## Training Options

### Basic Training
```bash
python train.py --category bottle --epochs 100
```

### Advanced Training
```bash
python train.py \
    --category bottle \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --model anomaly_unet \
    --optimizer adam \
    --scheduler cosine \
    --recon_weight 1.0 \
    --seg_weight 1.0 \
    --use_ssim
```

### Training Arguments
- `--category`: Object category to train on
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--model`: Model architecture (`unet` or `anomaly_unet`)
- `--optimizer`: Optimizer (`adam`, `adamw`, `sgd`)
- `--scheduler`: Learning rate scheduler (`cosine`, `step`, `plateau`, `none`)
- `--recon_weight`: Weight for reconstruction loss (default: 1.0)
- `--seg_weight`: Weight for segmentation loss (default: 1.0)
- `--use_ssim`: Use SSIM loss for reconstruction
- `--resume`: Path to checkpoint to resume from

## Testing Options

### Basic Testing
```bash
python test.py --category bottle --checkpoint path/to/checkpoint.pth
```

### Advanced Testing
```bash
python test.py \
    --category bottle \
    --checkpoint path/to/checkpoint.pth \
    --save_visualizations \
    --max_vis_samples 20 \
    --pixel_thresholds 0.3 0.5 0.7
```

### Testing Arguments
- `--category`: Object category to test on
- `--checkpoint`: Path to model checkpoint (required)
- `--threshold`: Threshold for anomaly detection (auto-optimized if not provided)
- `--save_visualizations`: Save visualization images
- `--max_vis_samples`: Maximum samples to visualize (default: 20)
- `--pixel_thresholds`: Thresholds for pixel-level evaluation

## Model Architectures

### Standard UNet
- Encoder-decoder architecture with skip connections
- Single output for anomaly segmentation
- Suitable for direct segmentation tasks

### AnomalyUNet
- Shared encoder with dual decoders
- Reconstruction decoder: Reconstructs input images
- Segmentation decoder: Predicts anomaly masks
- Better for unsupervised anomaly detection

## Loss Functions

### Combined Loss
- **Reconstruction Loss**: MSE or SSIM loss between input and reconstruction
- **Segmentation Loss**: Focal loss for anomaly mask prediction
- **Total Loss**: Weighted combination of both losses

### Focal Loss
- Addresses class imbalance in anomaly segmentation
- Focuses learning on hard examples
- Configurable alpha and gamma parameters

## Evaluation Metrics

### Image-level Metrics
- Accuracy, Precision, Recall, F1-score
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)

### Pixel-level Metrics
- Pixel-wise accuracy, precision, recall, F1-score
- Evaluated at multiple thresholds
- Only computed for anomalous images

### Per-anomaly-type Analysis
- Separate metrics for each anomaly type
- Helps identify model strengths and weaknesses

## Output Structure

Training outputs are saved to:
```
outputs/
└── {category}_{model}_{timestamp}/
    ├── checkpoints/
    │   ├── best_model.pth
    │   └── checkpoint_epoch_*.pth
    ├── results/
    │   ├── training_curves.png
    │   └── training_results.json
    └── visualizations/
```

Testing outputs are saved to:
```
test_results/
└── {category}_test_results/
    ├── test_metrics.json
    ├── detailed_results.json
    ├── confusion_matrix.png
    └── visualizations.png
```

## Tips for Better Results

1. **Data Preprocessing**: The dataset loader includes normalization and augmentation
2. **Model Selection**: Use AnomalyUNet for better reconstruction-based anomaly detection
3. **Loss Weighting**: Adjust `--recon_weight` and `--seg_weight` based on your priorities
4. **Learning Rate**: Start with 1e-3 and adjust based on training curves
5. **Batch Size**: Use largest batch size that fits in memory (16-32 typically work well)
6. **Epochs**: 50-100 epochs are usually sufficient for convergence

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or image size
2. **Dataset not found**: Check dataset path and structure
3. **Poor performance**: Try different loss weights or learning rates
4. **Slow training**: Increase num_workers or use GPU

### Performance Optimization
- Use GPU if available
- Increase num_workers for data loading
- Use mixed precision training for larger models
- Consider smaller image sizes for faster training

## File Structure

```
src/
├── model.py          # UNet and AnomalyUNet architectures
├── dataset.py        # MVTec dataset loader
├── train_utils.py    # Training utilities and loss functions
├── utils.py          # General utilities and metrics
├── train.py          # Training script
├── test.py           # Testing script
└── demo.py           # Demo script for quick testing
```

## Citation

If you use this implementation, please cite the original MVTec paper:

```bibtex
@inproceedings{bergmann2019mvtec,
  title={A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

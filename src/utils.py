import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image with mean and standard deviation."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}, epoch {epoch}, loss {loss:.4f}")
    return epoch, loss


def calculate_metrics(y_true, y_pred, y_scores=None):
    """Calculate various metrics for anomaly detection."""
    metrics = {}

    # Ensure arrays are numpy arrays with correct dtype
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # Ensure both arrays are 1D
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    # Basic classification metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # AUC metrics if scores are provided
    if y_scores is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            metrics['auprc'] = auc(recall, precision)
        except ValueError:
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
    
    return metrics


def calculate_pixel_metrics(y_true_masks, y_pred_masks, threshold=0.5):
    """Calculate pixel-level metrics for segmentation."""
    # Convert to binary masks
    y_true_binary = (y_true_masks > 0.5).astype(np.uint8)
    y_pred_binary = (y_pred_masks > threshold).astype(np.uint8)
    
    # Flatten arrays
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()
    
    # Calculate metrics
    return calculate_metrics(y_true_flat, y_pred_flat, y_pred_masks.flatten())


def visualize_results(images, masks_true, masks_pred, reconstructions=None, save_path=None, max_samples=8):
    """Visualize anomaly detection results."""
    n_samples = min(len(images), max_samples)
    n_cols = 4 if reconstructions is not None else 3
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 4, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original image
        img = denormalize_image(images[i])
        img = torch.clamp(img, 0, 1)
        img_np = tensor_to_numpy(img).transpose(1, 2, 0)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # True mask
        mask_true = tensor_to_numpy(masks_true[i]).squeeze()
        axes[i, 1].imshow(mask_true, cmap='gray')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask
        mask_pred = tensor_to_numpy(masks_pred[i]).squeeze()
        axes[i, 2].imshow(mask_pred, cmap='hot')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
        
        # Reconstruction (if available)
        if reconstructions is not None:
            recon = torch.clamp(reconstructions[i], 0, 1)
            recon_np = tensor_to_numpy(recon).transpose(1, 2, 0)
            axes[i, 3].imshow(recon_np)
            axes[i, 3].set_title('Reconstruction')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=['Normal', 'Anomaly'], save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_anomaly_score(reconstruction, original, method='mse'):
    """Compute anomaly score based on reconstruction error."""
    if method == 'mse':
        return F.mse_loss(reconstruction, original, reduction='none').mean(dim=1)
    elif method == 'l1':
        return F.l1_loss(reconstruction, original, reduction='none').mean(dim=1)
    elif method == 'ssim':
        # Simplified SSIM-based score (would need proper SSIM implementation)
        return F.mse_loss(reconstruction, original, reduction='none').mean(dim=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def create_output_dirs(base_dir):
    """Create necessary output directories."""
    dirs = ['checkpoints', 'results', 'visualizations']
    created_dirs = {}
    
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        created_dirs[dir_name] = dir_path
    
    return created_dirs


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_metrics(metrics, prefix=""):
    """Print metrics in a formatted way."""
    print(f"\n{prefix} Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.capitalize()}: {value:.4f}")
        else:
            print(f"{key.capitalize()}: {value}")
    print("-" * 40)


def get_optimal_threshold(y_true, y_scores):
    """Find optimal threshold for binary classification using Youden's J statistic."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find threshold with maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold, f1_scores[optimal_idx]


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test metrics calculation
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    y_scores = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4, 0.1, 0.7])
    
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    print_metrics(metrics, "Test")
    
    # Test optimal threshold
    threshold, f1 = get_optimal_threshold(y_true, y_scores)
    print(f"\nOptimal threshold: {threshold:.3f}, F1 score: {f1:.3f}")
    
    print("Utility functions test completed!")

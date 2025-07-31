import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationMetrics:
    """Comprehensive metrics for semantic segmentation tasks"""
    
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, pred, target):
        """
        Update metrics with new predictions and targets
        
        Args:
            pred: (N, H, W) or (N, C, H, W) predictions
            target: (N, H, W) ground truth labels
        """
        if pred.dim() == 4:  # (N, C, H, W)
            pred = torch.argmax(pred, dim=1)  # (N, H, W)
        
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # Remove ignored pixels
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        
        # Update confusion matrix
        cm = confusion_matrix(target, pred, labels=range(self.num_classes))
        self.confusion_matrix += cm
        self.total_samples += len(target)
    
    def compute_iou(self, per_class=True):
        """Compute Intersection over Union (IoU)"""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - 
                intersection)
        
        # Avoid division by zero
        union = np.maximum(union, 1e-8)
        iou = intersection / union
        
        if per_class:
            return iou
        else:
            return np.nanmean(iou)  # Mean IoU
    
    def compute_dice(self, per_class=True):
        """Compute Dice coefficient"""
        intersection = np.diag(self.confusion_matrix)
        dice_denominator = (self.confusion_matrix.sum(axis=1) + 
                           self.confusion_matrix.sum(axis=0))
        
        # Avoid division by zero
        dice_denominator = np.maximum(dice_denominator, 1e-8)
        dice = 2 * intersection / dice_denominator
        
        if per_class:
            return dice
        else:
            return np.nanmean(dice)  # Mean Dice
    
    def compute_pixel_accuracy(self):
        """Compute pixel-wise accuracy"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / max(total, 1e-8)
    
    def compute_mean_accuracy(self):
        """Compute mean class accuracy"""
        class_accuracies = np.diag(self.confusion_matrix) / np.maximum(
            self.confusion_matrix.sum(axis=1), 1e-8)
        return np.nanmean(class_accuracies)
    
    def compute_precision_recall_f1(self, per_class=True):
        """Compute precision, recall, and F1 score"""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Precision
        precision = tp / np.maximum(tp + fp, 1e-8)
        
        # Recall
        recall = tp / np.maximum(tp + fn, 1e-8)
        
        # F1 Score
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        
        if per_class:
            return precision, recall, f1
        else:
            return np.nanmean(precision), np.nanmean(recall), np.nanmean(f1)
    
    def compute_all_metrics(self):
        """Compute all metrics and return as dictionary"""
        metrics = {}
        
        # IoU metrics
        iou_per_class = self.compute_iou(per_class=True)
        metrics['iou_per_class'] = iou_per_class
        metrics['mean_iou'] = np.nanmean(iou_per_class)
        
        # Dice metrics
        dice_per_class = self.compute_dice(per_class=True)
        metrics['dice_per_class'] = dice_per_class
        metrics['mean_dice'] = np.nanmean(dice_per_class)
        
        # Accuracy metrics
        metrics['pixel_accuracy'] = self.compute_pixel_accuracy()
        metrics['mean_accuracy'] = self.compute_mean_accuracy()
        
        # Precision, Recall, F1
        precision, recall, f1 = self.compute_precision_recall_f1(per_class=True)
        metrics['precision_per_class'] = precision
        metrics['recall_per_class'] = recall
        metrics['f1_per_class'] = f1
        metrics['mean_precision'] = np.nanmean(precision)
        metrics['mean_recall'] = np.nanmean(recall)
        metrics['mean_f1'] = np.nanmean(f1)
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.confusion_matrix
        
        return metrics
    
    def print_metrics(self, class_names=None):
        """Print all metrics in a formatted way"""
        metrics = self.compute_all_metrics()
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        print("=" * 60)
        print("SEMANTIC SEGMENTATION METRICS")
        print("=" * 60)
        
        # Overall metrics
        print(f"Pixel Accuracy:     {metrics['pixel_accuracy']:.4f}")
        print(f"Mean Accuracy:      {metrics['mean_accuracy']:.4f}")
        print(f"Mean IoU:          {metrics['mean_iou']:.4f}")
        print(f"Mean Dice:         {metrics['mean_dice']:.4f}")
        print(f"Mean Precision:    {metrics['mean_precision']:.4f}")
        print(f"Mean Recall:       {metrics['mean_recall']:.4f}")
        print(f"Mean F1:           {metrics['mean_f1']:.4f}")
        
        print("\n" + "=" * 60)
        print("PER-CLASS METRICS")
        print("=" * 60)
        
        # Per-class metrics
        print(f"{'Class':<15} {'IoU':<8} {'Dice':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
        print("-" * 60)
        
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<15} "
                  f"{metrics['iou_per_class'][i]:.4f}   "
                  f"{metrics['dice_per_class'][i]:.4f}   "
                  f"{metrics['precision_per_class'][i]:.4f}   "
                  f"{metrics['recall_per_class'][i]:.4f}   "
                  f"{metrics['f1_per_class'][i]:.4f}")
    
    def plot_confusion_matrix(self, class_names=None, save_path=None, figsize=(10, 8)):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = self.confusion_matrix.astype('float') / (
            self.confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


def compute_metrics_from_predictions(predictions, targets, num_classes, class_names=None):
    """
    Compute metrics from prediction and target tensors
    
    Args:
        predictions: (N, C, H, W) or (N, H, W) tensor of predictions
        targets: (N, H, W) tensor of ground truth labels
        num_classes: Number of classes
        class_names: List of class names
    
    Returns:
        Dictionary of computed metrics
    """
    metrics_calculator = SegmentationMetrics(num_classes)
    
    # Handle batch dimension
    if predictions.dim() == 4:  # (N, C, H, W)
        for i in range(predictions.size(0)):
            metrics_calculator.update(predictions[i:i+1], targets[i:i+1])
    else:  # (N, H, W)
        for i in range(predictions.size(0)):
            metrics_calculator.update(predictions[i:i+1], targets[i:i+1])
    
    return metrics_calculator.compute_all_metrics()


def dice_loss(pred, target, smooth=1e-8):
    """
    Dice loss for segmentation
    
    Args:
        pred: (N, C, H, W) predictions (after softmax)
        target: (N, H, W) ground truth labels
        smooth: Smoothing factor
    
    Returns:
        Dice loss
    """
    # Convert target to one-hot
    num_classes = pred.size(1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    # Flatten tensors
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    
    # Compute dice coefficient
    dice = (2 * intersection + smooth) / (union + smooth)
    
    # Return loss (1 - dice)
    return 1 - dice.mean()


def focal_loss(pred, target, alpha=1, gamma=2, ignore_index=None):
    """
    Focal loss for addressing class imbalance
    
    Args:
        pred: (N, C, H, W) predictions (logits)
        target: (N, H, W) ground truth labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        ignore_index: Index to ignore
    
    Returns:
        Focal loss
    """
    ce_loss = F.cross_entropy(pred, target, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()


class CombinedSegmentationLoss(torch.nn.Module):
    """Combined loss for segmentation"""
    
    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.0, 
                 ignore_index=None, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (N, C, H, W) predictions (logits)
            target: (N, H, W) ground truth labels
        """
        loss = 0
        
        # Cross entropy loss
        if self.ce_weight > 0:
            # Ensure class weights are on the same device as pred
            class_weights = self.class_weights
            if class_weights is not None:
                class_weights = class_weights.to(pred.device)
            
            if self.ignore_index is not None:
                ce_loss = F.cross_entropy(pred, target, 
                                        weight=class_weights,
                                        ignore_index=self.ignore_index)
            else:
                ce_loss = F.cross_entropy(pred, target, 
                                        weight=class_weights)
            loss += self.ce_weight * ce_loss
        
        # Dice loss
        if self.dice_weight > 0:
            pred_softmax = F.softmax(pred, dim=1)
            dice_loss_val = dice_loss(pred_softmax, target)
            loss += self.dice_weight * dice_loss_val
        
        # Focal loss
        if self.focal_weight > 0:
            focal_loss_val = focal_loss(pred, target, ignore_index=self.ignore_index)
            loss += self.focal_weight * focal_loss_val
        
        return loss


if __name__ == "__main__":
    # Test metrics computation
    num_classes = 4
    batch_size = 2
    height, width = 64, 64
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test metrics
    metrics = compute_metrics_from_predictions(predictions, targets, num_classes)
    
    # Print metrics
    calc = SegmentationMetrics(num_classes)
    calc.update(predictions, targets)
    calc.print_metrics()
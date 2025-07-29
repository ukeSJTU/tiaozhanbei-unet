import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import AverageMeter, calculate_metrics, compute_anomaly_score


class CombinedLoss(nn.Module):
    """Combined loss for anomaly detection with reconstruction and segmentation."""
    
    def __init__(self, recon_weight=1.0, seg_weight=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.seg_weight = seg_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def focal_loss(self, pred, target):
        """Focal loss for handling class imbalance in segmentation."""
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        return focal_loss.mean()
    
    def forward(self, reconstruction, anomaly_map, original_image, true_mask):
        # Reconstruction loss
        recon_loss = self.mse_loss(reconstruction, original_image)
        
        # Segmentation loss (focal loss for better handling of imbalanced data)
        seg_loss = self.focal_loss(anomaly_map, true_mask)
        
        # Combined loss
        total_loss = self.recon_weight * recon_loss + self.seg_weight * seg_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'seg_loss': seg_loss
        }


class SSIMLoss(nn.Module):
    """SSIM Loss for better perceptual reconstruction."""
    
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    recon_losses = AverageMeter()
    seg_losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        reconstruction, anomaly_map = model(images)
        
        # Calculate loss
        loss_dict = criterion(reconstruction, anomaly_map, images, masks)
        total_loss = loss_dict['total_loss']
        recon_loss = loss_dict['recon_loss']
        seg_loss = loss_dict['seg_loss']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update meters
        batch_size = images.size(0)
        losses.update(total_loss.item(), batch_size)
        recon_losses.update(recon_loss.item(), batch_size)
        seg_losses.update(seg_loss.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Recon': f'{recon_losses.avg:.4f}',
            'Seg': f'{seg_losses.avg:.4f}'
        })
    
    return {
        'total_loss': losses.avg,
        'recon_loss': recon_losses.avg,
        'seg_loss': seg_losses.avg
    }


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    
    losses = AverageMeter()
    recon_losses = AverageMeter()
    seg_losses = AverageMeter()
    
    all_labels = []
    all_scores = []
    all_masks_true = []
    all_masks_pred = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].numpy()
            
            # Forward pass
            reconstruction, anomaly_map = model(images)
            
            # Calculate loss
            loss_dict = criterion(reconstruction, anomaly_map, images, masks)
            total_loss = loss_dict['total_loss']
            recon_loss = loss_dict['recon_loss']
            seg_loss = loss_dict['seg_loss']
            
            # Update meters
            batch_size = images.size(0)
            losses.update(total_loss.item(), batch_size)
            recon_losses.update(recon_loss.item(), batch_size)
            seg_losses.update(seg_loss.item(), batch_size)
            
            # Collect predictions for metrics
            anomaly_scores = compute_anomaly_score(reconstruction, images).cpu().numpy()

            all_labels.extend(labels)
            all_scores.extend(anomaly_scores)
            all_masks_true.extend(masks.cpu().numpy())
            all_masks_pred.extend(anomaly_map.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Recon': f'{recon_losses.avg:.4f}',
                'Seg': f'{seg_losses.avg:.4f}'
            })
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_masks_true = np.array(all_masks_true)
    all_masks_pred = np.array(all_masks_pred)

    # Image-level metrics
    if len(np.unique(all_labels)) > 1:  # Only if both classes are present
        # Use anomaly scores for threshold-based prediction
        threshold = np.percentile(all_scores, 95)  # Use 95th percentile as threshold
        predictions = (all_scores > threshold).astype(int)
        image_metrics = calculate_metrics(all_labels, predictions, all_scores)
    else:
        # All samples are the same class (likely all normal in validation)
        predictions = np.zeros_like(all_labels)  # Predict all as normal
        image_metrics = {
            'accuracy': 1.0 if all_labels[0] == 0 else 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 1.0 if all_labels[0] == 0 else 0.0,
            'f1_score': 0.0,
            'auroc': 0.0,
            'auprc': 0.0
        }
    
    # Pixel-level metrics (for anomalous images only)
    anomaly_indices = all_labels == 1
    if np.sum(anomaly_indices) > 0:
        pixel_metrics = {}
        for threshold in [0.3, 0.5, 0.7]:
            pred_masks = (all_masks_pred[anomaly_indices] > threshold).astype(np.uint8)
            true_masks = (all_masks_true[anomaly_indices] > 0.5).astype(np.uint8)
            
            # Flatten and calculate metrics
            true_flat = true_masks.flatten()
            pred_flat = pred_masks.flatten()
            
            if len(np.unique(true_flat)) > 1:  # Only if there are positive pixels
                metrics = calculate_metrics(true_flat, pred_flat)
                pixel_metrics[f'pixel_f1_@{threshold}'] = metrics['f1_score']
    else:
        pixel_metrics = {}
    
    return {
        'total_loss': losses.avg,
        'recon_loss': recon_losses.avg,
        'seg_loss': seg_losses.avg,
        'image_metrics': image_metrics,
        'pixel_metrics': pixel_metrics,
        'predictions': {
            'labels': all_labels,
            'scores': all_scores,
            'masks_true': all_masks_true,
            'masks_pred': all_masks_pred
        }
    }


def get_optimizer(model, optimizer_name='adam', learning_rate=1e-3, weight_decay=1e-4):
    """Get optimizer for training."""
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='cosine', num_epochs=100, eta_min=1e-6):
    """Get learning rate scheduler."""
    if scheduler_name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    elif scheduler_name.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
    elif scheduler_name.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    else:
        return None


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size, channels, height, width = 4, 3, 256, 256
    reconstruction = torch.randn(batch_size, channels, height, width).to(device)
    anomaly_map = torch.sigmoid(torch.randn(batch_size, 1, height, width)).to(device)
    original_image = torch.randn(batch_size, channels, height, width).to(device)
    true_mask = torch.randint(0, 2, (batch_size, 1, height, width)).float().to(device)
    
    # Test combined loss
    criterion = CombinedLoss().to(device)
    loss_dict = criterion(reconstruction, anomaly_map, original_image, true_mask)
    
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"Segmentation loss: {loss_dict['seg_loss'].item():.4f}")
    
    # Test SSIM loss
    ssim_loss = SSIMLoss().to(device)
    ssim_value = ssim_loss(reconstruction, original_image)
    print(f"SSIM loss: {ssim_value.item():.4f}")
    
    print("Loss functions test completed!")

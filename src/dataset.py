import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset
    
    Args:
        root_dir: Path to the MVTec dataset root directory
        category: Object category (e.g., 'bottle', 'cable', etc.)
        split: 'train' or 'test'
        transform: Image transformations
        target_transform: Target transformations
        is_train: If True, only loads normal images. If False, loads both normal and anomalous images
    """
    
    def __init__(self, root_dir, category, split='train', transform=None, target_transform=None, is_train=True):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # 0 for normal, 1 for anomalous
        self.anomaly_types = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        category_dir = os.path.join(self.root_dir, self.category)
        
        if self.split == 'train':
            # Training only uses normal images
            good_dir = os.path.join(category_dir, 'train', 'good')
            if os.path.exists(good_dir):
                good_images = glob.glob(os.path.join(good_dir, '*.png'))
                self.image_paths.extend(good_images)
                self.labels.extend([0] * len(good_images))
                self.mask_paths.extend([None] * len(good_images))
                self.anomaly_types.extend(['good'] * len(good_images))
        
        elif self.split == 'test':
            test_dir = os.path.join(category_dir, 'test')
            ground_truth_dir = os.path.join(category_dir, 'ground_truth')
            
            # Load normal test images
            good_dir = os.path.join(test_dir, 'good')
            if os.path.exists(good_dir):
                good_images = glob.glob(os.path.join(good_dir, '*.png'))
                self.image_paths.extend(good_images)
                self.labels.extend([0] * len(good_images))
                self.mask_paths.extend([None] * len(good_images))
                self.anomaly_types.extend(['good'] * len(good_images))
            
            # Load anomalous test images if not training mode
            if not self.is_train:
                for anomaly_type in os.listdir(test_dir):
                    if anomaly_type == 'good':
                        continue
                    
                    anomaly_dir = os.path.join(test_dir, anomaly_type)
                    if os.path.isdir(anomaly_dir):
                        anomaly_images = glob.glob(os.path.join(anomaly_dir, '*.png'))
                        self.image_paths.extend(anomaly_images)
                        self.labels.extend([1] * len(anomaly_images))
                        self.anomaly_types.extend([anomaly_type] * len(anomaly_images))
                        
                        # Load corresponding masks
                        mask_dir = os.path.join(ground_truth_dir, anomaly_type)
                        for img_path in anomaly_images:
                            img_name = os.path.basename(img_path)
                            mask_name = img_name.replace('.png', '_mask.png')
                            mask_path = os.path.join(mask_dir, mask_name)
                            if os.path.exists(mask_path):
                                self.mask_paths.append(mask_path)
                            else:
                                self.mask_paths.append(None)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load mask if available
        mask_path = self.mask_paths[idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            # Convert mask to binary (0 or 1)
            mask = np.array(mask)
            mask = (mask > 0).astype(np.uint8)
            mask = Image.fromarray(mask)
        else:
            # Create empty mask for normal images
            mask = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Default mask transform
            mask = transforms.ToTensor()(mask)
        
        label = self.labels[idx]
        anomaly_type = self.anomaly_types[idx]
        
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'anomaly_type': anomaly_type,
            'image_path': image_path
        }


def get_transforms(image_size=256, is_train=True):
    """Get image and mask transforms"""
    
    if is_train:
        image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform


def get_dataloaders(root_dir, category, batch_size=16, image_size=256, num_workers=4):
    """Get train and test dataloaders for a specific category"""
    
    # Get transforms
    train_img_transform, train_mask_transform = get_transforms(image_size, is_train=True)
    test_img_transform, test_mask_transform = get_transforms(image_size, is_train=False)
    
    # Create datasets
    train_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        transform=train_img_transform,
        target_transform=train_mask_transform,
        is_train=True
    )
    
    test_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='test',
        transform=test_img_transform,
        target_transform=test_mask_transform,
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_available_categories(root_dir):
    """Get list of available categories in the dataset"""
    categories = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if it has the expected structure
            train_dir = os.path.join(item_path, 'train')
            test_dir = os.path.join(item_path, 'test')
            if os.path.exists(train_dir) and os.path.exists(test_dir):
                categories.append(item)
    return sorted(categories)


if __name__ == "__main__":
    # Test the dataset
    root_dir = "../datasets/mvtec_anomaly_detection"
    
    # Get available categories
    categories = get_available_categories(root_dir)
    print(f"Available categories: {categories}")
    
    if categories:
        category = categories[0]  # Use first category for testing
        print(f"\nTesting with category: {category}")
        
        # Create datasets
        train_loader, test_loader = get_dataloaders(root_dir, category, batch_size=4)
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        
        # Test loading a batch
        for batch in train_loader:
            print(f"Train batch - Image shape: {batch['image'].shape}")
            print(f"Train batch - Mask shape: {batch['mask'].shape}")
            print(f"Train batch - Labels: {batch['label']}")
            print(f"Train batch - Anomaly types: {batch['anomaly_type']}")
            break
        
        for batch in test_loader:
            print(f"Test batch - Image shape: {batch['image'].shape}")
            print(f"Test batch - Mask shape: {batch['mask'].shape}")
            print(f"Test batch - Labels: {batch['label']}")
            print(f"Test batch - Anomaly types: {batch['anomaly_type']}")
            break

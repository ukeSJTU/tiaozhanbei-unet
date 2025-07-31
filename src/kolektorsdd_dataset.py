import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class KolektorSDDDataset(Dataset):
    """
    KolektorSDD Dataset for surface defect detection
    
    Args:
        root_dir: Path to the KolektorSDD dataset root directory
        split: 'train', 'val', or 'test'
        transform: Image transformations
        target_transform: Target transformations
        image_size: Target image size (height, width)
        train_split: Fraction of data to use for training (default: 0.7)
        val_split: Fraction of data to use for validation (default: 0.15)
    """
    
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, 
                 image_size=(1024, 512), train_split=0.7, val_split=0.15):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        
        # KolektorSDD has 3 classes: background (0), defect_type_1 (1), defect_type_2 (2)
        self.class_names = ['background', 'defect_type_1', 'defect_type_2']
        self.num_classes = 3
        
        self.image_paths = []
        self.mask_paths = []
        
        self._load_dataset_with_splits(train_split, val_split)
        
        print(f"Found {len(self.image_paths)} samples in {split} split")
        print(f"Classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
    
    def _load_dataset_with_splits(self, train_split, val_split):
        """Load dataset and create train/val/test splits"""
        # Collect all image and mask pairs
        all_samples = []
        
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset root directory not found: {self.root_dir}")
        
        # Iterate through all kos folders
        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder_name)
            
            if os.path.isdir(folder_path) and folder_name.startswith('kos'):
                # Find all image files in this folder
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.jpg'):
                        img_path = os.path.join(folder_path, file_name)
                        
                        # Corresponding mask file
                        mask_name = file_name.replace('.jpg', '_label.bmp')
                        mask_path = os.path.join(folder_path, mask_name)
                        
                        if os.path.exists(mask_path):
                            all_samples.append((img_path, mask_path))
        
        # Sort for consistent splits across runs
        all_samples.sort()
        
        # Calculate split indices
        total_samples = len(all_samples)
        train_end = int(total_samples * train_split)
        val_end = int(total_samples * (train_split + val_split))
        
        # Set random seed for reproducible splits
        random.seed(42)
        random.shuffle(all_samples)
        
        # Split the data
        if self.split == 'train':
            selected_samples = all_samples[:train_end]
        elif self.split == 'val':
            selected_samples = all_samples[train_end:val_end]
        elif self.split == 'test':
            selected_samples = all_samples[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
        
        # Extract paths
        for img_path, mask_path in selected_samples:
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask_array = np.array(mask)
        
        # Ensure mask values are in expected range [0, 1, 2]
        # Some masks might have values > 2, clamp them to 2
        mask_array = np.clip(mask_array, 0, 2)
        mask = Image.fromarray(mask_array, mode='L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            # Note: target_transform should handle resizing to (width, height) for PIL
            mask = self.target_transform(mask)
        else:
            # Default mask transform: resize to (width, height) and convert to tensor
            mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask, img_path


class MaskToTensor:
    """Convert PIL mask to tensor - picklable alternative to Lambda"""
    
    def __call__(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def get_kolektorsdd_transforms(image_size=(1024, 512), is_train=True):
    """Get transforms for KolektorSDD dataset"""
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),  # Smaller rotation for industrial images
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Target transform should match the image transform exactly
    target_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        MaskToTensor()
    ])
    
    return transform, target_transform


def get_kolektorsdd_dataloaders(root_dir, batch_size=16, image_size=(1024, 512), num_workers=4,
                               train_split=0.7, val_split=0.15):
    """Create data loaders for KolektorSDD dataset"""
    
    # Get transforms
    train_transform, train_target_transform = get_kolektorsdd_transforms(image_size, is_train=True)
    val_transform, val_target_transform = get_kolektorsdd_transforms(image_size, is_train=False)
    
    # Create datasets
    train_dataset = KolektorSDDDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        target_transform=train_target_transform,
        image_size=image_size,
        train_split=train_split,
        val_split=val_split
    )
    
    val_dataset = KolektorSDDDataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform,
        target_transform=val_target_transform,
        image_size=image_size,
        train_split=train_split,
        val_split=val_split
    )
    
    test_dataset = KolektorSDDDataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform,
        target_transform=val_target_transform,
        image_size=image_size,
        train_split=train_split,
        val_split=val_split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes


if __name__ == "__main__":
    # Test the dataset
    root_dir = "datasets/KolektorSDD"
    
    if os.path.exists(root_dir):
        train_loader, val_loader, test_loader, num_classes = get_kolektorsdd_dataloaders(
            root_dir=root_dir,
            batch_size=4,
            image_size=(1024, 512),
            num_workers=0
        )
        
        print(f"Number of classes: {num_classes}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test loading one batch
        for images, masks, paths in train_loader:
            print(f"Image batch shape: {images.shape}")
            print(f"Mask batch shape: {masks.shape}")
            print(f"Mask unique values: {torch.unique(masks)}")
            print(f"Sample paths: {paths[0]}")
            break
    else:
        print(f"Dataset not found at: {root_dir}")
import os
import json
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class GearDataset(Dataset):
    """
    Gear Dataset with LabelMe annotation support
    
    Args:
        root_dir: Path to the Gear dataset root directory
        split: 'train', 'val', or 'test'
        transform: Image transformations
        target_transform: Target transformations
        image_size: Target image size (height, width)
    """
    
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, image_size=(512, 512), enable_priority_logging=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.enable_priority_logging = enable_priority_logging
        
        self.image_paths = []
        self.label_paths = []
        self.class_names = set()
        
        # Statistics for priority-based resolution
        self.priority_stats = {
            'files_processed': 0,
            'files_with_overlaps': 0,
            'pixels_resolved': {'spalling_over_pitting': 0, 'spalling_over_scrape': 0, 'pitting_over_scrape': 0}
        }
        
        self._load_dataset()
        # Sort class names in a meaningful order: pitting, spalling, scrape
        class_order = ["pitting", "spalling", "scrape"]
        self.class_names = [name for name in class_order if name in self.class_names]
        
        self.num_classes = len(self.class_names) + 1  # +1 for background
        
        # Map class names to indices (background=0, pitting=1, spalling=2, scrape=3)
        self.class_to_idx = {'background': 0}
        self.class_to_idx['pitting'] = 1    # original class 0 -> new class 1
        self.class_to_idx['spalling'] = 2   # original class 1 -> new class 2
        self.class_to_idx['scrape'] = 3     # original class 2 -> new class 3
        
        print(f"Found {len(self.image_paths)} images in {split} split")
        print(f"Classes: {self.class_names}")
        print(f"Number of classes (including background): {self.num_classes}")
        
        if self.enable_priority_logging and self.priority_stats['files_processed'] > 0:
            print(f"\n🔧 Priority Resolution Stats for {split} split:")
            print(f"   Files with overlaps resolved: {self.priority_stats['files_with_overlaps']}/{self.priority_stats['files_processed']}")
            for conflict, pixels in self.priority_stats['pixels_resolved'].items():
                if pixels > 0:
                    print(f"   {conflict.replace('_', ' ')}: {pixels:,} pixels resolved")
    
    def _load_dataset(self):
        images_dir = os.path.join(self.root_dir, 'images', self.split)
        labels_dir = os.path.join(self.root_dir, 'labels', self.split)
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        if not os.path.exists(labels_dir):
            raise ValueError(f"Labels directory not found: {labels_dir}")
        
        # Get all image files
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_file)
                
                # Corresponding label file
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                
                if os.path.exists(label_path):
                    self.image_paths.append(img_path)
                    self.label_paths.append(label_path)
                    
                    # Parse label file to get class names
                    self._parse_label_file(label_path)
    
    def _parse_label_file(self, label_path):
        """Parse LabelMe format txt file to extract class names"""
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:  # class_id + at least 4 coordinates
                            class_id = parts[0]
                            if class_id.isdigit():
                                class_id = int(class_id)
                                # Map class IDs to defect names
                                if class_id == 0:
                                    self.class_names.add("pitting")
                                elif class_id == 1:
                                    self.class_names.add("spalling")
                                elif class_id == 2:
                                    self.class_names.add("scrape")
        except Exception as e:
            print(f"Warning: Could not parse label file {label_path}: {e}")
    
    def _create_mask_from_labelme(self, label_path, img_width, img_height):
        """
        Convert LabelMe txt format to segmentation mask with priority-based overlap resolution.
        Priority order: spalling > pitting > scrape
        When regions overlap, the higher priority class takes precedence.
        """
        # Create separate masks for each class to handle overlaps
        class_masks = {}
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:  # class_id + at least 4 coordinates (2 points minimum)
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
                            
                            # Create polygon mask for this specific class
                            if len(pixel_coords) >= 3:  # Need at least 3 points for a polygon
                                if class_id not in class_masks:
                                    class_masks[class_id] = np.zeros((img_height, img_width), dtype=np.uint8)
                                
                                # Create temporary mask for this polygon
                                temp_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                                img_pil = Image.fromarray(temp_mask)
                                draw = ImageDraw.Draw(img_pil)
                                draw.polygon(pixel_coords, fill=1)
                                temp_mask = np.array(img_pil)
                                
                                # Add this polygon to the class mask
                                class_masks[class_id] = np.logical_or(class_masks[class_id], temp_mask).astype(np.uint8)
        
        except Exception as e:
            print(f"Warning: Could not create mask from {label_path}: {e}")
            return np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Create final mask with priority-based resolution
        final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Track statistics if enabled
        self.priority_stats['files_processed'] += 1
        has_overlaps = False
        
        # Define priority order: spalling (1) > pitting (0) > scrape (2)
        # Process in reverse priority order so higher priority classes overwrite lower ones
        class_priority_order = [2, 0, 1]  # scrape, pitting, spalling (lowest to highest priority)
        class_id_to_final_id = {0: 1, 1: 2, 2: 3}  # pitting->1, spalling->2, scrape->3
        class_names_map = {0: 'pitting', 1: 'spalling', 2: 'scrape'}
        
        for class_id in class_priority_order:
            if class_id in class_masks:
                final_id = class_id_to_final_id[class_id]
                current_mask = class_masks[class_id] == 1
                
                # Check for overlaps with existing assignments (for logging)
                if self.enable_priority_logging and np.any(final_mask > 0):
                    overlap_mask = current_mask & (final_mask > 0)
                    if np.any(overlap_mask):
                        has_overlaps = True
                        overlap_pixels = np.sum(overlap_mask)
                        
                        # Count specific conflict types
                        if class_id == 1:  # spalling overriding others
                            if np.any(overlap_mask & (final_mask == 1)):  # spalling over pitting
                                self.priority_stats['pixels_resolved']['spalling_over_pitting'] += np.sum(overlap_mask & (final_mask == 1))
                            if np.any(overlap_mask & (final_mask == 3)):  # spalling over scrape
                                self.priority_stats['pixels_resolved']['spalling_over_scrape'] += np.sum(overlap_mask & (final_mask == 3))
                        elif class_id == 0:  # pitting overriding scrape
                            if np.any(overlap_mask & (final_mask == 3)):  # pitting over scrape
                                self.priority_stats['pixels_resolved']['pitting_over_scrape'] += np.sum(overlap_mask & (final_mask == 3))
                
                # Apply this class mask, overwriting any previous assignments
                final_mask[current_mask] = final_id
        
        if has_overlaps:
            self.priority_stats['files_with_overlaps'] += 1
        
        return final_mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Load label and create mask
        label_path = self.label_paths[idx]
        mask = self._create_mask_from_labelme(label_path, orig_width, orig_height)
        mask = Image.fromarray(mask, mode='L')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Default mask transform: resize and convert to tensor
            mask = mask.resize(self.image_size, Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask, img_path


class MaskToTensor:
    """Convert PIL mask to tensor - picklable alternative to Lambda"""
    
    def __call__(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def get_gear_transforms(image_size=(512, 512), is_train=True):
    """Get transforms for Gear dataset"""
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    target_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        MaskToTensor()
    ])
    
    return transform, target_transform


def get_gear_dataloaders(root_dir, batch_size=16, image_size=(512, 512), num_workers=4, enable_priority_logging=False):
    """Create data loaders for Gear dataset"""
    
    # Get transforms
    train_transform, train_target_transform = get_gear_transforms(image_size, is_train=True)
    val_transform, val_target_transform = get_gear_transforms(image_size, is_train=False)
    
    # Create datasets
    train_dataset = GearDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        target_transform=train_target_transform,
        image_size=image_size,
        enable_priority_logging=enable_priority_logging
    )
    
    val_dataset = GearDataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform,
        target_transform=val_target_transform,
        image_size=image_size,
        enable_priority_logging=enable_priority_logging
    )
    
    test_dataset = GearDataset(
        root_dir=root_dir,
        split='test',
        transform=val_transform,
        target_transform=val_target_transform,
        image_size=image_size,
        enable_priority_logging=enable_priority_logging
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
    root_dir = "datasets/Gear"
    
    train_loader, val_loader, test_loader, num_classes = get_gear_dataloaders(
        root_dir=root_dir,
        batch_size=4,
        image_size=(512, 512),
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
        break
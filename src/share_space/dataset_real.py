import os
from typing import Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader

class SimExpPairedDataset(Dataset):
    """
    Dataset for paired simulation (WikiArt) and experimental (COCO) images.
    
    This creates pseudo-pairs by matching images from two different domains.
    For style transfer tasks, we pair content images (COCO) with style images (WikiArt).
    """
    
    def __init__(
        self,
        exp_dir: str = "coco/train2017",  # COCO images directory
        sim_dir: str = "wikiart/images",   # WikiArt images directory
        img_size: int = 224,
        transform: Optional[T.Compose] = None,
        pairing_strategy: str = "random",  # "random", "ordered", or "fixed"
        seed: int = 42
    ):
        """
        Args:
            exp_dir: Directory containing COCO images (experimental/content domain)
            sim_dir: Directory containing WikiArt images (simulation/style domain)
            img_size: Target image size for resizing
            transform: Optional custom transform
            pairing_strategy: How to pair images across domains
                - "random": Random pairing on each epoch
                - "ordered": Sequential pairing (deterministic)
                - "fixed": Fixed random pairing (using seed)
            seed: Random seed for reproducible pairing
        """
        self.exp_dir = Path(exp_dir)
        self.sim_dir = Path(sim_dir)
        self.img_size = img_size
        self.pairing_strategy = pairing_strategy
        self.seed = seed
        
        # Get all image files from both domains
        self.exp_images = self._get_image_files(self.exp_dir)
        self.sim_images = self._get_image_files(self.sim_dir)
        
        if len(self.exp_images) == 0:
            raise ValueError(f"No images found in experimental directory: {exp_dir}")
        if len(self.sim_images) == 0:
            raise ValueError(f"No images found in simulation directory: {sim_dir}")
        
        print(f"Found {len(self.exp_images)} COCO images")
        print(f"Found {len(self.sim_images)} WikiArt images")
        
        # Create pairing based on strategy
        if pairing_strategy == "fixed":
            random.seed(seed)
            self.sim_indices = random.sample(
                range(len(self.sim_images)), 
                len(self.exp_images)
            ) if len(self.sim_images) >= len(self.exp_images) else \
               [random.randint(0, len(self.sim_images)-1) for _ in range(len(self.exp_images))]
        
        # Default transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
    
    def _get_image_files(self, directory: Path) -> list:
        """Recursively get all image files from directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = []
        
        if directory.is_dir():
            for ext in image_extensions:
                image_files.extend(directory.rglob(f"*{ext}"))
                image_files.extend(directory.rglob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _get_sim_index(self, idx: int) -> int:
        """Get the simulation image index based on pairing strategy."""
        if self.pairing_strategy == "fixed":
            return self.sim_indices[idx]
        elif self.pairing_strategy == "ordered":
            return idx % len(self.sim_images)
        else:  # random
            return random.randint(0, len(self.sim_images) - 1)
    
    def __len__(self) -> int:
        """Dataset length is determined by the experimental (COCO) dataset."""
        return len(self.exp_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a paired sample.
        
        Returns:
            Tuple of (exp_image, sim_image) tensors
            - exp_image: COCO image (content/experimental domain)
            - sim_image: WikiArt image (style/simulation domain)
        """
        # Load experimental (COCO) image
        exp_path = self.exp_images[idx]
        exp_image = Image.open(exp_path).convert('RGB')
        
        # Load simulation (WikiArt) image
        sim_idx = self._get_sim_index(idx)
        sim_path = self.sim_images[sim_idx]
        sim_image = Image.open(sim_path).convert('RGB')
        
        # Apply transforms
        exp_tensor = self.transform(exp_image)
        sim_tensor = self.transform(sim_image)
        
        return exp_tensor, sim_tensor
    
    def get_image_paths(self, idx: int) -> Tuple[str, str]:
        """Get the file paths for a given index (useful for debugging)."""
        sim_idx = self._get_sim_index(idx)
        return str(self.exp_images[idx]), str(self.sim_images[sim_idx])

def get_real_dataloaders(
    exp_dir: str = "exp",
    sim_dir: str = "sim",
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    train_split: float = 0.9,
    seed: int = 42,
    transform: Optional[T.Compose] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for real paired simulation and experimental images.
    
    Args:
        exp_dir: Directory containing experimental images (train_{number}_{sub_image_number}.png)
        sim_dir: Directory containing simulation images (train_{number}_{sub_image_number}_0000.000000.{state|count}.png)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        img_size: Target image size for resizing
        train_split: Fraction of data to use for training (default: 0.9)
        seed: Random seed for reproducible splits
        transform: Optional custom transform (if None, uses default)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = SimExpPairedDataset(
        exp_dir=exp_dir,
        sim_dir=sim_dir,
        img_size=img_size,
        transform=transform
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    dataset = SimExpPairedDataset(
        exp_dir="../../data/coco/val2017",
        sim_dir="../../data/wikiart/train_2",
        img_size=224,
        pairing_strategy="fixed",
        seed=42
    )
    
    # Test loading a sample
    exp_img, sim_img = dataset[0]
    print(f"Experimental (COCO) shape: {exp_img.shape}")
    print(f"Simulation (WikiArt) shape: {sim_img.shape}")
    
    # Get the paths for debugging
    exp_path, sim_path = dataset.get_image_paths(0)
    print(f"Experimental image: {exp_path}")
    print(f"Simulation image: {sim_path}")

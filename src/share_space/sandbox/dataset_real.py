import os
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader

class SimExpPairedDataset(Dataset):
    """
    Dataset for paired simulation (WikiArt) and experimental (COCO) images.
    
    Each content image (COCO) is paired with a randomly selected style image (WikiArt).
    When content images outnumber style images, style images are reused with different
    content images through random sampling.
    """
    
    def __init__(
        self,
        exp_dir: str = "coco/train2017",  # COCO images directory (content)
        sim_dir: str = "wikiart/images",   # WikiArt images directory (style)
        img_size: int = 224,
        transform: Optional[T.Compose] = None,
        seed: Optional[int] = None  # If provided, creates reproducible random pairs
    ):
        """
        Args:
            exp_dir: Directory containing COCO images (experimental/content domain)
            sim_dir: Directory containing WikiArt images (simulation/style domain)
            img_size: Target image size for resizing
            transform: Optional custom transform
            seed: Optional random seed for reproducible pairing (if None, truly random each time)
        """
        self.exp_dir = Path(exp_dir)
        self.sim_dir = Path(sim_dir)
        self.img_size = img_size
        self.seed = seed
        
        # Get all image files from both domains
        self.exp_images = self._get_image_files(self.exp_dir)
        self.sim_images = self._get_image_files(self.sim_dir)
        
        if len(self.exp_images) == 0:
            raise ValueError(f"No images found in experimental directory: {exp_dir}")
        if len(self.sim_images) == 0:
            raise ValueError(f"No images found in simulation directory: {sim_dir}")
        
        print(f"Found {len(self.exp_images)} COCO images (content)")
        print(f"Found {len(self.sim_images)} WikiArt images (style)")
        
        # Create random pairing: each content image gets a random style image
        if seed is not None:
            # Fixed random pairing for reproducibility
            rng = np.random.RandomState(seed)
            self.style_indices = rng.randint(0, len(self.sim_images), size=len(self.exp_images))
            print(f"Created fixed random pairing with seed={seed}")
        else:
            # Will generate random pairs on-the-fly each epoch
            self.style_indices = None
            print("Using dynamic random pairing (changes each epoch)")
        
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
        
        return sorted(image_files)[:100]
    
    def __len__(self) -> int:
        """Dataset length is determined by the content (COCO) dataset."""
        return len(self.exp_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a paired sample.
        
        Returns:
            Tuple of (exp_image, sim_image) tensors
            - exp_image: COCO image (content/experimental domain)
            - sim_image: WikiArt image (style/simulation domain), randomly selected
        """
        # Load experimental (COCO) image
        exp_path = self.exp_images[idx]
        exp_image = Image.open(exp_path).convert('RGB')
        
        # Get style image index
        if self.style_indices is not None:
            # Use pre-computed fixed random pairing
            sim_idx = int(self.style_indices[idx])
        else:
            # Generate random pairing on-the-fly
            sim_idx = random.randint(0, len(self.sim_images) - 1)
        
        # Load simulation (WikiArt) image
        sim_path = self.sim_images[sim_idx]
        sim_image = Image.open(sim_path).convert('RGB')
        
        # Apply transforms
        exp_tensor = self.transform(exp_image)
        sim_tensor = self.transform(sim_image)
        
        return exp_tensor, sim_tensor
    
    def get_image_paths(self, idx: int) -> Tuple[str, str]:
        """Get the file paths for a given index (useful for debugging)."""
        if self.style_indices is not None:
            sim_idx = int(self.style_indices[idx])
        else:
            # For dynamic pairing, just show an example (will be different when actually loaded)
            sim_idx = random.randint(0, len(self.sim_images) - 1)
        
        return str(self.exp_images[idx]), str(self.sim_images[sim_idx])
    
    def reshuffle_styles(self, seed: Optional[int] = None):
        """
        Reshuffle the style pairing (useful between epochs for variety).
        Only works if the dataset was initialized with a seed.
        """
        if seed is None:
            seed = self.seed if self.seed is not None else random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(seed)
        self.style_indices = rng.randint(0, len(self.sim_images), size=len(self.exp_images))
        print(f"Reshuffled style pairing with seed={seed}")



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

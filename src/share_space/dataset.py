# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import re
import matplotlib.pyplot as plt

class DummySimExpDataset(Dataset):
    """Dummy dataset for simulation-experimental image pairs"""
    def __init__(self, num_samples: int = 1000, img_size: int = 224, 
                 transform: Optional[T.Compose] = None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        Generate dummy simulation and experimental image pairs
        Simulation: cleaner, more structured
        Experimental: noisier, more realistic artifacts
        """
        # Simulation image (geometric patterns)
        sim_img = self._generate_simulation_image()
        
        # Experimental image (simulation + noise + artifacts)
        exp_img = self._generate_experimental_image(sim_img)
        
        # Convert to PIL Image first (ensures proper dtype handling)
        from PIL import Image
        sim_img_pil = Image.fromarray((sim_img * 255).astype(np.uint8))
        exp_img_pil = Image.fromarray((exp_img * 255).astype(np.uint8))
        
        # Apply transforms
        sim_img = self.transform(sim_img_pil)
        exp_img = self.transform(exp_img_pil)
        
        return {
            'simulation': sim_img.float(),  # Ensure float32
            'experimental': exp_img.float(),  # Ensure float32
            'idx': idx
        }
    
    def _generate_simulation_image(self) -> np.ndarray:
        """Generate synthetic simulation image"""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)  # Changed to float32
        
        # Add geometric patterns
        num_circles = np.random.randint(3, 8)
        for _ in range(num_circles):
            center = (np.random.randint(0, self.img_size), np.random.randint(0, self.img_size))
            radius = np.random.randint(10, 40)
            color = np.random.rand(3).astype(np.float32)  # Ensure float32
            
            y, x = np.ogrid[:self.img_size, :self.img_size]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask] = color
        
        return np.clip(img, 0, 1).astype(np.float32)  # Ensure float32
    
    def _generate_experimental_image(self, sim_img: np.ndarray) -> np.ndarray:
        """Add noise and artifacts to simulation image to mimic experimental data"""
        exp_img = sim_img.copy().astype(np.float32)  # Ensure float32
        # Add Gaussian noise
        noise = np.random.normal(0, 0.05, exp_img.shape).astype(np.float32)  # Ensure float32
        exp_img = exp_img + noise
        
        # Add random brightness variation
        brightness_factor = np.random.uniform(0.8, 1.2)
        exp_img = exp_img * np.float32(brightness_factor)  # Ensure float32
        
        # Add some random artifacts (spots)
        num_artifacts = np.random.randint(5, 15)
        for _ in range(num_artifacts):
            x, y = np.random.randint(0, self.img_size, 2)
            size = np.random.randint(2, 6)
            artifact = (np.random.rand(3) * 0.3).astype(np.float32)  # Ensure float32
            
            x_start, x_end = max(0, x-size), min(self.img_size, x+size)
            y_start, y_end = max(0, y-size), min(self.img_size, y+size)
            exp_img[y_start:y_end, x_start:x_end] += artifact
        
        return np.clip(exp_img, 0, 1).astype(np.float32)  # Ensure float32


def get_dummy_dataloaders(batch_size: int = 32, num_workers: int = 4, img_size: int = 224,
                   train_samples: int = 500, val_samples: int = 50) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders"""
    
    train_dataset = DummySimExpDataset(num_samples=train_samples, img_size=img_size)
    val_dataset = DummySimExpDataset(num_samples=val_samples, img_size=img_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

class SimExpPairedDataset(Dataset):
    """Dataset for paired simulation and experimental images"""
    
    # Class-level color mapping (hex codes)
    STATE_COLOR_MAP = {
        'OTHER': "#FFFF00",                    # Yellow
        'INFLAMMATORY': "#FF00FF",             # Pink/Magenta
        'HEALTHY_EPITHELIAL': "#00FF00",       # Green
        'DYSPLASTIC/MALIGNANT': "#FF0000",     # Red
        'FIBROBLAST': "#0000FF",               # Blue
        'MUSCLE': "#00FFFF",                   # Cyan
        'ENDOTHELIAL': "#F49E42"               # Orange
    }
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Map RGB to state index [1-7]
    RGB_TO_STATE = {
        hex_to_rgb("#FFFF00"): 1,      # OTHER
        hex_to_rgb("#FF00FF"): 2,      # INFLAMMATORY
        hex_to_rgb("#00FF00"): 3,      # HEALTHY_EPITHELIAL
        hex_to_rgb("#FF0000"): 4,      # DYSPLASTIC/MALIGNANT
        hex_to_rgb("#0000FF"): 5,      # FIBROBLAST
        hex_to_rgb("#00FFFF"): 6,      # MUSCLE
        hex_to_rgb("#F49E42"): 7       # ENDOTHELIAL
    }
    
    def __init__(self, 
                 exp_dir: str = "exp",
                 sim_dir: str = "sim", 
                 img_size: int = 224,
                 transform: Optional[T.Compose] = None,
                 debugging: bool = False):
        """
        Args:
            exp_dir: Directory containing experimental images
            sim_dir: Directory containing simulation images (with .state.png and .count.png)
            img_size: Target image size for resizing
            transform: Optional custom transform
            debugging: Whether to print debugging information
        """
        self.exp_dir = Path(exp_dir)
        self.sim_dir = Path(sim_dir)
        self.img_size = img_size
        self.debugging = debugging
        self.exp_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        self.count_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        # State channel: convert to one-hot encoding
        self.state_transform = T.Compose([
            T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])
        # Find all paired images
        self.pairs = self._find_paired_images()
        
        if len(self.pairs) == 0:
            raise ValueError(f"No paired images found in {exp_dir} and {sim_dir}")
        
        print(f"Found {len(self.pairs)} paired images")
    
    def _find_paired_images(self) -> List[Tuple[Path, Path, Path]]:
        """
        Find all paired simulation and experimental images.
        
        Returns:
            List of tuples: (exp_image_path, sim_state_path, sim_count_path)
        """
        pairs = []
        
        # Get all experimental images
        exp_images = sorted(self.exp_dir.glob("train_*.png"))
        for exp_path in exp_images[:100]:
            # Extract the base name: train_{number}_{sub_image_number}
            match = re.match(r'train_(\d+)_(\d+)\.png', exp_path.name)
            if not match:
                continue
            number, sub_number = match.groups()
            # Construct corresponding simulation file names
            sim_base = f"train_{number}_{int(sub_number)}_0000.000000.population"
            sim_state_path = self.sim_dir / f"{sim_base}.state.png"
            sim_count_path = self.sim_dir / f"{sim_base}.count.png"
            
            # Check if both simulation channels exist
            if sim_state_path.exists() and sim_count_path.exists():
                pairs.append((exp_path, sim_state_path, sim_count_path))
            else:
                print(f"Warning: Missing simulation files for {exp_path.name}")
        
        return pairs
    
    def _rgb_to_state_index(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB state image to state indices [1-7].
        
        Args:
            rgb_image: (H, W, 3) numpy array with RGB values [0-255]
        
        Returns:
            (H, W) numpy array with state indices [1-7]
        """
        H, W = rgb_image.shape[:2]
        state_indices = np.zeros((H, W), dtype=np.int64)
        
        # Map each color to its state index
        for rgb_tuple, state_idx in self.RGB_TO_STATE.items():
            # Create mask for pixels matching this color
            # Use small tolerance for potential compression artifacts
            color = np.array(rgb_tuple, dtype=np.uint8)
            diff = np.abs(rgb_image.astype(np.int16) - color.astype(np.int16))
            matches = np.all(diff <= 10, axis=2)  # Tolerance of 10 for each channel
            state_indices[matches] = state_idx
        
        # Check for unmapped pixels (shouldn't happen with clean data)
        unmapped = (state_indices == 0)
        if self.debugging:
            if np.any(unmapped):
                # print the number of unmapped pixels
                print(f"Warning: {np.sum(unmapped)} pixels could not be mapped to any state")
        
        return state_indices
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        exp_path, sim_state_path, sim_count_path = self.pairs[idx]
        
        # Load experimental image (RGB)
        exp_img = Image.open(exp_path).convert('RGB')
        exp_tensor = self.exp_transform(exp_img)  # (3, H, W), normalized to [-1, 1]
        
        # Load simulation count (grayscale, continuous [0, 1])
        sim_count = Image.open(sim_count_path).convert('L')
        sim_count_tensor = self.count_transform(sim_count)  # (1, H, W), normalized to [-1, 1]
        
        # Load simulation state (RGB, categorical)
        sim_state_rgb = Image.open(sim_state_path).convert('RGB')
        # Use NEAREST interpolation to preserve discrete colors
        sim_state_rgb = sim_state_rgb.resize(
            (self.img_size, self.img_size), 
            Image.Resampling.NEAREST
        )
        sim_state_np = np.array(sim_state_rgb)  # (H, W, 3)
        
        # Map RGB colors to state indices [1-7]
        sim_state_indices = self._rgb_to_state_index(sim_state_np)  # (H, W)
        sim_state_tensor = torch.from_numpy(sim_state_indices).long()  # (H, W)
        
        # One-hot encode: 7 states [1-7]
        sim_state_onehot = torch.zeros(7, self.img_size, self.img_size)
        for state_idx in range(1, 8):
            sim_state_onehot[state_idx-1] = (sim_state_tensor == state_idx).float()
        
        # Combine: (1, H, W) + (7, H, W) = (8, H, W)
        sim_tensor = torch.cat([sim_count_tensor, sim_state_onehot], dim=0)
        
        return {
            'experimental': exp_tensor,      # (3, H, W), RGB normalized to [-1, 1]
            'simulation': sim_tensor,        # (8, H, W), count + 7 one-hot state channels
            'exp_path': str(exp_path),
            'sim_state_path': str(sim_state_path),
            'sim_count_path': str(sim_count_path)
        }

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


# Example usage:
if __name__ == "__main__":
    train_loader, val_loader = get_real_dataloaders(
        exp_dir="../../data/exp",
        sim_dir="../../data/sim",
        batch_size=32,
        num_workers=4,
        img_size=224,
        train_split=0.9
    )
    state_names = {
        1: 'OTHER',
        2: 'INFLAMMATORY',
        3: 'HEALTHY_EPITHELIAL',
        4: 'DYSPLASTIC/MALIGNANT',
        5: 'FIBROBLAST',
        6: 'MUSCLE',
        7: 'ENDOTHELIAL'
    }
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"Experimental batch shape: {batch['experimental'].shape}")  # [32, 3, 224, 224]
    print(f"Simulation batch shape: {batch['simulation'].shape}")      # [32, 2, 224, 224]
    n_viz = 3
    # Viz the {n_viz} samples from the first batch
    first_batch = next(iter(train_loader))
    fig, ax = plt.subplots(n_viz, 9, figsize=(2 * 8, 2 * n_viz))
    for i in range(n_viz):
        exp_img = first_batch['experimental'][i].permute(1, 2, 0).cpu().numpy()
        sim_img = first_batch['simulation'][i].permute(1, 2, 0).cpu().numpy()
        ax[i, 0].imshow(exp_img)
        for j in range(8):
            ax[i, j+1].imshow(sim_img[..., j])
            ax[i, j+1].axis('off')
            if i == 0 and j == 0:
                ax[i, j+1].set_title(f'Cell Count', loc='center')
            elif i == 0:
                ax[i, j+1].set_title(f'{state_names[j]}', loc='center')
        if i == 0:
            ax[i, 0].set_title('Experimental\n(Target)', loc='center')
        ax[i, 0].axis('off')
    plt.tight_layout()
    plt.show()
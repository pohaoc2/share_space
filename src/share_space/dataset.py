# data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional
import torchvision.transforms as T

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


def get_dataloaders(batch_size: int = 32, num_workers: int = 4, img_size: int = 224,
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
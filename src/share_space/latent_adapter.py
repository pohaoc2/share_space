import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
from torch.utils.data import DataLoader

class LatentDomainAdapter(nn.Module):
    """
    Fully connected network that maps simulation latent features to experimental domain.
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        latent_dim: int,
        n_patches: int,
        hidden_dims: Optional[list] = None
    ):
        """
        Args:
            feature_extractor: Frozen feature extractor (e.g., pretrained encoder)
            latent_dim: Dimension of the latent representation
            n_patches: Number of patches in the latent representation
            hidden_dims: List of hidden layer dimensions. If None, uses [latent_dim*2, latent_dim]
        """
        super().__init__()
        
        # Freeze feature extractor
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        
        # Build FCNN
        if hidden_dims is None:
            hidden_dims = [latent_dim * 2, latent_dim]
        
        layers = []
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(n_patches)
            ])
            input_dim = hidden_dim
        
        # Output layer (no activation to keep it in latent space)
        layers.append(nn.Linear(input_dim, latent_dim))
        
        self.fcnn = nn.Sequential(*layers)
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract frozen features from images."""
        with torch.no_grad():
            _, features = self.feature_extractor(images)
        return features
    
    def forward(self, sim_images: torch.Tensor, exp_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sim_images: Simulation images
            exp_images: Experimental images
            
        Returns:
            Tuple of (adapted_sim_latent, exp_latent)
        """
        # Extract frozen features
        sim_latent = self.extract_features(sim_images)
        exp_latent = self.extract_features(exp_images)
        # Map simulation latent to experimental domain
        adapted_sim_latent = self.fcnn(sim_latent)
        return adapted_sim_latent, exp_latent


def train_step(
    model: LatentDomainAdapter,
    sim_images: torch.Tensor,
    exp_images: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> float:
    """
    Single training step.
    
    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    adapted_sim_latent, exp_latent = model(sim_images, exp_images)
    
    # Calculate loss
    loss = criterion(adapted_sim_latent, exp_latent)
    
    # Backward pass and update
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_epoch(
    model: LatentDomainAdapter,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module = nn.MSELoss(),
    device: str = 'cuda'
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.to(device)
    total_loss = 0.0
    
    for batch in dataloader:
        sim_images = batch['simulation'].to(device)
        exp_images = batch['experimental'].to(device)
        
        loss = train_step(model, sim_images, exp_images, optimizer, criterion)
        total_loss += loss
    
    return total_loss / len(dataloader)

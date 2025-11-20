import torch
import torch.nn as nn

class AdaINFusion(nn.Module):
    """
    Adaptive Instance Normalization for fusing content and style features
    Implements Equation 5 from the paper
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, content_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN to fuse content and style features
        
        Args:
            content_features: Features from content image (B, C, H, W) or (B, N, C)
            style_features: Features from style image (B, C, H, W) or (B, N, C)
        
        Returns:
            Fused features: F_adapted = γ(F_c) · σ(F_s) + μ(F_s)
        """
        # Ensure features are in (B, C, ...) format
        if content_features.dim() == 3:  # (B, N, C) -> (B, C, N)
            content_features = content_features.transpose(1, 2)
            style_features = style_features.transpose(1, 2)
            transposed = True
        else:
            transposed = False
        
        # Calculate statistics over spatial dimensions
        # For (B, C, H, W), calculate over (H, W)
        # For (B, C, N), calculate over (N,)
        dims = list(range(2, content_features.dim()))
        
        # Content normalization: γ(F_c)
        content_mean = content_features.mean(dim=dims, keepdim=True)
        content_std = content_features.std(dim=dims, keepdim=True) + 1e-5
        content_normalized = (content_features - content_mean) / content_std
        
        # Style statistics: μ(F_s) and σ(F_s)
        style_mean = style_features.mean(dim=dims, keepdim=True)
        style_std = style_features.std(dim=dims, keepdim=True) + 1e-5
        
        # Apply AdaIN: F_adapted = γ(F_c) · σ(F_s) + μ(F_s)
        fused_features = content_normalized * style_std + style_mean
        
        # Restore original shape if needed
        if transposed:
            fused_features = fused_features.transpose(1, 2)
        
        return fused_features


class HistoAdaIN(nn.Module):
    """
    AdaIN with learnable modulation for sim2real histopathology
    Supports both 3D (B, N, D) and 4D (B, C, H, W) feature tensors
    """
    def __init__(self, num_features, style_dim=None):
        super().__init__()
        if style_dim is None:
            style_dim = num_features
        
        self.num_features = num_features
        self.style_dim = style_dim
        
        # Optional: learn to refine style statistics
        # For 4D tensors: use Conv2d
        self.style_encoder_2d = nn.Sequential(
            nn.Conv2d(style_dim, style_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        # For 3D tensors: use Linear
        self.style_encoder_1d = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU()
        )
        
        # Learnable modulation strength - will be reshaped based on input format
        self.gamma_scale = nn.Parameter(torch.ones(1, num_features))
        self.beta_scale = nn.Parameter(torch.zeros(1, num_features))
        
    def forward(self, content_features, style_features):
        # Normalize content (preserve structure)
        print(f"content_features shape: {content_features.shape}")
        print(f"style_features shape: {style_features.shape}")
        
        # Handle both 3D (B, N, D) and 4D (B, C, H, W) tensors
        if content_features.dim() == 3:  # (B, N, D) - patch tokens
            # Compute statistics over sequence dimension (dim=1)
            content_mean = content_features.mean(dim=1, keepdim=True)  # (B, 1, D)
            content_std = content_features.std(dim=1, keepdim=True, unbiased=False) + 1e-5
            content_normalized = (content_features - content_mean) / content_std
            
            # Extract style statistics
            style_encoded = self.style_encoder_1d(style_features)  # (B, N, D)
            style_mean = style_encoded.mean(dim=1, keepdim=True)  # (B, 1, D)
            style_std = style_encoded.std(dim=1, keepdim=True, unbiased=False) + 1e-5
            
            # Apply with learned modulation
            # Reshape gamma_scale and beta_scale to (1, 1, D)
            gamma = style_std * self.gamma_scale.unsqueeze(1)  # (B, 1, D)
            beta = style_mean + self.beta_scale.unsqueeze(1)  # (B, 1, D)
            
            return content_normalized * gamma + beta
            
        else:  # 4D (B, C, H, W) - feature maps
            # Compute statistics over spatial dimensions (dim=[2, 3])
            content_mean = content_features.mean(dim=[2, 3], keepdim=True)
            content_std = content_features.std(dim=[2, 3], keepdim=True, unbiased=False) + 1e-5
            content_normalized = (content_features - content_mean) / content_std
            
            # Extract style statistics
            style_encoded = self.style_encoder_2d(style_features)
            style_mean = style_encoded.mean(dim=[2, 3], keepdim=True)
            style_std = style_encoded.std(dim=[2, 3], keepdim=True, unbiased=False) + 1e-5
            
            # Apply with learned modulation
            # Reshape gamma_scale and beta_scale to (1, C, 1, 1)
            gamma = style_std * self.gamma_scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = style_mean + self.beta_scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            
            return content_normalized * gamma + beta
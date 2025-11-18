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

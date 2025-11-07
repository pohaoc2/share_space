# models/decoders.py
import torch
import torch.nn as nn
from typing import Optional
from typing import List

class ConvDecoder(nn.Module):
    """Convolutional decoder for image reconstruction"""
    def __init__(self, embed_dim: int = 768, img_size: int = 224, patch_size: int = 16, 
                 out_chans: int = 3, hidden_dims: List[int] = [512, 256, 128, 64]):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.feature_size = img_size // patch_size
        
        # Reshape patch tokens to 2D feature map
        self.initial_conv = nn.Conv2d(embed_dim, hidden_dims[0], kernel_size=1)
        
        # Upsampling layers
        layers = []
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = hidden_dim
        
        # Final layer to get original image size
        layers.append(nn.ConvTranspose2d(in_dim, out_chans, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patch tokens (B, num_patches, embed_dim)
        Returns:
            reconstructed image (B, out_chans, img_size, img_size)
        """
        B = x.shape[0]
        # Reshape to 2D feature map
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.feature_size, self.feature_size)
        x = self.initial_conv(x)
        x = self.decoder(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer-based decoder with learned mask tokens"""
    def __init__(self, embed_dim: int = 768, depth: int = 4, num_heads: int = 8, 
                 img_size: int = 224, patch_size: int = 16, out_chans: int = 3):
        super().__init__()
        from share_space.models.vit import Block
        
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Predict pixel values for each patch
        self.head = nn.Linear(embed_dim, patch_size * patch_size * out_chans)
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.out_chans = out_chans
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: patch tokens (B, num_patches, embed_dim)
            mask: binary mask (B, num_patches) - 1 means masked
        Returns:
            reconstructed image (B, out_chans, img_size, img_size)
        """
        B, N, C = x.shape
        
        # Apply mask if provided
        if mask is not None:
            mask_tokens = self.mask_token.expand(B, N, -1)
            w = mask.unsqueeze(-1).type_as(x)
            x = x * (1 - w) + mask_tokens * w
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Predict patches
        x = self.head(x)  # (B, N, patch_size^2 * out_chans)
        
        # Reshape to image
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(B, h, w, p, p, self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, self.img_size, self.img_size)
        
        return x

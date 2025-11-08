# models/decoders.py
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
import math

class ConvDecoder(nn.Module):
    """Convolutional decoder for image reconstruction"""
    def __init__(self, embed_dim: int = 768, 
                 img_size: Union[int, Tuple[int, int]] = 224, 
                 patch_size: Union[int, Tuple[int, int]] = 16, 
                 out_chans: int = 3, 
                 hidden_dims: Optional[List[int]] = None):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Handle both int and tuple for img_size and patch_size
        if isinstance(img_size, int):
            self.img_height, self.img_width = img_size, img_size
        else:
            self.img_height, self.img_width = img_size
            
        if isinstance(patch_size, int):
            self.patch_height, self.patch_width = patch_size, patch_size
        else:
            self.patch_height, self.patch_width = patch_size
        
        # Calculate feature map dimensions
        self.feature_height = self.img_height // self.patch_height
        self.feature_width = self.img_width // self.patch_width
        self.num_patches = self.feature_height * self.feature_width
        
        # Calculate required upsampling factors
        self.upsample_factor_h = self.patch_height
        self.upsample_factor_w = self.patch_width
        
        # Determine number of upsampling layers needed
        # Use the maximum of the two factors
        max_upsample = max(self.upsample_factor_h, self.upsample_factor_w)
        num_upsample_layers = int(math.log2(max_upsample))
        
        # Verify that patch sizes are powers of 2
        if 2 ** num_upsample_layers != max_upsample:
            raise ValueError(
                f"Patch sizes must be powers of 2. Got patch_size=({self.patch_height}, {self.patch_width})"
            )
        
        # Set default hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = [embed_dim // (2 ** i) for i in range(1, num_upsample_layers + 1)]
            hidden_dims = [max(64, dim) for dim in hidden_dims]
        
        if len(hidden_dims) < num_upsample_layers:
            raise ValueError(
                f"Need at least {num_upsample_layers} hidden dims for patch_size="
                f"({self.patch_height}, {self.patch_width}), got {len(hidden_dims)}"
            )
        
        # Initial conv to project from embedding dim
        self.initial_conv = nn.Conv2d(embed_dim, hidden_dims[0], kernel_size=1)
        
        # Upsampling layers
        layers = []
        in_dim = hidden_dims[0]
        
        for i in range(num_upsample_layers):
            out_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else hidden_dims[-1]
            layers.extend([
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = out_dim
        
        # Final layer to output channels
        layers.append(nn.Conv2d(in_dim, out_chans, kernel_size=3, padding=1))
        layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
        
        # Verify the output size will be correct
        self._verify_architecture()
    
    def _verify_architecture(self):
        """Verify the decoder will produce correct output size"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_patches, self.embed_dim)
            try:
                output = self.forward(dummy_input)
                expected_shape = (1, 3, self.img_height, self.img_width)
                if output.shape != expected_shape:
                    print(f"Warning: Decoder output shape {output.shape} doesn't match expected {expected_shape}")
                    print(f"This may require interpolation to fix.")
            except Exception as e:
                print(f"Warning: Architecture verification failed: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patch tokens (B, num_patches, embed_dim)
        Returns:
            reconstructed image (B, out_chans, img_height, img_width)
        """
        B = x.shape[0]
        
        # Reshape to 2D feature map
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.feature_height, self.feature_width)
        x = self.initial_conv(x)
        x = self.decoder(x)
        
        # Handle case where upsampling doesn't perfectly match target size
        if x.shape[2] != self.img_height or x.shape[3] != self.img_width:
            x = nn.functional.interpolate(
                x, 
                size=(self.img_height, self.img_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer-based decoder with learned mask tokens"""
    def __init__(self, embed_dim: int = 768, depth: int = 4, num_heads: int = 8, 
                 img_size: Union[int, Tuple[int, int]] = 224, 
                 patch_size: Union[int, Tuple[int, int]] = 16, 
                 out_chans: int = 3):
        super().__init__()
        from models.vit import Block
        
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        
        # Handle both int and tuple for img_size and patch_size
        if isinstance(img_size, int):
            self.img_height, self.img_width = img_size, img_size
        else:
            self.img_height, self.img_width = img_size
            
        if isinstance(patch_size, int):
            self.patch_height, self.patch_width = patch_size, patch_size
        else:
            self.patch_height, self.patch_width = patch_size
        
        # Validate dimensions
        if self.img_height % self.patch_height != 0:
            raise ValueError(
                f"img_height ({self.img_height}) must be divisible by patch_height ({self.patch_height})"
            )
        if self.img_width % self.patch_width != 0:
            raise ValueError(
                f"img_width ({self.img_width}) must be divisible by patch_width ({self.patch_width})"
            )
        
        # Calculate grid dimensions
        self.grid_height = self.img_height // self.patch_height
        self.grid_width = self.img_width // self.patch_width
        self.num_patches = self.grid_height * self.grid_width
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Predict pixel values for each patch
        self.head = nn.Linear(embed_dim, self.patch_height * self.patch_width * out_chans)
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: patch tokens (B, num_patches, embed_dim)
            mask: binary mask (B, num_patches) - 1 means masked
        Returns:
            reconstructed image (B, out_chans, img_height, img_width)
        """
        B, N, C = x.shape
        
        # Validate input
        if N != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches (grid: {self.grid_height}x{self.grid_width}) "
                f"but got {N}. Check encoder and decoder have matching img_size and patch_size."
            )
        
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
        x = self.head(x)  # (B, N, patch_height * patch_width * out_chans)
        
        # Reshape to image
        ph, pw = self.patch_height, self.patch_width
        gh, gw = self.grid_height, self.grid_width
        
        # Reshape: (B, gh*gw, ph*pw*c) -> (B, gh, gw, ph, pw, c)
        x = x.reshape(B, gh, gw, ph, pw, self.out_chans)
        
        # Rearrange: (B, gh, gw, ph, pw, c) -> (B, c, gh, ph, gw, pw) -> (B, c, gh*ph, gw*pw)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, self.img_height, self.img_width)
        
        return x


class SimpleLinearDecoder(nn.Module):
    """Simple linear decoder - most flexible for different configurations"""
    def __init__(self, embed_dim: int = 768, 
                 img_size: Union[int, Tuple[int, int]] = 224, 
                 patch_size: Union[int, Tuple[int, int]] = 16, 
                 out_chans: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        
        # Handle both int and tuple for img_size and patch_size
        if isinstance(img_size, int):
            self.img_height, self.img_width = img_size, img_size
        else:
            self.img_height, self.img_width = img_size
            
        if isinstance(patch_size, int):
            self.patch_height, self.patch_width = patch_size, patch_size
        else:
            self.patch_height, self.patch_width = patch_size
        
        # Calculate grid dimensions
        self.grid_height = self.img_height // self.patch_height
        self.grid_width = self.img_width // self.patch_width
        self.num_patches = self.grid_height * self.grid_width
        
        # Direct projection to patch pixels
        self.head = nn.Linear(embed_dim, self.patch_height * self.patch_width * out_chans)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patch tokens (B, num_patches, embed_dim)
        Returns:
            reconstructed image (B, out_chans, img_height, img_width)
        """
        B, N, C = x.shape
        
        # Validate input
        if N != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches but got {N}"
            )
        
        # Project to pixels
        x = self.head(x)  # (B, N, patch_height * patch_width * out_chans)
        
        # Unpatchify
        ph, pw = self.patch_height, self.patch_width
        gh, gw = self.grid_height, self.grid_width
        
        x = x.reshape(B, gh, gw, ph, pw, self.out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_chans, self.img_height, self.img_width)
        
        return x
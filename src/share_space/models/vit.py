# models/vit.py
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, 
                 img_size: Union[int, Tuple[int, int]] = 224, 
                 patch_size: Union[int, Tuple[int, int]] = 16, 
                 in_chans: int = 3, 
                 embed_dim: int = 768):
        super().__init__()
        
        # Handle both int and tuple for img_size
        if isinstance(img_size, int):
            self.img_height, self.img_width = img_size, img_size
        else:
            self.img_height, self.img_width = img_size
        
        # Handle both int and tuple for patch_size
        if isinstance(patch_size, int):
            self.patch_height = self.patch_width = patch_size
            self.patch_size = patch_size  # Keep for backward compatibility
        else:
            self.patch_height, self.patch_width = patch_size
            self.patch_size = patch_size
        
        # Calculate grid dimensions
        self.grid_height = self.img_height // self.patch_height
        self.grid_width = self.img_width // self.patch_width
        self.num_patches = self.grid_height * self.grid_width
        
        # Convolution projection
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=(self.patch_height, self.patch_width), 
            stride=(self.patch_height, self.patch_width)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Optional: validate input size matches expected dimensions
        # Can be disabled for dynamic sizing
        if hasattr(self, 'img_height') and hasattr(self, 'img_width'):
            if H % self.patch_height != 0 or W % self.patch_width != 0:
                raise ValueError(
                    f"Input size ({H}x{W}) is not divisible by patch size "
                    f"({self.patch_height}x{self.patch_width})"
                )
        
        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
    
    def get_num_patches(self, img_height: int, img_width: int) -> int:
        """Calculate number of patches for given image dimensions"""
        return (img_height // self.patch_height) * (img_width // self.patch_width)


class Attention(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, 
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block"""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, 
                 out_features: Optional[int] = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image encoding"""
    def __init__(self, 
                 img_size: Union[int, Tuple[int, int]] = 224, 
                 patch_size: Union[int, Tuple[int, int]] = 16, 
                 in_chans: int = 3, 
                 embed_dim: int = 768, 
                 depth: int = 12, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True, 
                 drop_rate: float = 0., 
                 attn_drop_rate: float = 0.):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0], x[:, 1:]  # cls_token, patch_tokens
    
    def get_num_patches_for_image(self, img_height: int, img_width: int) -> int:
        """Helper method to get number of patches for given image dimensions"""
        return self.patch_embed.get_num_patches(img_height, img_width)
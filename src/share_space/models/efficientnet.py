# models/efficientnet.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    from torchvision.models import efficientnet_b0, efficientnet_b3, EfficientNet_B0_Weights, EfficientNet_B3_Weights
except ImportError:
    from torchvision.models import efficientnet_b0, efficientnet_b3

class EfficientNetEncoder(nn.Module):
    """EfficientNet encoder for feature extraction"""
    def __init__(self, variant: str = 'b0', pretrained: bool = True, embed_dim: int = 1280):
        super().__init__()
        
        if variant == 'b0':
            try:
                self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif variant == 'b3':
            try:
                self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = efficientnet_b3(pretrained=pretrained)
            self.feature_dim = 1536
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")
        
        # Remove the classifier head
        self.backbone.classifier = nn.Identity()
        
        # Project to desired embedding dimension if different
        self.embed_dim = embed_dim
        if self.feature_dim != embed_dim:
            self.proj = nn.Linear(self.feature_dim, embed_dim)
        else:
            self.proj = nn.Identity()
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input image (B, C, H, W)
        Returns:
            global features (B, embed_dim), spatial features (B, feature_dim, H', W')
        """
        # Get features before final pooling
        features = self.backbone.features(x)  # (B, feature_dim, H', W')
        
        # Global features
        global_feat = self.pool(features).flatten(1)  # (B, feature_dim)
        global_feat = self.proj(global_feat)  # (B, embed_dim)
        
        return global_feat, features


class EfficientNetBackbone(nn.Module):
    """EfficientNet as a complete encoder (matching VisionTransformer interface)"""
    def __init__(self, variant: str = 'b0', pretrained: bool = True, 
                 img_size: int = 224, embed_dim: int = 768, **kwargs):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        self.encoder = EfficientNetEncoder(variant=variant, pretrained=pretrained, embed_dim=embed_dim)
        
        # For compatibility, store feature dimension
        self.num_features = embed_dim
        
        # Calculate spatial dimensions after backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            _, spatial_features = self.encoder(dummy_input)
            self.spatial_h, self.spatial_w = spatial_features.shape[2], spatial_features.shape[3]
            self.spatial_dim = self.encoder.feature_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input image (B, C, H, W)
        Returns:
            cls_token (global features), patch_tokens (spatial features as sequence)
        """
        global_feat, spatial_feat = self.encoder(x)
        
        # Convert spatial features to sequence (like patch tokens)
        B, C, H, W = spatial_feat.shape
        spatial_seq = spatial_feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        return global_feat, spatial_seq

# models/mobilenet.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
except ImportError:
    from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

class MobileNetEncoder(nn.Module):
    """MobileNet encoder for feature extraction"""
    def __init__(self, variant: str = 'v2', pretrained: bool = True, embed_dim: int = 1280):
        super().__init__()
        
        if variant == 'v2':
            try:
                self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = mobilenet_v2(pretrained=pretrained)
            self.feature_dim = 1280
        elif variant == 'v3_small':
            try:
                self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = mobilenet_v3_small(pretrained=pretrained)
            self.feature_dim = 576
        elif variant == 'v3_large':
            try:
                self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = mobilenet_v3_large(pretrained=pretrained)
            self.feature_dim = 960
        else:
            raise ValueError(f"Unknown MobileNet variant: {variant}")
        
        # Remove the classifier
        self.backbone.classifier = nn.Identity()
        
        # Project to desired embedding dimension
        self.embed_dim = embed_dim
        if self.feature_dim != embed_dim:
            self.proj = nn.Linear(self.feature_dim, embed_dim)
        else:
            self.proj = nn.Identity()
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input image (B, C, H, W)
        Returns:
            global features (B, embed_dim), spatial features (B, feature_dim, H', W')
        """
        # Get features
        features = self.backbone.features(x)
        
        # Global features
        global_feat = self.pool(features).flatten(1)
        global_feat = self.proj(global_feat)
        
        return global_feat, features


class MobileNetBackbone(nn.Module):
    """MobileNet as a complete encoder (matching VisionTransformer interface)"""
    def __init__(self, variant: str = 'v2', pretrained: bool = True,
                 img_size: int = 224, embed_dim: int = 768, **kwargs):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        self.encoder = MobileNetEncoder(variant=variant, pretrained=pretrained, embed_dim=embed_dim)
        
        self.num_features = embed_dim
        
        # Calculate spatial dimensions
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
        
        # Convert to sequence
        B, C, H, W = spatial_feat.shape
        spatial_seq = spatial_feat.flatten(2).transpose(1, 2)
        
        return global_feat, spatial_seq

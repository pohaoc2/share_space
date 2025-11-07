# models/resnet.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
try:
    from torchvision.models import resnet18, resnet34, resnet50, resnet101
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
except ImportError:
    from torchvision.models import resnet18, resnet34, resnet50, resnet101

class ResNetEncoder(nn.Module):
    """ResNet encoder for feature extraction"""
    def __init__(self, variant: str = '50', pretrained: bool = True, embed_dim: int = 2048):
        super().__init__()
        
        if variant == '18':
            try:
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif variant == '34':
            try:
                self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif variant == '50':
            try:
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif variant == '101':
            try:
                self.backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            except:
                self.backbone = resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
        
        # Remove the classifier and final pooling
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        
        # Custom pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection
        self.embed_dim = embed_dim
        if self.feature_dim != embed_dim:
            self.proj = nn.Linear(self.feature_dim, embed_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input image (B, C, H, W)
        Returns:
            global features (B, embed_dim), spatial features (B, feature_dim, H', W')
        """
        # Forward through layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)  # Spatial features
        
        # Global features
        global_feat = self.pool(features).flatten(1)
        global_feat = self.proj(global_feat)
        
        return global_feat, features


class ResNetBackbone(nn.Module):
    """ResNet as a complete encoder (matching VisionTransformer interface)"""
    def __init__(self, variant: str = '50', pretrained: bool = True,
                 img_size: int = 224, embed_dim: int = 768, **kwargs):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        self.encoder = ResNetEncoder(variant=variant, pretrained=pretrained, embed_dim=embed_dim)
        
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

"""
AdaIN (Adaptive Instance Normalization) module and VGG feature extractor
Based on the paper: "StyDiff: a refined style transfer method based on diffusion models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    VGG-16 feature extractor for content and style features
    Extracts features from multiple layers: conv1_2, conv2_2, conv3_2, conv4_2
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Extract specific layers
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # conv1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])   # conv2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # conv3_2
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)]) # conv4_2
        
        # Freeze weights if not training
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        """
        Extract multi-scale features
        Returns: List of feature maps from different layers
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        return [h1, h2, h3, h4]


def calc_mean_std(features):
    """
    Calculate mean and standard deviation of features
    
    Args:
        features: Feature tensor of shape (B, C, H, W)
    
    Returns:
        mean: Mean of shape (B, C, 1, 1)
        std: Standard deviation of shape (B, C, 1, 1)
    """
    batch_size, channels = features.size()[:2]
    
    features_mean = features.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    features_std = features.view(batch_size, channels, -1).std(dim=2).view(batch_size, channels, 1, 1) + 1e-6
    
    return features_mean, features_std


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization
    
    Normalizes content features and modifies them using style statistics.
    Formula from paper (Eq. 5):
    F_adapted = γ(F_c) · σ(F_s) + μ(F_s)
    
    where:
    - F_c: content features
    - F_s: style features
    - γ: normalization function
    - μ, σ: mean and standard deviation
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, content_features, style_features):
        """
        Apply AdaIN transformation
        
        Args:
            content_features: Content feature tensor (B, C, H, W)
            style_features: Style feature tensor (B, C, H, W)
        
        Returns:
            Adapted features with style statistics
        """
        assert content_features.size()[:2] == style_features.size()[:2], \
            "Content and style features must have same batch size and channels"
        
        # Calculate statistics
        content_mean, content_std = calc_mean_std(content_features)
        style_mean, style_std = calc_mean_std(style_features)
        
        # Normalize content features
        normalized_content = (content_features - content_mean) / content_std
        
        # Apply style statistics
        adapted_features = normalized_content * style_std + style_mean
        
        return adapted_features


class MultiScaleAdaIN(nn.Module):
    """
    Multi-scale AdaIN that applies AdaIN at multiple feature levels
    """
    def __init__(self):
        super().__init__()
        self.adain = AdaIN()
        
    def forward(self, content_features_list, style_features_list):
        """
        Apply AdaIN at multiple scales
        
        Args:
            content_features_list: List of content features from different layers
            style_features_list: List of style features from different layers
        
        Returns:
            List of adapted features
        """
        adapted_features = []
        
        for content_feat, style_feat in zip(content_features_list, style_features_list):
            adapted = self.adain(content_feat, style_feat)
            adapted_features.append(adapted)
        
        return adapted_features


class AdaINFeatureFusion(nn.Module):
    """
    Complete AdaIN feature fusion module
    Combines VGG feature extraction with AdaIN normalization
    """
    def __init__(self):
        super().__init__()
        
        self.vgg_extractor = VGGFeatureExtractor(requires_grad=False)
        self.multi_scale_adain = MultiScaleAdaIN()
        
    def forward(self, content_image, style_image):
        """
        Extract and fuse content and style features
        
        Args:
            content_image: Content image tensor (B, 3, H, W)
            style_image: Style image tensor (B, 3, H, W)
        
        Returns:
            adapted_features: List of adapted features at multiple scales
            content_features: Original content features for loss calculation
            style_features: Original style features for loss calculation
        """
        # Normalize images for VGG (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(content_image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(content_image.device)
        
        content_normalized = (content_image - mean) / std
        style_normalized = (style_image - mean) / std
        
        # Extract features
        content_features = self.vgg_extractor(content_normalized)
        style_features = self.vgg_extractor(style_normalized)
        
        # Apply AdaIN
        adapted_features = self.multi_scale_adain(content_features, style_features)
        
        return adapted_features, content_features, style_features

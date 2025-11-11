"""
Evaluation metrics for StyDiff
Based on the paper: "StyDiff: a refined style transfer method based on diffusion models"

Metrics:
1. SSIM: Structural Similarity Index
2. GM: Gram Matrix distance (style fidelity)
3. LPIPS: Learned Perceptual Image Patch Similarity
4. PD: Perceptual Dissimilarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images
    
    Args:
        img1: First image tensor (B, C, H, W) in range [0, 1]
        img2: Second image tensor (B, C, H, W) in range [0, 1]
        window_size: Size of the Gaussian window
        size_average: Whether to average over batch
    
    Returns:
        SSIM value
    """
    channel = img1.size(1)
    
    # Create Gaussian window
    def create_window(window_size, channel):
        def gaussian(window_size, sigma=1.5):
            gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / (2 * sigma ** 2) 
                                           for x in range(window_size)]))
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    window = create_window(window_size, channel).to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation"""
    def forward(self, features):
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)


def gram_matrix_distance(style_features, generated_features):
    """
    Calculate Gram Matrix distance (GM) for style fidelity
    
    Args:
        style_features: Style image features from VGG
        generated_features: Generated image features from VGG
    
    Returns:
        GM distance (lower is better)
    """
    gram_module = GramMatrix()
    
    style_gram = gram_module(style_features)
    generated_gram = gram_module(generated_features)
    
    distance = torch.norm(style_gram - generated_gram, p=2)
    return distance.item()


class LPIPS(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity)
    Uses pretrained VGG features to compute perceptual similarity
    """
    def __init__(self):
        super().__init__()
        
        # Use VGG16 pretrained features
        vgg = models.vgg16(pretrained=True).features
        
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(16, 23)])
        self.slice5 = nn.Sequential(*[vgg[i] for i in range(23, 30)])
        
        for param in self.parameters():
            param.requires_grad = False
        
        # Learned linear layers for weighting features
        self.linear1 = nn.Conv2d(64, 1, 1, bias=False)
        self.linear2 = nn.Conv2d(128, 1, 1, bias=False)
        self.linear3 = nn.Conv2d(256, 1, 1, bias=False)
        self.linear4 = nn.Conv2d(512, 1, 1, bias=False)
        self.linear5 = nn.Conv2d(512, 1, 1, bias=False)
        
    def forward(self, img1, img2):
        """
        Calculate LPIPS distance
        
        Args:
            img1: First image (B, 3, H, W) in range [0, 1]
            img2: Second image (B, 3, H, W) in range [0, 1]
        
        Returns:
            LPIPS distance (lower is better)
        """
        # Normalize for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
        
        img1 = (img1 - mean) / std
        img2 = (img2 - mean) / std
        
        # Extract features
        h1_1 = self.slice1(img1)
        h1_2 = self.slice2(h1_1)
        h1_3 = self.slice3(h1_2)
        h1_4 = self.slice4(h1_3)
        h1_5 = self.slice5(h1_4)
        
        h2_1 = self.slice1(img2)
        h2_2 = self.slice2(h2_1)
        h2_3 = self.slice3(h2_2)
        h2_4 = self.slice4(h2_3)
        h2_5 = self.slice5(h2_4)
        
        # Compute differences
        diff1 = (h1_1 - h2_1) ** 2
        diff2 = (h1_2 - h2_2) ** 2
        diff3 = (h1_3 - h2_3) ** 2
        diff4 = (h1_4 - h2_4) ** 2
        diff5 = (h1_5 - h2_5) ** 2
        
        # Spatial average
        val1 = self.linear1(diff1).mean(dim=[2, 3])
        val2 = self.linear2(diff2).mean(dim=[2, 3])
        val3 = self.linear3(diff3).mean(dim=[2, 3])
        val4 = self.linear4(diff4).mean(dim=[2, 3])
        val5 = self.linear5(diff5).mean(dim=[2, 3])
        
        return (val1 + val2 + val3 + val4 + val5).mean()


class PerceptualDissimilarity(nn.Module):
    """
    PD (Perceptual Dissimilarity) metric
    Measures perceptual differences using high-level features
    """
    def __init__(self):
        super().__init__()
        
        # Use VGG for high-level features
        vgg = models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*[vgg[i] for i in range(23)])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, img1, img2):
        """
        Calculate PD
        
        Args:
            img1: First image (B, 3, H, W)
            img2: Second image (B, 3, H, W)
        
        Returns:
            PD value (higher means more dissimilar)
        """
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
        
        img1 = (img1 - mean) / std
        img2 = (img2 - mean) / std
        
        # Extract features
        feat1 = self.features(img1)
        feat2 = self.features(img2)
        
        # Calculate L2 distance
        distance = torch.norm(feat1 - feat2, p=2)
        return distance


class StyDiffMetrics(nn.Module):
    """
    Complete metrics evaluator for StyDiff
    Calculates SSIM, GM, LPIPS, and PD
    """
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.device = device
        self.lpips = LPIPS().to(device)
        self.pd = PerceptualDissimilarity().to(device)
        self.gram = GramMatrix()
        
        # VGG for GM calculation
        vgg = models.vgg16(pretrained=True).features
        self.vgg_conv4_2 = nn.Sequential(*[vgg[i] for i in range(23)]).to(device)
        
        for param in self.vgg_conv4_2.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def evaluate(self, content_img, style_img, generated_img):
        """
        Evaluate all metrics
        
        Args:
            content_img: Content image (B, 3, H, W) in range [0, 1]
            style_img: Style image (B, 3, H, W) in range [0, 1]
            generated_img: Generated image (B, 3, H, W) in range [0, 1]
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # 1. SSIM (compare with content for structure preservation)
        metrics['ssim'] = ssim(content_img, generated_img).item()
        
        # 2. GM (Gram Matrix distance - style fidelity)
        # Normalize for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        style_norm = (style_img - mean) / std
        generated_norm = (generated_img - mean) / std
        
        style_feat = self.vgg_conv4_2(style_norm)
        generated_feat = self.vgg_conv4_2(generated_norm)
        
        metrics['gm'] = gram_matrix_distance(style_feat, generated_feat)
        
        # 3. LPIPS (perceptual similarity)
        metrics['lpips'] = self.lpips(content_img, generated_img).item()
        
        # 4. PD (perceptual dissimilarity with style)
        metrics['pd'] = self.pd(style_img, generated_img).item()
        
        return metrics
    
    @torch.no_grad()
    def evaluate_batch(self, content_imgs, style_imgs, generated_imgs):
        """
        Evaluate metrics for a batch and return averages
        
        Args:
            content_imgs: Batch of content images
            style_imgs: Batch of style images  
            generated_imgs: Batch of generated images
        
        Returns:
            Dictionary with averaged metrics
        """
        all_metrics = {
            'ssim': [],
            'gm': [],
            'lpips': [],
            'pd': []
        }
        
        for i in range(content_imgs.shape[0]):
            metrics = self.evaluate(
                content_imgs[i:i+1],
                style_imgs[i:i+1],
                generated_imgs[i:i+1]
            )
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
        
        # Average
        avg_metrics = {
            key: np.mean(values) for key, values in all_metrics.items()
        }
        
        return avg_metrics

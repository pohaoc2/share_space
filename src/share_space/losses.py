"""
Loss functions for StyDiff
Based on the paper: "StyDiff: a refined style transfer method based on diffusion models"

Includes:
1. Content Loss (L_ImageLatent): Preserves content structure in latent space
2. Style Loss (L_StyleLatent): Ensures style consistency in latent space
3. Element Loss (L_Element): Fine-grained element-level matching
4. Diffusion Loss (L_diff): Optimizes the diffusion model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """
    Content Loss (Equation 6 in paper)
    L_ImageLatent = ||VDVAE(X_i) - VDVAE(X_output)||²
    
    Ensures generated image preserves content structure in latent space
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, content_latent, output_latent):
        """
        Args:
            content_latent: Encoded content image latent (from AutoKL)
            output_latent: Encoded output image latent
        
        Returns:
            Content preservation loss
        """
        return self.criterion(content_latent, output_latent)


class StyleLoss(nn.Module):
    """
    Style Loss (Equation 7 in paper)
    L_StyleLatent = ||VDVAE(X_s) - VDVAE(X_output)||²
    
    Ensures generated image captures style characteristics in latent space
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, style_latent, output_latent):
        """
        Args:
            style_latent: Encoded style image latent
            output_latent: Encoded output image latent
        
        Returns:
            Style consistency loss
        """
        return self.criterion(style_latent, output_latent)

class ElementLoss(nn.Module):
    """
    Element Loss (Equation 9 in paper)
    L_Element = ||A(X_s, X_i) - VDVAE(X_output)||²
    
    where A(X_s, X_i) represents the AdaIN-adapted features
    
    Ensures fine-grained element-level matching between adapted features and output
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, adapted_latent, output_latent):
        """
        Args:
            adapted_latent: AdaIN-adapted features in latent space
            output_latent: Encoded output image latent
        
        Returns:
            Element-level matching loss
        """
        return self.criterion(adapted_latent, output_latent)


class DiffusionLoss(nn.Module):
    """
    Diffusion Model Loss (Equation 8 in paper)
    L_diff = -log P_θ(p(x_t|X_0), t, A(X_s, X_i))
    
    Optimizes the noise prediction in the diffusion process
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, noise_pred, noise_target):
        """
        Args:
            noise_pred: Predicted noise from diffusion model
            noise_target: Actual noise added to the image
        
        Returns:
            Diffusion loss (noise prediction error)
        """
        return self.criterion(noise_pred, noise_target)


class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation"""
    def forward(self, features):
        """
        Args:
            features: Feature tensor (B, C, H, W)
        
        Returns:
            Gram matrix (B, C, C)
        """
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)


class StyleFeatureLoss(nn.Module):
    """
    Style feature loss using Gram matrices
    Measures style similarity in VGG feature space
    """
    def __init__(self):
        super().__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
    
    def forward(self, style_features_list, output_features_list):
        """
        Args:
            style_features_list: List of style features from VGG
            output_features_list: List of output features from VGG
        
        Returns:
            Style feature loss across multiple layers
        """
        loss = 0.0
        
        for style_feat, output_feat in zip(style_features_list, output_features_list):
            style_gram = self.gram(style_feat)
            output_gram = self.gram(output_feat)
            loss += self.criterion(style_gram, output_gram)
        
        return loss / len(style_features_list)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    Measures perceptual similarity between images
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    
    def forward(self, content_features_list, output_features_list):
        """
        Args:
            content_features_list: List of content features from VGG
            output_features_list: List of output features from VGG
        
        Returns:
            Perceptual loss across multiple layers
        """
        loss = 0.0
        
        for content_feat, output_feat in zip(content_features_list, output_features_list):
            loss += self.criterion(content_feat, output_feat)
        
        return loss / len(content_features_list)


class AutoKLLoss(nn.Module):
    """
    Loss for AutoKL
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, content_original, style_original, content_recon, style_recon):
        """
        Args:
            original: Original image
            recon: Reconstructed image
        
        Returns:
            AutoKL loss
        """
        losses = {}
        losses['auto_kl'] = self.criterion(content_original, content_recon) + self.criterion(style_original, style_recon)
        losses['total'] = losses['auto_kl']
        return losses

class StyDiffLoss(nn.Module):
    """
    Complete StyDiff loss function combining all components
    
    L_total = α * L_ImageLatent + β * L_StyleLatent + γ * L_Element + δ * L_diff
    
    Default weights from paper's ablation study (Table 4):
    α=1.0, β=1.0, γ=1.0, δ=1.0 (baseline)
    """
    def __init__(
        self,
        content_weight=1.0,      # α
        style_weight=1.0,        # β
        element_weight=1.0,      # γ
        diffusion_weight=1.0,    # δ
        perceptual_weight=0.1,   # Additional perceptual loss
        style_feature_weight=1.0  # Additional style feature loss
    ):
        super().__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.element_weight = element_weight
        self.diffusion_weight = diffusion_weight
        self.perceptual_weight = perceptual_weight
        self.style_feature_weight = style_feature_weight
        
        # Loss components
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.element_loss = ElementLoss()
        self.diffusion_loss = DiffusionLoss()
        self.perceptual_loss = PerceptualLoss()
        self.style_feature_loss = StyleFeatureLoss()
    
    def forward(
        self,
        content_latent,
        style_latent,
        output_latent,
        fused_latent,
        noise_pred,
        noise_target,
        content_features=None,
        style_features=None,
        output_features=None
    ):
        """
        Calculate total loss
        
        Args:
            content_latent: Encoded content image latent
            style_latent: Encoded style image latent
            output_latent: Encoded output image latent
            fused_latent: AdaIN-fused latent (A(Xs, Xi))
            noise_pred: Predicted noise from diffusion
            noise_target: Actual noise
            content_features: VGG features of content (optional)
            style_features: VGG features of style (optional)
            output_features: VGG features of output (optional)
        
        Returns:
            Dictionary containing total loss and individual components
        """
        losses = {}
        
        # 1. Content Loss (L_ImageLatent)
        losses['content'] = self.content_loss(content_latent, output_latent)
        
        # 2. Style Loss (L_StyleLatent) - we want output to capture style
        # Note: In style transfer, we don't want exact match to style latent
        # Instead, we use style features for better style capture
        losses['style'] = self.style_loss(style_latent, output_latent)
        
        # 3. Element Loss (L_Element)
        losses['element'] = self.element_loss(fused_latent, output_latent)
        
        # 4. Diffusion Loss (L_diff)
        losses['diffusion'] = self.diffusion_loss(noise_pred, noise_target)
        
        # Additional losses
        if content_features is not None and output_features is not None:
            losses['perceptual'] = self.perceptual_loss(content_features, output_features)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=content_latent.device)
        
        if style_features is not None and output_features is not None:
            losses['style_feature'] = self.style_feature_loss(style_features, output_features)
        else:
            losses['style_feature'] = torch.tensor(0.0, device=content_latent.device)
        
        # Total loss
        losses['total'] = (
            self.content_weight * losses['content'] +
            self.style_weight * losses['style'] +
            self.element_weight * losses['element'] +
            self.diffusion_weight * losses['diffusion'] +
            self.perceptual_weight * losses['perceptual'] +
            self.style_feature_weight * losses['style_feature']
        )
        
        return losses

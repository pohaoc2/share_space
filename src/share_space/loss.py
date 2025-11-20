# loss.py (Updated with dtype fixes)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from typing import Optional
class Sim2ExpLoss(nn.Module):
    """Combined loss for simulation to experimental translation"""
    def __init__(self, recon_weight: float = 1.0, distill_weight: float = 1.0, 
                 mask_weight: float = 1.0, perceptual_weight: float = 0.1,
                 diffusion_weight: float = 1.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.distill_weight = distill_weight
        self.mask_weight = mask_weight
        self.perceptual_weight = perceptual_weight
        self.diffusion_weight = diffusion_weight
    
    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 + L2 reconstruction loss"""
        # Ensure float32
        pred = pred.float()
        target = target.float()
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        return l1_loss + l2_loss
    
    def distillation_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                         temperature: float = 0.04) -> torch.Tensor:
        """
        Cross-entropy distillation loss between student and teacher features
        Similar to DINO/DINOv2
        """
        # Ensure float32
        student_feat = student_feat.float()
        teacher_feat = teacher_feat.float()
        
        student_feat = F.normalize(student_feat, dim=-1, p=2)
        teacher_feat = F.normalize(teacher_feat, dim=-1, p=2)
        
        # Compute similarity (using matmul for proper broadcasting)
        student_out = student_feat / temperature
        teacher_out = teacher_feat / temperature
        
        # Softmax on teacher (sharper)
        teacher_out = F.softmax(teacher_out, dim=-1)
        
        # Log softmax on student
        student_out = F.log_softmax(student_out, dim=-1)
        
        # Cross entropy
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        
        return loss
    
    def masked_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                                   mask: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
        """
        Reconstruction loss only on masked patches
        
        Args:
            pred: predicted image (B, C, H, W)
            target: target image (B, C, H, W)
            mask: binary mask (B, num_patches)
            patch_size: size of each patch
        """
        # Ensure float32
        pred = pred.float()
        target = target.float()
        mask = mask.float()
        
        B, C, H, W = pred.shape
        if isinstance(patch_size, int):
            p_h = p_w = patch_size
        else:
            p_h, p_w = patch_size
        
        # Convert image to patches
        pred_patches = pred.unfold(2, p_h, p_h).unfold(3, p_w, p_w)  # (B, C, H//p, W//p, p, p)
        target_patches = target.unfold(2, p_h, p_h).unfold(3, p_w, p_w)
        
        pred_patches = pred_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p_h * p_w)
        target_patches = target_patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p_h * p_w)
        
        # Compute loss only on masked patches
        mask_expanded = mask.unsqueeze(-1).expand_as(pred_patches)
        
        # Compute masked MSE
        diff = (pred_patches - target_patches) ** 2
        masked_diff = diff * mask_expanded
        
        # Average over masked elements
        num_masked = mask.sum() * C * p_h * p_w
        if num_masked > 0:
            loss = masked_diff.sum() / num_masked
        else:
            loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        
        return loss
    
    def diffusion_loss(self, noisy_latent: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """
        Diffusion loss
        """
        return F.mse_loss(noise_pred, noisy_latent)
    
    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor,
                mask: torch.Tensor, patch_size: Tuple[int, int], noisy_latent: torch.Tensor, noise_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        
        Args:
            outputs: dict from forward pass containing:
                - student_recon: reconstructed image from student
                - student_cls: cls token from student
                - student_patches: patch tokens from student
                - teacher_recon: reconstructed image from teacher
                - teacher_cls: cls token from teacher
                - teacher_patches: patch tokens from teacher
            target: target experimental image
            mask: binary mask for masked patches
        
        Returns:
            dict of losses
        """
        losses = {}
        
        # Ensure float32 for target
        target = target.float()
        
        # Reconstruction loss (student predicts experimental image)
        losses['recon'] = self.recon_weight * self.reconstruction_loss(
            outputs['student_recon'], target
        )
        
        # Distillation loss (student matches teacher representations)
        losses['distill_cls'] = self.distill_weight * self.distillation_loss(
            outputs['student_cls'], outputs['teacher_cls']
        )
        
        # Masked reconstruction loss
        losses['mask_recon'] = self.mask_weight * self.masked_reconstruction_loss(
            outputs['student_recon'], target, mask, patch_size
        )

        # Diffusion loss
        losses['diffusion'] = self.diffusion_weight * self.diffusion_loss(
            noisy_latent, noise_pred
        )
        return losses


class PerceptualLoss(nn.Module):
    """Perceptual loss using pretrained network"""
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        # Use VGG16 for perceptual loss (common choice)
        from torchvision import models
        vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
        
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure float32
        pred = pred.float()
        target = target.float()
        
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return F.mse_loss(pred_features, target_features)


class StyleTransferLoss(nn.Module):
    """
    Multi-component loss function for style transfer
    Implements equations 6, 7, 8 (optional), and 9 from the paper
    """
    def __init__(self, 
                 use_diffusion: bool = True,
                 content_weight: float = 1.0,
                 style_weight: float = 1.0,
                 element_weight: float = 1.0,
                 diffusion_weight: float = 1.0,
                 image_weight: float = 1.0,
                 fused_patches_weight: float = 1.0):
        super().__init__()
        self.use_diffusion = use_diffusion
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.element_weight = element_weight
        self.diffusion_weight = diffusion_weight
        self.image_weight = image_weight
        self.fused_patches_weight = fused_patches_weight

    def content_loss(self, content_latent: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Content Loss (Equation 6): L_ImageLatent = ||VDVAE(X_i) - VDVAE(X_output)||²
        Preserves content structure in latent space
        """
        return F.mse_loss(content_latent, output_latent)
    
    def style_loss(self, style_latent: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Style Loss (Equation 7): L_StyleLatent = ||VDVAE(X_s) - VDVAE(X_output)||²_2
        Ensures style consistency in latent space
        """
        return F.mse_loss(style_latent, output_latent)
    
    def diffusion_loss(self, noise_pred: torch.Tensor, noise_target: torch.Tensor) -> torch.Tensor:
        """
        Diffusion Model Loss (Equation 8): L_diff = -log P_θ(p(x_t|X_0), t, A(X_s, X_i))
        Optimizes noise alignment in diffusion process
        
        In practice, this is implemented as MSE between predicted and actual noise
        """
        return F.mse_loss(noise_pred, noise_target)
    
    def element_loss(self, adain_features: torch.Tensor, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Element Loss (Equation 9): L_Element = ||A(X_s, X_i) - VDVAE(X_output)||²_2
        Measures fine-grained differences at element level
        """
        return F.mse_loss(adain_features, output_latent)
    

    def image_loss(self, output_images: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Image Loss (Eq. 10): L_Image = ||X_output - X_target||²_2
        Measures image-level similarity
        """
        return F.mse_loss(output_images, target)
    
    def fused_patches_loss(self, fused_features_patches: torch.Tensor, style_features_patches: torch.Tensor) -> torch.Tensor:
        """
        Fused features patches loss (Eq. 11): L_FusedPatches = ||A(X_s, X_i) - X_s||²_2
        Measures fine-grained differences at element level
        """
        return F.mse_loss(fused_features_patches, style_features_patches)
    
    def forward(self, 
                content_latent: torch.Tensor,
                style_latent: torch.Tensor,
                style_features_patches: torch.Tensor,
                output_latent: torch.Tensor,
                adain_features: torch.Tensor,
                fused_features_patches: torch.Tensor,
                output_images: torch.Tensor,
                target_images: torch.Tensor,
                noise_pred: Optional[torch.Tensor] = None,
                noise_target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components
        
        Returns:
            Dictionary with all loss components
        """
        losses = {}
        
        # Content loss (Eq. 6)
        losses['content'] = self.content_weight * self.content_loss(content_latent, output_latent)
        
        # Style loss (Eq. 7)
        losses['style'] = self.style_weight * self.style_loss(style_latent, output_latent)
        
        # Element loss (Eq. 9)
        losses['element'] = self.element_weight * self.element_loss(adain_features, output_latent)
        
        # Fused features patches loss (Eq. 11)
        losses['fused_patches'] = self.fused_patches_weight * self.fused_patches_loss(fused_features_patches, style_features_patches)
        
        # Diffusion loss (Eq. 8) - optional
        if self.use_diffusion and noise_pred is not None and noise_target is not None:
            losses['diffusion'] = self.diffusion_weight * self.diffusion_loss(noise_pred, noise_target)
        
        # Image loss (Eq. 10)
        losses['image'] = self.image_weight * self.image_loss(output_images, target_images)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses

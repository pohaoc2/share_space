# loss.py (Updated with dtype fixes)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class Sim2ExpLoss(nn.Module):
    """Combined loss for simulation to experimental translation"""
    def __init__(self, recon_weight: float = 1.0, distill_weight: float = 1.0, 
                 mask_weight: float = 1.0, perceptual_weight: float = 0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.distill_weight = distill_weight
        self.mask_weight = mask_weight
        self.perceptual_weight = perceptual_weight
    
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
    
    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor,
                mask: torch.Tensor, patch_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
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
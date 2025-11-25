# style_transfer_trainer.py
"""
StyleTransferTrainer: Implementation of StyDiff framework for style transfer
Based on "StyDiff: a refined style transfer method based on diffusion models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import copy
from share_space.loss import StyleTransferLoss
from share_space.models.sim2exp_model import StyleTransferModel
import math
import matplotlib.pyplot as plt

class StyleTransferTrainer:
    """
    Main trainer class for StyDiff-style style transfer
    
    Architecture:
    1. Frozen feature extractor for content and style
    2. AdaIN fusion module
    3. Trainable decoder
    4. Optional diffusion model for refinement
    """
    def __init__(self,
                 model: StyleTransferModel,
                 device: str = 'cuda',
                 use_diffusion: bool = True,
                 diffusion_model: Optional[nn.Module] = None,
                 use_all_pairs: bool = False,
                 ):
        """
        Args:
            model: StyleTransferModel
            device: Device to run on
            use_diffusion: Whether to use diffusion model refinement
            diffusion_model: Optional diffusion model for refinement
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            element_weight: Weight for element loss
            diffusion_weight: Weight for diffusion loss
        """
        self.device = device
        self.use_diffusion = use_diffusion
        self.model = model.to(device)
        # Feature extractor (frozen)
        self.feature_extractor = model.encoder.to(device)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.decoder = model.decoder.to(device)
        # AdaIN fusion module
        self.adain = model.adain.to(device)
        self.use_all_pairs = use_all_pairs
        # Optional diffusion model
        if use_diffusion and diffusion_model is not None:
            self.diffusion = diffusion_model.to(device)
        else:
            self.diffusion = None
            self.use_diffusion = False
            
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using frozen feature extractor
        
        Args:
            images: Input images (B, C, H, W)
        
        Returns:
            Features in appropriate format (B, embed_dim)
        """
        self.feature_extractor.eval()
        features = self.feature_extractor(images)
        return features
    
    def forward_pass(self, 
                    content_images: torch.Tensor,
                    style_images: torch.Tensor,
                    shuffled_style_images: torch.Tensor,
                    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the style transfer pipeline
        
        Args:
            content_images: Content images (B, C, H, W)
            style_images: Style images (B, C, H, W)
            shuffled_style_images: Only used for 1:1 mode
        
        Returns:
            Dictionary containing all intermediate outputs
        """
        content_images = content_images.float()
        style_images = style_images.float()
        shuffled_style_images = shuffled_style_images.float()
        B = content_images.shape[0]
        
        # Step 1: Extract features (frozen)
        with torch.no_grad():
            _, content_features_cls, content_features_patches = self.extract_features(content_images)
            _, style_features_cls, style_features_patches = self.extract_features(style_images)    
            _, shuffled_style_features_cls, shuffled_style_features_patches = self.extract_features(shuffled_style_images)
        # Step 2: Fuse features using AdaIN
        if not self.use_all_pairs:
            # === 1:1 Pairing Mode ===
            fused_features_patches = self.adain(content_features_patches, shuffled_style_features_patches)
            fused_features_cls = self.adain(content_features_cls, shuffled_style_features_cls)
            fused_features_cls = fused_features_cls.view(fused_features_cls.shape[0], -1)
            
            # Decode
            output_images = self.decoder(fused_features_patches)
            
            # Extract features from output
            with torch.no_grad():
                _, output_features_cls, output_features_patches = self.extract_features(output_images)
            
            content_features_cls_expanded = content_features_cls
            style_features_patches_target = shuffled_style_features_patches
            style_features_cls_target = shuffled_style_features_cls            
            n_pairs = B
            
        else:
            # === All-Pairs Mode ===
            N, D = content_features_patches.shape[1], content_features_patches.shape[2]
            
            # Expand to (B, B, N, D) for all pairs
            content_expanded = content_features_patches.unsqueeze(1).expand(-1, B, -1, -1)
            shuffled_style_expanded = shuffled_style_features_patches.unsqueeze(0).expand(B, -1, -1, -1)
            # Flatten to (B*B, N, D)
            content_flat = content_expanded.reshape(B * B, N, D)
            shuffled_style_flat = shuffled_style_expanded.reshape(B * B, N, D)
            
            # Apply AdaIN to all pairs at once
            fused_features_patches = self.adain(content_flat, shuffled_style_flat)
            
            # Also fuse CLS tokens for all pairs
            content_cls_expanded = content_features_cls.unsqueeze(1).expand(-1, B, -1)
            shuffled_style_cls_expanded = shuffled_style_features_cls.unsqueeze(1).expand(-1, B, -1)
            
            content_cls_flat = content_cls_expanded.reshape(B * B, -1)
            shuffled_style_cls_flat = shuffled_style_cls_expanded.reshape(B * B, -1)
            fused_features_cls = self.adain(content_cls_flat, shuffled_style_cls_flat)
            fused_features_cls = fused_features_cls.view(fused_features_cls.shape[0], -1)
            
            # Remove self-pairs (diagonal: i==j)
            mask = ~torch.eye(B, dtype=torch.bool, device=content_images.device).reshape(-1)
            
            fused_features_patches = fused_features_patches[mask]  # (B*(B-1), N, D)
            fused_features_cls = fused_features_cls[mask]  # (B*(B-1), D_cls)
            
            # === Expand content features to match output shape ===
            # For each pair (i,j), we need content[i]
            content_indices = []
            style_indices = []
            for i in range(B):
                for j in range(B):
                    if i != j:
                        content_indices.append(i)
                        style_indices.append(j)
            
            # Expand content CLS to (B*(B-1), D)
            content_features_cls_expanded = content_features_cls[content_indices]
            
            # Decode all pairs
            output_images = self.decoder(fused_features_patches)
            
            # Extract features from outputs
            with torch.no_grad():
                _, output_features_cls, output_features_patches = self.extract_features(output_images)
            
            # Get style targets
            style_features_patches_target = style_features_patches[style_indices]
            style_features_cls_target = style_features_cls[style_indices]
            n_pairs = B * (B - 1)
        
        # Step 3: Optional diffusion refinement
        noise_pred = None
        noise_target = None
        
        if self.use_diffusion and self.diffusion is not None:
            batch_size_effective = fused_features_cls.shape[0]
            t = torch.randint(
                0, 
                self.diffusion.timesteps,
                (batch_size_effective,),
                device=self.device
            )
            n_patches = fused_features_patches.shape[1]
            fused_latent_input = fused_features_patches.view(
                batch_size_effective, 
                self.diffusion.unet.in_channels,
                n_patches
            )
            noise_target = torch.randn_like(fused_latent_input)
            
            # Get alpha values
            alpha_t = self.diffusion.alphas_cumprod[t].view(-1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            
            # Create noisy version
            noisy_features = sqrt_alpha_t * fused_latent_input + sqrt_one_minus_alpha_t * noise_target
            # Predict noise
            noise_pred = self.diffusion.unet(noisy_features, t)
        
        return {
            'content_features_cls': content_features_cls_expanded,
            'style_features_cls': style_features_cls_target,
            'style_features_patches': style_features_patches_target,
            'fused_features_patches': fused_features_patches,
            'fused_features_cls': fused_features_cls,
            'output_images': output_images,
            'output_features_cls': output_features_cls,
            'output_features_patches': output_features_patches,
            'noise_pred': noise_pred,
            'noise_target': noise_target,
            'n_pairs': n_pairs
        }


    def train_step(self, 
                batch: Dict[str, torch.Tensor],
                optimizer: torch.optim.Optimizer,
                loss_fn,
                mask_ratio: None = None,
                ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary with 'simulation' and 'experimental' keys
            optimizer: Optimizer for decoder (and optionally diffusion model)
            loss_fn: Loss function
            mask_ratio: Not used in style transfer
        
        Returns:
            Dictionary of loss values
        """
        self.decoder.train()
        self.adain.train()  # Make sure AdaIN is in training mode
        if self.diffusion is not None:
            self.diffusion.train()
        
        # Get images
        content_images = batch['simulation'].to(self.device).float()
        style_images = batch['experimental'].to(self.device).float()
        shuffled_style_images = batch.get('shuffled_exp', None)
        if shuffled_style_images is not None:
            shuffled_style_images = shuffled_style_images.to(self.device).float()
        
        # Forward pass
        outputs = self.forward_pass(
            content_images, 
            style_images, 
            shuffled_style_images,
        )
        if self.use_all_pairs:
            B = style_images.shape[0]
            style_indices = []
            for i in range(B):
                for j in range(B):
                    if i != j:
                        style_indices.append(j)
            target_images = style_images[style_indices]  # (B*(B-1), C, H, W)
        else:
            target_images = style_images  # (B, C, H, W)
        # Compute losses
        losses = loss_fn(
            content_latent=outputs['content_features_cls'],
            style_latent=outputs['style_features_cls'],
            style_features_patches=outputs['style_features_patches'],
            output_latent=outputs['output_features_cls'],
            adain_features=outputs['fused_features_cls'],
            fused_features_patches=outputs['fused_features_patches'],
            output_images=outputs['output_images'],
            target_images=target_images,
            noise_pred=outputs['noise_pred'],
            noise_target=outputs['noise_target']
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.adain.parameters(), max_norm=1.0)
        if self.diffusion is not None:
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=1.0)
        
        optimizer.step()
        loss_dict = {k: v.item() for k, v in losses.items()}
        
        return loss_dict

    def train_epoch(self,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn, mask_ratio: None) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader providing content-style pairs
            optimizer: Optimizer
            loss_fn: Loss function
            mask_ratio: Not used in style transfer
        
        Returns:
            Dictionary of average losses for the epoch
        """
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            losses = self.train_step(batch, optimizer, loss_fn, mask_ratio)
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
        
        # Average losses
        epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
    
    @torch.no_grad()
    def generate(self,
                 content_images: torch.Tensor,
                 style_images: torch.Tensor,
                 shuffled_style_images: torch.Tensor) -> torch.Tensor:
        """
        Generate style-transferred images (inference mode)
        
        Args:
            content_images: Content images (B, C, H, W)
            style_images: Style images (B, C, H, W)
        
        Returns:
            Style-transferred images (B, C, H, W)
        """
        self.decoder.eval()
        if self.diffusion is not None:
            self.diffusion.eval()
        
        outputs = self.forward_pass(content_images, style_images, shuffled_style_images)
        return outputs['output_images']
    
    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if self.diffusion is not None:
            checkpoint['diffusion_state_dict'] = self.diffusion.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        if self.diffusion is not None and 'diffusion_state_dict' in checkpoint:
            self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint.get('epoch', 0)
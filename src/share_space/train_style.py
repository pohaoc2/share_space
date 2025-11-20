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
                     shuffled_style_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the style transfer pipeline
        
        Args:
            content_images: Content images (B, C, H, W)
            style_images: Style images (B, C, H, W)
        
        Returns:
            Dictionary containing all intermediate outputs
        """
        # Ensure float32
        content_images = content_images.float()
        style_images = style_images.float()
        shuffled_style_images = shuffled_style_images.float()
        # Step 1: Extract features (frozen)
        with torch.no_grad():
            _, content_features_cls, content_features_patches = self.extract_features(content_images)
            _, style_features_cls, style_features_patches = self.extract_features(style_images)
            _, shuffled_style_features_cls, shuffled_style_features_patches = self.extract_features(shuffled_style_images)
            _, _, mix_feature_patches = self.extract_features(0.1*content_images+style_images)

        # Step 2: Fuse features using AdaIN (Equation 5)
        fused_features_patches = self.adain(content_features_patches, shuffled_style_features_patches)
        fused_features_cls = self.adain(content_features_cls, style_features_cls)
        # Step 3: Decode to generate output

        #output_images = self.decoder(0.5*content_features_patches+style_features_patches)#fused_features_patches)
        output_images = self.decoder(fused_features_patches)
        
        # Step 4: Extract features from output for loss computation
        with torch.no_grad():
            _, output_features_cls, output_features_patches = self.extract_features(output_images)
        
        # Step 5: Optional diffusion refinement
        noise_pred = None
        noise_target = None
        
        if self.use_diffusion and self.diffusion is not None:
            # Add noise to fused features
            t = torch.randint(
                0, 
                self.diffusion.timesteps,
                (content_images.shape[0],),
                device=self.device
            )
            H = W = int(math.sqrt(fused_features_cls.shape[1]/self.diffusion.unet.in_channels))
            fused_features_cls_image = fused_features_cls.view(fused_features_cls.shape[0], self.diffusion.unet.in_channels, H, W)
            noise_target = torch.randn_like(fused_features_cls_image)
            # Get alpha values
            alpha_t = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            
            # Create noisy version
            noisy_features = sqrt_alpha_t * fused_features_cls_image + sqrt_one_minus_alpha_t * noise_target
            
            # Predict noise
            noise_pred = self.diffusion.unet(noisy_features, t)
        return {
            'content_features_cls': content_features_cls,
            'style_features_cls': style_features_cls,
            'fused_features_cls': fused_features_cls,
            'output_images': output_images,
            'output_features_cls': output_features_cls,
            'noise_pred': noise_pred,
            'noise_target': noise_target
        }
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer,
                   loss_fn, mask_ratio: None) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary with 'content' and 'style' keys
            optimizer: Optimizer for decoder (and optionally diffusion model)
            loss_fn: Loss function
            mask_ratio: Not used in style transfer
        Returns:
            Dictionary of loss values
        """
        self.decoder.train()
        if self.diffusion is not None:
            self.diffusion.train()
        
        # Get images
        content_images = batch['simulation'].to(self.device).float()
        style_images = batch['experimental'].to(self.device).float()
        shuffled_style_images = batch['shuffled_exp'].to(self.device).float()
        #content_images = batch[0].to(self.device).float()
        #style_images = batch[1].to(self.device).float()
        
        # Forward pass
        outputs = self.forward_pass(content_images, style_images, shuffled_style_images)
        
        # Compute losses
        losses = loss_fn(
            content_latent=outputs['content_features_cls'],
            style_latent=outputs['style_features_cls'],
            output_latent=outputs['output_features_cls'],
            adain_features=outputs['fused_features_cls'],
            output_images=outputs['output_images'],
            target_images=style_images,
            noise_pred=outputs['noise_pred'],
            noise_target=outputs['noise_target']
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        if self.diffusion is not None:
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Return scalar losses
        return {k: v.item() for k, v in losses.items()}
    
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
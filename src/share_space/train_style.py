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


class AdaINFusion(nn.Module):
    """
    Adaptive Instance Normalization for fusing content and style features
    Implements Equation 5 from the paper
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, content_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN to fuse content and style features
        
        Args:
            content_features: Features from content image (B, C, H, W) or (B, N, C)
            style_features: Features from style image (B, C, H, W) or (B, N, C)
        
        Returns:
            Fused features: F_adapted = γ(F_c) · σ(F_s) + μ(F_s)
        """
        # Ensure features are in (B, C, ...) format
        if content_features.dim() == 3:  # (B, N, C) -> (B, C, N)
            content_features = content_features.transpose(1, 2)
            style_features = style_features.transpose(1, 2)
            transposed = True
        else:
            transposed = False
        
        # Calculate statistics over spatial dimensions
        # For (B, C, H, W), calculate over (H, W)
        # For (B, C, N), calculate over (N,)
        dims = list(range(2, content_features.dim()))
        
        # Content normalization: γ(F_c)
        content_mean = content_features.mean(dim=dims, keepdim=True)
        content_std = content_features.std(dim=dims, keepdim=True) + 1e-5
        content_normalized = (content_features - content_mean) / content_std
        
        # Style statistics: μ(F_s) and σ(F_s)
        style_mean = style_features.mean(dim=dims, keepdim=True)
        style_std = style_features.std(dim=dims, keepdim=True) + 1e-5
        
        # Apply AdaIN: F_adapted = γ(F_c) · σ(F_s) + μ(F_s)
        fused_features = content_normalized * style_std + style_mean
        
        # Restore original shape if needed
        if transposed:
            fused_features = fused_features.transpose(1, 2)
        
        return fused_features


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
                 diffusion_weight: float = 1.0):
        super().__init__()
        self.use_diffusion = use_diffusion
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.element_weight = element_weight
        self.diffusion_weight = diffusion_weight
    
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
    
    def forward(self, 
                content_latent: torch.Tensor,
                style_latent: torch.Tensor,
                output_latent: torch.Tensor,
                adain_features: torch.Tensor,
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
        
        # Diffusion loss (Eq. 8) - optional
        if self.use_diffusion and noise_pred is not None and noise_target is not None:
            losses['diffusion'] = self.diffusion_weight * self.diffusion_loss(noise_pred, noise_target)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


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
                 feature_extractor: nn.Module,
                 decoder: nn.Module,
                 device: str = 'cuda',
                 use_diffusion: bool = True,
                 diffusion_model: Optional[nn.Module] = None,
                 content_weight: float = 1.0,
                 style_weight: float = 1.0,
                 element_weight: float = 1.0,
                 diffusion_weight: float = 1.0):
        """
        Args:
            feature_extractor: Pre-trained feature extractor (will be frozen)
            decoder: Decoder network to reconstruct images
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
        
        # Feature extractor (frozen)
        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # AdaIN fusion module
        self.adain = AdaINFusion().to(device)
        
        # Decoder (trainable)
        self.decoder = decoder.to(device)
        
        # Optional diffusion model
        if use_diffusion and diffusion_model is not None:
            self.diffusion = diffusion_model.to(device)
        else:
            self.diffusion = None
            self.use_diffusion = False
        
        # Loss function
        self.criterion = StyleTransferLoss(
            use_diffusion=self.use_diffusion,
            content_weight=content_weight,
            style_weight=style_weight,
            element_weight=element_weight,
            diffusion_weight=diffusion_weight
        ).to(device)
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using frozen feature extractor
        
        Args:
            images: Input images (B, C, H, W)
        
        Returns:
            Features in appropriate format
        """
        self.feature_extractor.eval()
        
        # Handle different feature extractor outputs
        outputs = self.feature_extractor(images)
        
        # If the feature extractor returns a tuple/dict, extract the relevant features
        if isinstance(outputs, tuple):
            # Assume format: (reconstruction, cls_token, patch_tokens)
            # We use patch_tokens for style transfer
            features = outputs[-1] if len(outputs) > 1 else outputs[0]
        elif isinstance(outputs, dict):
            # Try common keys
            features = outputs.get('features', outputs.get('patch_tokens', outputs.get('latent')))
        else:
            features = outputs
        
        return features
    
    def forward_pass(self, 
                     content_images: torch.Tensor,
                     style_images: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        
        # Step 1: Extract features (frozen)
        with torch.no_grad():
            content_features = self.extract_features(content_images)
            style_features = self.extract_features(style_images)
        
        # Step 2: Fuse features using AdaIN (Equation 5)
        fused_features = self.adain(content_features, style_features)
        
        # Step 3: Decode to generate output
        output_images = self.decoder(fused_features)
        
        # Step 4: Extract features from output for loss computation
        with torch.no_grad():
            output_features = self.extract_features(output_images)
        
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
            
            noise_target = torch.randn_like(fused_features)
            
            # Get alpha values
            alpha_t = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            
            # Create noisy version
            noisy_features = sqrt_alpha_t * fused_features + sqrt_one_minus_alpha_t * noise_target
            
            # Predict noise
            noise_pred = self.diffusion.unet(noisy_features, t)
        
        return {
            'content_features': content_features,
            'style_features': style_features,
            'fused_features': fused_features,
            'output_images': output_images,
            'output_features': output_features,
            'noise_pred': noise_pred,
            'noise_target': noise_target
        }
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary with 'content' and 'style' keys
            optimizer: Optimizer for decoder (and optionally diffusion model)
        
        Returns:
            Dictionary of loss values
        """
        self.decoder.train()
        if self.diffusion is not None:
            self.diffusion.train()
        
        # Get images
        content_images = batch['content'].to(self.device).float()
        style_images = batch['style'].to(self.device).float()
        
        # Forward pass
        outputs = self.forward_pass(content_images, style_images)
        
        # Compute losses
        losses = self.criterion(
            content_latent=outputs['content_features'],
            style_latent=outputs['style_features'],
            output_latent=outputs['output_features'],
            adain_features=outputs['fused_features'],
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
                    optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader providing content-style pairs
            optimizer: Optimizer
        
        Returns:
            Dictionary of average losses for the epoch
        """
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            losses = self.train_step(batch, optimizer)
            
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
                 style_images: torch.Tensor) -> torch.Tensor:
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
        
        outputs = self.forward_pass(content_images, style_images)
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
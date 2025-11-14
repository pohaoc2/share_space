"""
StyDiff: Complete style transfer model combining AutoKL, AdaIN, and Diffusion
Based on the paper: "StyDiff: a refined style transfer method based on diffusion models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autokl import AutoKL
from .adain import AdaINFeatureFusion
from .diffusion import DiffusionModel


class StyDiff(nn.Module):
    """
    StyDiff: Style transfer framework combining:
    1. AutoKL encoder/decoder for latent space representation
    2. AdaIN for style-content fusion
    3. Diffusion model for high-quality generation
    """
    def __init__(
        self,
        img_size=256,
        in_channels_content=3,
        in_channels_style=3,
        out_channels=3,
        latent_channels=4,
        autokl_base_channels=128,
        diffusion_model_channels=256,
        num_embeddings=8192,
        diffusion_timesteps=1000,
        unet_config=None
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels_content = in_channels_content
        self.in_channels_style = in_channels_style
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        # Use 1x1 conv to map content channels to style channels (works with 4D tensors)
        self.map_content_style = nn.Conv2d(in_channels_content, in_channels_style, kernel_size=1)
        # AutoKL for encoding/decoding
        self.autokl = AutoKL(
            in_channels_style=in_channels_style,
            out_channels=out_channels,
            latent_channels=latent_channels,
            base_channels=autokl_base_channels,
            num_embeddings=num_embeddings
        )
        
        # AdaIN for style-content fusion
        self.adain_fusion = AdaINFeatureFusion()
        
        # Diffusion model for generation
        if unet_config is None:
            unet_config = {
            'in_channels': latent_channels,
            'out_channels': latent_channels,
            'model_channels': diffusion_model_channels,
            'num_res_blocks': 2,
            'attention_resolutions': [4, 8],
            'channel_mult': (1, 2, 4),
            'num_heads': 4
        }
        
        self.diffusion = DiffusionModel(
            unet_config=unet_config,
            timesteps=diffusion_timesteps
        )
        
        # Projection layer to combine AdaIN features with latent
        self.style_projection = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # VGG conv4_2 has 512 channels
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, 3, padding=1)
        )
        
    def encode_images(self, content_img, style_img):
        """
        Encode content and style images to latent space
        
        Args:
            content_img: Content image (B, 3, H, W)
            style_img: Style image (B, 3, H, W)
        
        Returns:
            content_latent: Encoded content latent
            style_latent: Encoded style latent
            adain_features: Features from AdaIN fusion
        """
        # Encode to latent space
        content_latent, _ = self.autokl.encode(content_img)
        style_latent, _ = self.autokl.encode(style_img)
        # Extract and fuse features using AdaIN
        adapted_features, content_features, style_features = self.adain_fusion(
            content_img, style_img
        )
        
        return content_latent, style_latent, adapted_features, content_features, style_features
    
    def forward(self, content_img, style_img, return_intermediates=False):
        """
        Complete forward pass for training
        
        Args:
            content_img: Content image (B, 3, H, W)
            style_img: Style image (B, 3, H, W)
            return_intermediates: Whether to return intermediate outputs
        
        Returns:
            Dictionary containing:
            - output: Final stylized image
            - content_latent: Content latent representation
            - style_latent: Style latent representation
            - noise_pred: Predicted noise from diffusion
            - noise_target: Actual noise added
            - adapted_features: AdaIN adapted features
            - content_features: Original content features
            - style_features: Original style features
        """
        # Encode images and get AdaIN features
        # map content channels (B, C, H, W) to style channels (B, C, H, W)
        #content_img = self.map_content_style(content_img)
        content_latent, style_latent, adapted_features, content_features, style_features = \
            self.encode_images(content_img, style_img)
        # Use the last (deepest) adapted feature as style conditioning
        # Resize to match latent size
        style_condition = adapted_features[-1]  # conv4_2 features -- global style features
        
        # Resize style condition to match latent spatial dimensions
        if style_condition.shape[2:] != content_latent.shape[2:]:
            style_condition = F.interpolate(
                style_condition,
                size=content_latent.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Project style features to latent space
        style_condition_proj = self.style_projection(style_condition)
        
        # Combine content latent with style conditioning
        # A(Xs, Xi) from the paper - fusion of style and content in latent space
        fused_latent = content_latent + style_condition_proj
        
        # Apply diffusion model (forward process for training)
        noise_pred, noise_target = self.diffusion(fused_latent, style_condition=style_condition_proj)
        
        # For training, we denoise the latent
        # Reconstruct from denoised latent
        denoised_latent = fused_latent
        output = self.autokl.decode(denoised_latent)
        
        results = {
            'output': output,
            'content_latent': content_latent,
            'style_latent': style_latent,
            'fused_latent': fused_latent,
            'noise_pred': noise_pred,
            'noise_target': noise_target,
            'adapted_features': adapted_features,
            'content_features': content_features,
            'style_features': style_features,
            'style_condition': style_condition_proj
        }
        
        if not return_intermediates:
            return results['output']
        
        return results
    
    @torch.no_grad()
    def transfer_style(self, content_img, style_img, num_inference_steps=50):
        """
        Transfer style from style_img to content_img
        
        Args:
            content_img: Content image (B, 3, H, W)
            style_img: Style image (B, 3, H, W)
            num_inference_steps: Number of denoising steps
        
        Returns:
            Stylized image
        """
        # Encode images
        content_img = self.map_content_style(content_img)
        content_latent, style_latent, adapted_features, _, _ = \
            self.encode_images(content_img, style_img)
        
        # Get style conditioning
        style_condition = adapted_features[-1]
        
        # Resize to match latent size
        if style_condition.shape[2:] != content_latent.shape[2:]:
            style_condition = F.interpolate(
                style_condition,
                size=content_latent.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Project style features
        style_condition_proj = self.style_projection(style_condition)
        
        # Fuse content and style
        fused_latent = content_latent + style_condition_proj
        
        # Sample from diffusion model (could optionally start from fused_latent instead of noise)
        # For better quality, we can do iterative refinement
        generated_latent = self.diffusion.sample(
            shape=content_latent.shape,
            style_condition=style_condition_proj,
            num_inference_steps=num_inference_steps
        )
        
        # Alternatively, use the fused latent directly for faster inference
        # generated_latent = fused_latent
        
        # Decode to image
        output = self.autokl.decode(generated_latent)
        
        return output
    
    def get_latent_size(self):
        """Get the size of latent representation"""
        latent_h = self.img_size // (2 ** 4)  # 4 downsampling layers
        latent_w = self.img_size // (2 ** 4)
        return latent_h, latent_w

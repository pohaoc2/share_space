"""
AutoKL Encoder/Decoder for StyDiff framework
Based on the paper: "StyDiff: a refined style transfer method based on diffusion models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for AutoKL"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        
    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x + residual


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(32, out_channels)
        
    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(32, out_channels)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return F.silu(self.norm(self.conv(x)))


class AutoKLEncoder(nn.Module):
    """
    AutoKL Encoder - encodes images to latent space
    Inspired by VQVAE and Stable Diffusion's VAE encoder
    """
    def __init__(self, in_channels=3, latent_channels=4, base_channels=128):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList([
            DownBlock(base_channels, base_channels),
            DownBlock(base_channels, base_channels * 2),
            DownBlock(base_channels * 2, base_channels * 2),
            #DownBlock(base_channels * 4, base_channels * 4),
        ])
        
        # Middle residual blocks
        self.mid_blocks = nn.ModuleList([
            #ResidualBlock(base_channels * 4),
            #ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
        ])
        
        # Output layers
        self.norm_out = nn.GroupNorm(32, base_channels * 2)
        self.conv_out = nn.Conv2d(base_channels * 2, latent_channels, 3, padding=1)
        
    def forward(self, x):
        # Initial conv
        h = self.conv_in(x)
        
        # Downsample
        for down in self.down_blocks:
            h = down(h)
        
        # Middle blocks
        for mid in self.mid_blocks:
            h = mid(h)
        
        # Output
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        return h


class AutoKLDecoder(nn.Module):
    """
    AutoKL Decoder - decodes from latent space to images
    """
    def __init__(self, out_channels=3, latent_channels=4, base_channels=128):
        super().__init__()
        
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        
        # Input convolution
        self.conv_in = nn.Conv2d(latent_channels, base_channels * 2, 3, padding=1)
        
        # Middle residual blocks
        self.mid_blocks = nn.ModuleList([
            #ResidualBlock(base_channels * 4),
            #ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList([
            #UpBlock(base_channels * 4, base_channels * 4),
            UpBlock(base_channels * 2, base_channels * 2),
            UpBlock(base_channels * 2, base_channels),
            UpBlock(base_channels, base_channels),
        ])
        
        # Output layers
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, z):
        # Input conv
        h = self.conv_in(z)
        
        # Middle blocks
        for mid in self.mid_blocks:
            h = mid(h)
        
        # Upsample
        for up in self.up_blocks:
            h = up(h)
        
        # Output
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        
        return h


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer for discrete latent space
    Used in Q() function from the paper
    """
    def __init__(self, num_embeddings=8192, embedding_dim=4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, z):
        # Flatten input
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)
        
        # Calculate distances to embedding vectors
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Find closest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, encoding_indices


class AutoKL(nn.Module):
    """
    Complete AutoKL model with encoder, decoder, and quantizer
    """
    def __init__(self, in_channels_style=3, out_channels=3, latent_channels=4, base_channels=128, num_embeddings=8192):
        super().__init__()
        self.encoder = AutoKLEncoder(in_channels_style, latent_channels, base_channels)
        self.decoder = AutoKLDecoder(out_channels, latent_channels, base_channels)
        self.quantizer = VectorQuantizer(num_embeddings, latent_channels)
        

    def encode(self, x):
        """Encode image to latent space"""
        z = self.encoder(x)
        z_quantized, indices = self.quantizer(z)
        return z_quantized, indices
    
    def decode(self, z):
        """Decode from latent space to image"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        z_quantized, indices = self.encode(x)
        x_recon = self.decode(z_quantized)
        return x_recon, z_quantized, indices

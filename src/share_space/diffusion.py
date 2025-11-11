"""
Diffusion Model for StyDiff
Based on DDPM (Denoising Diffusion Probabilistic Models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings
    
    Args:
        timesteps: Tensor of timesteps
        embedding_dim: Dimension of the embedding
    
    Returns:
        Timestep embeddings
    """
    assert len(timesteps.shape) == 1
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    
    return emb


class TimeEmbedding(nn.Module):
    """Time embedding layer"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        
    def forward(self, t):
        t_emb = get_timestep_embedding(t, self.dim)
        t_emb = self.linear1(t_emb)
        t_emb = F.silu(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


class ResBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_emb_proj(F.silu(t_emb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        
        # Attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj_out(out)
        
        return x + out


class UNetModel(nn.Module):
    """
    U-Net architecture for diffusion model
    Conditioned on timestep and style features (from AdaIN)
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        model_channels=256,
        num_res_blocks=2,
        attention_resolutions=[8, 16],
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        time_emb_dim=None
    ):
        super().__init__()
        
        if time_emb_dim is None:
            time_emb_dim = model_channels * 4
        
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_emb_dim)]
                ch = out_ch
                if level in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_emb_dim),
        ])
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + input_block_chans.pop(), out_ch, time_emb_dim)]
                ch = out_ch
                if level in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.up_blocks.append(nn.ModuleList(layers))
            
            if level != 0:
                self.up_blocks.append(nn.ModuleList([nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)]))
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, x, timesteps, style_condition=None):
        """
        Args:
            x: Noisy latent (B, C, H, W)
            timesteps: Timestep (B,)
            style_condition: Optional style conditioning from AdaIN
        """
        # Time embedding
        t_emb = self.time_embedding(timesteps)
        
        # Input
        h = self.conv_in(x)
        
        # Downsampling
        hs = [h]
        for module_list in self.down_blocks:
            for module in module_list:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                else:
                    h = module(h)
            hs.append(h)
        
        # Middle
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # Upsampling
        for module_list in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for module in module_list:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                else:
                    h = module(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class DiffusionModel(nn.Module):
    """
    Complete diffusion model with noise scheduling
    """
    def __init__(
        self,
        unet_config=None,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02
    ):
        super().__init__()
        
        if unet_config is None:
            unet_config = {}
        
        self.unet = UNetModel(**unet_config)
        self.timesteps = timesteps
        
        # Define beta schedule
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
    
    def add_noise(self, x0, t, noise=None):
        """Add noise to clean image x0 at timestep t"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise, noise
    
    def forward(self, x0, style_condition=None):
        """
        Forward diffusion process for training
        
        Args:
            x0: Clean latent representation
            style_condition: Style conditioning from AdaIN
        
        Returns:
            Predicted noise and actual noise
        """
        batch_size = x0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=x0.device).long()
        
        # Add noise
        xt, noise = self.add_noise(x0, t)
        
        # Predict noise
        noise_pred = self.unet(xt, t, style_condition)
        
        return noise_pred, noise
    
    @torch.no_grad()
    def sample(self, shape, style_condition=None, num_inference_steps=50):
        """
        Sample from the diffusion model (reverse process)
        
        Args:
            shape: Shape of the latent to generate
            style_condition: Style conditioning
            num_inference_steps: Number of denoising steps
        
        Returns:
            Generated latent
        """
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Reverse diffusion
        timesteps = torch.linspace(self.timesteps - 1, 0, num_inference_steps, device=device).long()
        
        for t in timesteps:
            t_batch = t.repeat(shape[0])
            
            # Predict noise
            noise_pred = self.unet(x, t_batch, style_condition)
            
            # Denoise
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[max(0, t - self.timesteps // num_inference_steps)]
            
            beta_t = 1 - alpha_t / alpha_t_prev
            
            # Mean
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            mean = torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t) * pred_x0 + \
                   torch.sqrt(self.alphas[t]) * (1 - alpha_t_prev) / (1 - alpha_t) * x
            
            # Add noise
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        
        return x

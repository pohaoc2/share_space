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
    """Sinusoidal timestep embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)
    
    def forward(self, timesteps):
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP
        emb = self.linear1(emb)
        emb = F.silu(emb)
        emb = self.linear2(emb)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
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
        t = self.time_mlp(F.silu(t_emb))
        h = h + t[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        h, _ = self.attention(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class DownBlock(nn.Module):
    """Downsampling block with residual blocks and optional attention"""
    def __init__(self, in_ch, out_ch, time_emb_dim, num_res_blocks, 
                 has_attention=False, num_heads=8, has_downsample=True):
        super().__init__()
        
        # Build residual blocks
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            in_channels = in_ch if i == 0 else out_ch
            self.res_blocks.append(ResBlock(in_channels, out_ch, time_emb_dim))
            
            if has_attention:
                self.attention_blocks.append(AttentionBlock(out_ch, num_heads))
            else:
                self.attention_blocks.append(None)
        
        # Downsampling
        if has_downsample:
            self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x, t_emb):
        """Returns (output, skip_connection)"""
        h = x
        
        for res_block, attn_block in zip(self.res_blocks, self.attention_blocks):
            h = res_block(h, t_emb)
            if attn_block is not None:
                h = attn_block(h)
        
        # Store skip connection before downsampling
        skip = h
        
        if self.downsample is not None:
            h = self.downsample(h)
        
        return h, skip


class UpBlock(nn.Module):
    """Upsampling block with residual blocks and optional attention"""
    def __init__(self, in_ch, out_ch, time_emb_dim, num_res_blocks,
                 has_attention=False, num_heads=8, has_upsample=True):
        super().__init__()
        
        # Build residual blocks
        self.res_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            # First block takes concatenated input (after upsampling and skip connection)
            # If upsampling: out_ch (upsampled) + out_ch (skip) = 2 * out_ch
            # If no upsampling: in_ch (x) + out_ch (skip) = in_ch + out_ch
            # Subsequent blocks: out_ch channels
            if i == 0:
                if has_upsample:
                    res_in_ch = 2 * out_ch  # upsampled + skip
                else:
                    res_in_ch = in_ch + out_ch  # x + skip
            else:
                res_in_ch = out_ch
            self.res_blocks.append(ResBlock(res_in_ch, out_ch, time_emb_dim))
            
            if has_attention:
                self.attention_blocks.append(AttentionBlock(out_ch, num_heads))
            else:
                self.attention_blocks.append(None)
        
        # Upsampling: takes in_ch input and outputs out_ch to match skip connection
        if has_upsample:
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        else:
            self.upsample = None
    
    def forward(self, x, t_emb, skip_connection):
        """
        Args:
            x: Current feature map
            t_emb: Time embedding
            skip_connection: Skip connection from corresponding DownBlock
        """
        # Upsample first if needed
        if self.upsample is not None:
            h = self.upsample(x)
        else:
            h = x
        
        # Concatenate with skip connection
        # Handle dimension mismatch by interpolating if needed
        if h.shape[2:] != skip_connection.shape[2:]:
            h = F.interpolate(h, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
        
        h = torch.cat([h, skip_connection], dim=1)
        
        # Process through residual blocks
        for res_block, attn_block in zip(self.res_blocks, self.attention_blocks):
            h = res_block(h, t_emb)
            if attn_block is not None:
                h = attn_block(h)
        
        return h


class UNetModel(nn.Module):
    """
    Improved U-Net architecture for diffusion model
    Conditioned on timestep and optional style features
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        model_channels=256,
        num_res_blocks=2,
        attention_resolutions=[],  # List of levels (0-indexed) to add attention
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        time_emb_dim=None
    ):
        super().__init__()
        
        if time_emb_dim is None:
            time_emb_dim = model_channels * 4
        
        self.time_embedding = TimeEmbedding(time_emb_dim)
        # TimeEmbedding outputs time_emb_dim * 4, so we need to use that for ResBlocks
        self.time_emb_dim_output = time_emb_dim * 4
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            has_attention = level in attention_resolutions
            has_downsample = level != len(channel_mult) - 1
            
            self.down_blocks.append(DownBlock(
                in_ch=ch,
                out_ch=out_ch,
                time_emb_dim=self.time_emb_dim_output,
                num_res_blocks=num_res_blocks,
                has_attention=has_attention,
                num_heads=num_heads,
                has_downsample=has_downsample
            ))
            ch = out_ch
        
        # Middle block
        self.middle_res1 = ResBlock(ch, ch, self.time_emb_dim_output)
        self.middle_attn = AttentionBlock(ch, num_heads) if len(attention_resolutions) > 0 else None
        self.middle_res2 = ResBlock(ch, ch, self.time_emb_dim_output)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            has_attention = level in attention_resolutions
            has_upsample = level != 0
            
            self.up_blocks.append(UpBlock(
                in_ch=ch,
                out_ch=out_ch,
                time_emb_dim=self.time_emb_dim_output,
                num_res_blocks=num_res_blocks,
                has_attention=has_attention,
                num_heads=num_heads,
                has_upsample=has_upsample
            ))
            ch = out_ch
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, self.out_channels, 3, padding=1)
    
    def forward(self, x, timesteps, style_condition=None):
        """
        Args:
            x: Noisy latent (B, C, H, W)
            timesteps: Timestep (B,) or scalar
            style_condition: Optional style conditioning (not yet implemented)
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Ensure timesteps is a tensor
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor([timesteps], device=x.device)
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        if len(timesteps) == 1 and x.shape[0] > 1:
            timesteps = timesteps.expand(x.shape[0])
        
        # Time embedding
        t_emb = self.time_embedding(timesteps.float())
        
        # Input convolution
        h = self.conv_in(x)
        
        # Downsampling - collect skip connections
        skip_connections = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb)
            skip_connections.append(skip)
        
        # Middle block
        h = self.middle_res1(h, t_emb)
        if self.middle_attn is not None:
            h = self.middle_attn(h)
        h = self.middle_res2(h, t_emb)
        
        # Reverse skip connections for upsampling (last down block -> first up block)
        skip_connections = skip_connections[::-1]
        
        # Upsampling
        for up_block, skip in zip(self.up_blocks, skip_connections):
            h = up_block(h, t_emb, skip)
        
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
    def sample(self, shape, style_condition=None, num_inference_steps=50, init_latent=None, t_start=None):
        """
        Sample from the diffusion model (reverse process)
        
        Args:
            shape: Shape of the latent to generate
            style_condition: Style conditioning (B, C, H, W)
            num_inference_steps: Number of denoising steps
            init_latent: Optional initial latent to start from (e.g., fused_latent)
            t_start: Timestep to start from (if None, starts from pure noise at t=T-1)
                    Use smaller values to start from partially noised init_latent
                    Example: t_start=num_inference_steps//4 means start from 75% denoised
        
        Returns:
            Generated latent
        """
        device = next(self.parameters()).device
        
        # Determine starting point
        if init_latent is not None:
            # Start from provided latent
            x = init_latent.clone()
            
            # If t_start is provided, add noise to init_latent
            if t_start is not None:
                # Map t_start (inference step) to actual timestep
                start_timestep = self.timesteps - 1 - (t_start * self.timesteps // num_inference_steps)
                start_timestep = max(0, min(start_timestep, self.timesteps - 1))
                
                # Add noise to init_latent according to forward process
                noise = torch.randn_like(x)
                alpha_start = self.alphas_cumprod[start_timestep]
                x = torch.sqrt(alpha_start) * x + torch.sqrt(1 - alpha_start) * noise
                
                # Start denoising from this timestep
                timesteps = torch.linspace(start_timestep, 0, num_inference_steps - t_start, device=device).long()
            else:
                # Use init_latent as-is, denoise from T-1
                timesteps = torch.linspace(self.timesteps - 1, 0, num_inference_steps, device=device).long()
        else:
            # Start from pure noise (original behavior)
            x = torch.randn(shape, device=device)
            timesteps = torch.linspace(self.timesteps - 1, 0, num_inference_steps, device=device).long()
        
        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(shape[0])
            
            # Predict noise
            noise_pred = self.unet(x, t_batch, style_condition)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            
            # Determine previous alpha
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            # Compute beta_t
            beta_t = 1 - alpha_t / alpha_t_prev
            
            # Predict x0 (the clean latent)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Clip pred_x0 for stability (optional but recommended)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute mean of posterior q(x_{t-1} | x_t, x_0)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
            
            # Direction pointing to x_t
            dir_xt = (1 - alpha_t_prev - beta_t) / sqrt_one_minus_alpha_t * x
            
            # Compute mean
            mean = sqrt_alpha_t_prev * beta_t / (1 - alpha_t) * pred_x0 + dir_xt
            
            # Add noise (except for the last step)
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
        
        return x

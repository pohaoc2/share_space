"""
Trainer for Latent Diffusion Model
Handles training of diffusion model in latent space with frozen feature extractors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, Callable
import wandb
from datetime import datetime


class LatentDiffusionTrainer:
    """
    Trainer for latent diffusion model that translates simulation to experimental latents.
    
    Args:
        diffusion_model: The DiffusionModel from diffusion.py
        sim_feature_extractor: Feature extractor for simulation images (frozen)
        exp_feature_extractor: Feature extractor for experimental images (frozen)
        train_loader: DataLoader with simulation and experimental image pairs
        val_loader: Optional validation DataLoader
        lr: Learning rate
        device: Device to train on
        output_dir: Directory to save checkpoints and logs
        use_wandb: Whether to use Weights & Biases for logging
        compile_model: Whether to use torch.compile for speedup (PyTorch 2.0+)
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        sim_feature_extractor: nn.Module,
        exp_feature_extractor: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        embedding_dim: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda',
        output_dir: str = './outputs/diffusion',
        use_wandb: bool = False,
        compile_model: bool = False,
        gradient_clip: float = 1.0,
        ema_decay: float = 0.9999,
        loss_type: str = 'mse',  # 'mse', 'l1', or 'huber'
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.gradient_clip = gradient_clip
        self.loss_type = loss_type
        
        # Models
        self.diffusion_model = diffusion_model.to(device)
        self.sim_feature_extractor = sim_feature_extractor.to(device)
        self.exp_feature_extractor = exp_feature_extractor.to(device)
        self.embedding_dim = embedding_dim
        # Freeze feature extractors
        self.sim_feature_extractor.eval()
        self.exp_feature_extractor.eval()
        for param in self.sim_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.exp_feature_extractor.parameters():
            param.requires_grad = False
        self.input_proj = nn.Conv2d(embedding_dim * 2, self.diffusion_model.unet.in_channels, 1) if embedding_dim * 2 != self.diffusion_model.unet.in_channels else nn.Identity()
        self.output_proj = nn.Conv2d(self.diffusion_model.unet.out_channels, embedding_dim, 1) if self.diffusion_model.unet.out_channels != embedding_dim else nn.Identity()
        # Compile models for faster training (PyTorch 2.0+)
        if compile_model:
            try:
                print("Compiling models with torch.compile...")
                self.diffusion_model = torch.compile(self.diffusion_model)
                print("✓ Model compiled successfully")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = AdamW(
            self.diffusion_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # EMA model for better sampling quality
        self.ema_decay = ema_decay
        self.ema_model = self._create_ema_model()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
        
        # Loss function
        self.loss_fn = self._get_loss_function()
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Output directory: {output_dir}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Loss type: {loss_type}")
    
    def _get_loss_function(self) -> Callable:
        """Get the loss function based on loss_type"""
        if self.loss_type == 'mse':
            return F.mse_loss
        elif self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _create_ema_model(self):
        """Create EMA model for better sampling quality"""
        import copy
        ema_model = copy.deepcopy(self.diffusion_model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _update_ema(self):
        """Update EMA model parameters"""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.diffusion_model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    model_param.data, alpha=1 - self.ema_decay
                )
    
    @torch.no_grad()
    def extract_latents(self, sim_images: torch.Tensor, exp_images: torch.Tensor):
        """
        Extract latent representations from images using frozen feature extractors.
        
        Args:
            sim_images: Simulation images [B, C, H, W]
            exp_images: Experimental images [B, C, H, W]
            
        Returns:
            sim_latents, exp_latents: Latent representations
        """
        # Extract latents
        _, sim_latents = self.sim_feature_extractor(sim_images)
        _, exp_latents = self.exp_feature_extractor(exp_images)
        
        return sim_latents, exp_latents
    
    def compute_loss(
        self,
        exp_latents: torch.Tensor,
        sim_latents: torch.Tensor,
        style_condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.
        
        The model learns to denoise experimental latents, conditioned on simulation latents.
        
        Args:
            exp_latents: Target experimental latents [B, num_patches, embedding_dim]
            sim_latents: Conditioning simulation latents [B, num_patches, embedding_dim]
            style_condition: Optional style conditioning
            
        Returns:
            Dictionary with loss values
        """
        batch_size = exp_latents.shape[0]
        num_patches = exp_latents.shape[1]
        embedding_dim = exp_latents.shape[2]
        # Calculate spatial dimensions (assuming square patches)
        H = W = int(num_patches ** 0.5)
        assert H * W == num_patches, f"num_patches ({num_patches}) must be a perfect square"
        
        # Sample random timesteps for each sample in the batch
        t = torch.randint(
            0, self.diffusion_model.timesteps, 
            (batch_size,), 
            device=self.device
        ).long()
        
        # Add noise to experimental latents (forward diffusion)
        noise = torch.randn_like(exp_latents)
        
        # Adjust reshaping for 3D tensors [B, num_patches, embedding_dim]
        sqrt_alpha_t = self.diffusion_model.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.diffusion_model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        noisy_exp_latents = sqrt_alpha_t * exp_latents + sqrt_one_minus_alpha_t * noise
        
        # Reshape to 4D: [B, num_patches, embedding_dim] -> [B, embedding_dim, H, W]
        noisy_exp_latents_4d = noisy_exp_latents.transpose(1, 2).reshape(batch_size, embedding_dim, H, W)
        sim_latents_4d = sim_latents.transpose(1, 2).reshape(batch_size, embedding_dim, H, W)
        # Conditioning: concatenate along channel dimension
        # Shape: [B, embedding_dim * 2, H, W]
        conditioned_input = torch.cat([noisy_exp_latents_4d, sim_latents_4d], dim=1)
        
        # If embedding_dim * 2 != unet.in_channels, add a projection layer
        if hasattr(self, 'input_proj'):
            conditioned_input = self.input_proj(conditioned_input)
        
        # Predict the noise
        noise_pred = self.diffusion_model.unet(conditioned_input, t, style_condition)
        
        if hasattr(self, 'output_proj'):
            noise_pred = self.output_proj(noise_pred)
        
        # Reshape noise_pred back to [B, num_patches, embedding_dim]
        noise_pred = noise_pred.reshape(batch_size, embedding_dim, -1).transpose(1, 2)        
        # Compute loss
        loss = self.loss_fn(noise_pred, noise)
        return {
            'loss': loss,
            'mse_loss': F.mse_loss(noise_pred, noise),
            'mae_loss': F.l1_loss(noise_pred, noise)
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'simulation' and 'experimental' keys containing images
            
        Returns:
            Dictionary with loss values
        """
        self.diffusion_model.train()
        
        # Move data to device
        sim_images = batch['simulation'].to(self.device)
        exp_images = batch['experimental'].to(self.device)
        
        # Extract latents (frozen feature extractors)
        # Shape: [B, num_patches, embedding_dim]
        with torch.no_grad():
            sim_latents, exp_latents = self.extract_latents(sim_images, exp_images)
        
        
        # Forward pass and compute loss
        loss_dict = self.compute_loss(exp_latents, sim_latents)
        loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.diffusion_model.parameters(), 
                self.gradient_clip
            )
        
        self.optimizer.step()
        
        # Update EMA model
        self._update_ema()
        
        # Convert to float for logging
        return {k: v.item() for k, v in loss_dict.items()}


    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validation step.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.diffusion_model.eval()
        
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            sim_images = batch['simulation'].to(self.device)
            exp_images = batch['experimental'].to(self.device)
            
            # Extract latents
            sim_latents, exp_latents = self.extract_latents(sim_images, exp_images)
            
            # Compute loss
            loss_dict = self.compute_loss(exp_latents, sim_latents)
            
            total_loss += loss_dict['loss'].item()
            total_mse += loss_dict['mse_loss'].item()
            total_mae += loss_dict['mae_loss'].item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_mse': total_mse / num_batches,
            'val_mae': total_mae / num_batches
        }
    
    @torch.no_grad()
    def sample_translations(
        self,
        sim_images: torch.Tensor,
        num_inference_steps: int = 50,
        use_ema: bool = True
    ) -> torch.Tensor:
        """
        Generate experimental latents from simulation images.
        
        Args:
            sim_images: Simulation images [B, C, H, W]
            num_inference_steps: Number of denoising steps
            use_ema: Whether to use EMA model for sampling
            
        Returns:
            Generated experimental latents [B, num_patches, embedding_dim]
        """
        model = self.ema_model if use_ema else self.diffusion_model
        model.eval()
        
        # Extract simulation latents: [B, num_patches, embedding_dim]
        _, sim_latents = self.sim_feature_extractor(sim_images)
        
        batch_size = sim_latents.shape[0]
        num_patches = sim_latents.shape[1]
        embedding_dim = sim_latents.shape[2]
        
        # Calculate spatial dimensions
        H = W = int(num_patches ** 0.5)
        assert H * W == num_patches, f"num_patches ({num_patches}) must be a perfect square"
        
        # Reshape to 4D: [B, num_patches, embedding_dim] -> [B, embedding_dim, H, W]
        sim_latents_4d = sim_latents.transpose(1, 2).reshape(batch_size, embedding_dim, H, W)
        
        # Don't project here - projection happens inside the denoising loop with concatenation
        # Just use sim_latents_4d as conditioning
        
        # Initialize with noise or simulation latents
        # The shape should match what UNet expects (after input_proj)
        unet_channels = self.diffusion_model.unet.in_channels
        init_latent = torch.randn(batch_size, unet_channels, H, W, device=self.device)
        
        # Option: Start from simulation latents (need to project first)
        # init_latent = self.input_proj(torch.cat([sim_latents_4d, sim_latents_4d], dim=1))
        
        # Custom sampling loop with conditioning
        exp_latents_4d = self._sample_with_conditioning(
            model=model,
            sim_latents_4d=sim_latents_4d,
            init_latent=init_latent,
            num_inference_steps=num_inference_steps,
            t_start=num_inference_steps // 2
        )
        
        # Project back from UNet output channels
        exp_latents_4d = self.output_proj(exp_latents_4d)
        
        # Reshape back to patch format: [B, embedding_dim, H, W] -> [B, num_patches, embedding_dim]
        exp_latents = exp_latents_4d.reshape(batch_size, embedding_dim, -1).transpose(1, 2)
        
        return exp_latents

    def _sample_with_conditioning(
        self,
        model,
        sim_latents_4d: torch.Tensor,
        init_latent: torch.Tensor,
        num_inference_steps: int,
        t_start: Optional[int] = None
    ) -> torch.Tensor:
        """
        Custom sampling loop that applies conditioning via concatenation.
        
        Args:
            model: Diffusion model
            sim_latents_4d: Simulation latents [B, embedding_dim, H, W]
            init_latent: Initial latent [B, unet_channels, H, W]
            num_inference_steps: Number of denoising steps
            t_start: Starting timestep
        
        Returns:
            Denoised latent [B, unet_channels, H, W]
        """
        device = sim_latents_4d.device
        batch_size = sim_latents_4d.shape[0]
        
        # Start from init_latent
        x = init_latent.clone()
        
        # Generate timestep schedule
        if t_start is not None:
            start_timestep = model.timesteps - 1 - (t_start * model.timesteps // num_inference_steps)
            start_timestep = max(0, min(start_timestep, model.timesteps - 1))
            timesteps = torch.linspace(start_timestep, 0, num_inference_steps - t_start, device=device).long()
        else:
            timesteps = torch.linspace(model.timesteps - 1, 0, num_inference_steps, device=device).long()
        
        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # Project x back to embedding_dim space for concatenation
            x_embedding = self.output_proj(x)
            
            # Concatenate with simulation latents: [B, embedding_dim*2, H, W]
            conditioned = torch.cat([x_embedding, sim_latents_4d], dim=1)
            
            # Project to UNet input channels
            conditioned = self.input_proj(conditioned)
            
            # Predict noise
            noise_pred = model.unet(conditioned, t_batch, None)
            
            # Get alpha values
            alpha_t = model.alphas_cumprod[t]
            
            # Determine previous alpha
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t_prev = model.alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            # Compute beta_t
            beta_t = 1 - alpha_t / alpha_t_prev
            
            # Predict x0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            pred_x0 = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute mean
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
            dir_xt = (1 - alpha_t_prev - beta_t) / sqrt_one_minus_alpha_t * x
            mean = sqrt_alpha_t_prev * beta_t / (1 - alpha_t) * pred_x0 + dir_xt
            
            # Add noise (except for last step)
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
        
        return x    
    def train(
        self,
        num_epochs: int,
        scheduler: Optional[Any] = None,
        log_interval: int = 10,
        save_interval: int = 1,
        validate_interval: int = 1,
        sample_interval: Optional[int] = None,
        num_samples: int = 4
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            scheduler: Optional learning rate scheduler
            log_interval: Log every N steps
            save_interval: Save checkpoint every N epochs
            validate_interval: Validate every N epochs
            sample_interval: Generate samples every N epochs (None to disable)
            num_samples: Number of samples to generate
        """
        if self.use_wandb:
            wandb.init(
                project="latent-diffusion",
                config={
                    "epochs": num_epochs,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "batch_size": self.train_loader.batch_size,
                    "loss_type": self.loss_type
                }
            )
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_losses = []
            
            # Training loop
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                # Training step
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict['loss'])
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Logging
                if self.global_step % log_interval == 0:
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': loss_dict['loss'],
                            'train/mse': loss_dict['mse_loss'],
                            'train/mae': loss_dict['mae_loss'],
                            'train/lr': self.optimizer.param_groups[0]['lr'],
                            'epoch': epoch,
                            'step': self.global_step
                        })
            
            # Epoch statistics
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)
            
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Validation
            if self.val_loader and (epoch + 1) % validate_interval == 0:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics['val_loss'])
                
                print(f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                      f"MSE: {val_metrics['val_mse']:.4f}, "
                      f"MAE: {val_metrics['val_mae']:.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        'val/loss': val_metrics['val_loss'],
                        'val/mse': val_metrics['val_mse'],
                        'val/mae': val_metrics['val_mae'],
                        'epoch': epoch
                    })
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')
                    print(f"  ✓ New best model saved (val_loss: {self.best_val_loss:.4f})")
            
            # Sample translations
            if sample_interval and (epoch + 1) % sample_interval == 0:
                self.generate_and_log_samples(num_samples)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
            
            print("-" * 80)
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Final training loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        if self.use_wandb:
            wandb.finish()
    
    @torch.no_grad()
    def generate_and_log_samples(self, num_samples: int = 4):
        """Generate and log sample translations"""
        # Get a batch from validation set
        if self.val_loader is None:
            return
        
        batch = next(iter(self.val_loader))
        sim_images = batch['simulation'][:num_samples].to(self.device)
        exp_images = batch['experimental'][:num_samples].to(self.device)
        
        # Generate translations
        generated_latents = self.sample_translations(sim_images)
        
        # Extract target latents for comparison
        with torch.no_grad():
            _, target_latents = self.extract_latents(sim_images, exp_images)
        
        # Compute metrics between generated and target latents
        mse = F.mse_loss(generated_latents, target_latents).item()
        mae = F.l1_loss(generated_latents, target_latents).item()
        
        print(f"  Sample generation - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'samples/mse': mse,
                'samples/mae': mae,
                'epoch': self.current_epoch
            })
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.diffusion_model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  ✓ Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"✓ Checkpoint loaded from: {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}, step {self.global_step}")


def create_trainer_from_config(config: Dict[str, Any]) -> LatentDiffusionTrainer:
    """
    Create trainer from configuration dictionary.
    
    Example config:
    {
        'diffusion_model': diffusion_model,
        'sim_feature_extractor': sim_encoder,
        'exp_feature_extractor': exp_encoder,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'lr': 1e-4,
        'device': 'cuda',
        'output_dir': './outputs',
        'use_wandb': False
    }
    """
    return LatentDiffusionTrainer(**config)
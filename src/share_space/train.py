# train.py (Updated with dtype fixes)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
import copy
from tqdm import tqdm
from share_space.diffusion import DiffusionModel
import math

class TeacherStudentTrainer:
    """Teacher-Student training following DINOv2 approach"""
    def __init__(self, student_model: nn.Module, device: str = 'cuda', 
                 teacher_momentum: float = 0.996, center_momentum: float = 0.9,
                 diffusion_model: DiffusionModel = None, use_diffusion: bool = False):
        self.device = device
        self.student = student_model.to(device)
        self.teacher = copy.deepcopy(student_model).to(device)
        self.diffusion = diffusion_model.to(device)
        self.use_diffusion = use_diffusion
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher_momentum = teacher_momentum
        self.center_momentum = center_momentum
        
        # Center for teacher output (helps stabilize training) - ensure float32
        self.register_center = torch.zeros(1, student_model.encoder.embed_dim, dtype=torch.float32).to(device)
    
    @torch.no_grad()
    def update_teacher(self):
        """EMA update of teacher network"""
        for param_student, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            param_teacher.data = param_teacher.data * self.teacher_momentum + \
                               param_student.data * (1 - self.teacher_momentum)
    
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Update center used for teacher output"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.register_center = self.register_center * self.center_momentum + \
                              batch_center * (1 - self.center_momentum)
    
    def forward_pass(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through student and teacher
        
        Args:
            images: input images (B, C, H, W)
            masks: optional masks for masked reconstruction (B, num_patches)
        
        Returns:
            dict with reconstruction, features, etc.
        """
        # Ensure float32
        images = images.float()
        if masks is not None:
            masks = masks.float()
        
        # Student forward
        student_recon, student_cls, student_patches = self.student(images, masks)
        # Teacher forward (no gradient, no mask)
        with torch.no_grad():
            teacher_recon, teacher_cls, teacher_patches = self.teacher(images, None)
        
        return {
            'student_recon': student_recon,
            'student_cls': student_cls,
            'student_patches': student_patches,
            'teacher_recon': teacher_recon,
            'teacher_cls': teacher_cls,
            'teacher_patches': teacher_patches
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
                loss_fn, mask_ratio: float = 0.5) -> Dict[str, float]:
        """Single training step - processes both simulation and experimental images"""
        self.student.train()
        self.teacher.eval()
        
        sim_images = batch['simulation'].to(self.device).float()
        exp_images = batch['experimental'].to(self.device).float()

        # Store accumulated losses
        accumulated_losses = {}
        
        # Process simulation images
        sim_losses = self._process_images(
            images=sim_images,
            target_images=sim_images,
            mask_ratio=mask_ratio,
            loss_fn=loss_fn,
            prefix='sim'
        )
        
        # Process experimental images
        exp_losses = self._process_images(
            images=exp_images,
            target_images=exp_images,  # Or sim_images if you want reverse mapping
            mask_ratio=mask_ratio,
            loss_fn=loss_fn,
            prefix='exp'
        )
        
        # Combine losses
        total_loss = sum(sim_losses.values()) + sum(exp_losses.values())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update teacher (once per step)
        self.update_teacher()
        
        # Combine and return all losses
        accumulated_losses.update(sim_losses)
        accumulated_losses.update(exp_losses)
        accumulated_losses['total'] = total_loss.item()
        
        return accumulated_losses


    def _process_images(self, images: torch.Tensor, target_images: torch.Tensor,
                    mask_ratio: float, loss_fn, prefix: str = '') -> Dict[str, float]:
        """Process a batch of images through the model and compute losses
        
        Args:
            images: Input images
            target_images: Target images for reconstruction
            mask_ratio: Masking ratio for patches
            loss_fn: Loss function
            prefix: Prefix for loss keys (e.g., 'sim' or 'exp')
        
        Returns:
            Dictionary of losses with prefix
        """
        B, C, H, W = images.shape
        
        # Handle patch size
        patch_embed = self.student.encoder.patch_embed
        patch_size = patch_embed.patch_size
        
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size
        
        grid_h = H // patch_h
        grid_w = W // patch_w
        num_patches = grid_h * grid_w
        
        # Create random mask
        mask = (torch.rand(B, num_patches, device=self.device, dtype=torch.float32) < mask_ratio).float()
        
        # Forward pas
        outputs = self.forward_pass(images, mask)
        if self.use_diffusion:
            latent_vector = outputs['teacher_cls']
            
            # Reshape for diffusion
            H_latent = W_latent = int(math.sqrt(latent_vector.shape[1] / self.diffusion.unet.in_channels))
            latent_vector = latent_vector.view(B, self.diffusion.unet.in_channels, H_latent, W_latent)
            # Diffusion process
            t = torch.randint(
                0, 
                self.diffusion.timesteps, 
                (B,),
                device=self.device
            )
            
            noise = torch.randn_like(latent_vector)
            alpha_t = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            
            noisy_latent = sqrt_alpha_t * latent_vector + sqrt_one_minus_alpha_t * noise
            noise_pred = self.diffusion.unet(noisy_latent, t)
        else:
            noisy_latent = None
            noise_pred = None
        # Compute losses
        losses = loss_fn(outputs, target_images, mask, patch_size, noisy_latent, noise_pred)
        
        # Update teacher center (accumulate from both passes)
        self.update_center(outputs['teacher_cls'])
        
        # Add prefix to loss keys
        if prefix:
            losses = {f'{prefix}_{k}': v for k, v in losses.items()}
        
        return losses
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    loss_fn, mask_ratio: float = 0.5) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_losses = {}
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            losses = self.train_step(batch, optimizer, loss_fn, mask_ratio)
            
            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
        
        # Average losses
        num_batches = len(dataloader)
        epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return epoch_losses
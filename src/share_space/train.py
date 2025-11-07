# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
import copy
from tqdm import tqdm

class TeacherStudentTrainer:
    """Teacher-Student training following DINOv2 approach"""
    def __init__(self, student_model: nn.Module, device: str = 'cuda', 
                 teacher_momentum: float = 0.996, center_momentum: float = 0.9):
        self.device = device
        self.student = student_model.to(device)
        self.teacher = copy.deepcopy(student_model).to(device)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher_momentum = teacher_momentum
        self.center_momentum = center_momentum
        
        # Center for teacher output (helps stabilize training)
        self.register_center = torch.zeros(1, student_model.encoder.embed_dim).to(device)
    
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
        """Single training step"""
        self.student.train()
        self.teacher.eval()
        
        sim_images = batch['simulation'].to(self.device)
        exp_images = batch['experimental'].to(self.device)
        
        B, C, H, W = sim_images.shape
        num_patches = (H // self.student.encoder.patch_embed.patch_size) ** 2
        
        # Create random mask
        mask = torch.rand(B, num_patches, device=self.device) < mask_ratio
        
        # Forward pass
        outputs = self.forward_pass(sim_images, mask)
        
        # Compute losses
        losses = loss_fn(outputs, exp_images, mask)
        total_loss = sum(losses.values())
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update teacher
        self.update_teacher()
        self.update_center(outputs['teacher_cls'])
        
        return {k: v.item() for k, v in losses.items()}
    
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

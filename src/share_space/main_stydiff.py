"""
StyDiff Training Script
Based on: "StyDiff: a refined style transfer method based on diffusion models"

This replaces the teacher-student architecture with StyDiff framework.
"""

import yaml
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from share_space.losses import StyDiffLoss
from share_space.metrics import StyDiffMetrics
from share_space.stydiff import StyDiff
from share_space.dataset import get_real_dataloaders

# State names for visualization (from original code)
state_names = {
    1: 'OTHER',
    2: 'INFLAMMATORY',
    3: 'HEALTHY_EPITHELIAL',
    4: 'DYSPLASTIC/MALIGNANT',
    5: 'FIBROBLAST',
    6: 'MUSCLE',
    7: 'ENDOTHELIAL'
}


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train for one epoch
    
    Args:
        model: StyDiff model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Dictionary of average losses
    """
    model.train()
    
    total_losses = {
        'total': 0.0,
        'content': 0.0,
        'style': 0.0,
        'element': 0.0,
        'diffusion': 0.0,
        'perceptual': 0.0,
        'style_feature': 0.0
    }
    
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        content_img = batch['simulation'].to(device)
        style_img = batch['experimental'].to(device)
        
        # Forward pass
        outputs = model(content_img, style_img, return_intermediates=True)
        
        # Get output image and encode it for loss calculation
        output_img = outputs['output']
        output_latent, _, _, _ = model.autokl.encode(output_img, style_img)
        
        # Extract VGG features for output
        output_features = model.adain_fusion.vgg_extractor(
            (output_img - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) / 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )
        
        # Calculate losses
        losses = criterion(
            content_latent=outputs['content_latent'],
            style_latent=outputs['style_latent'],
            output_latent=output_latent,
            fused_latent=outputs['fused_latent'],
            noise_pred=outputs['noise_pred'],
            noise_target=outputs['noise_target'],
            content_features=outputs['content_features'],
            style_features=outputs['style_features'],
            output_features=output_features
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses.keys():
            total_losses[key] += losses[key].item()
        
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {losses['total'].item():.4f}")
    
    # Average losses
    avg_losses = {key: val / num_batches for key, val in total_losses.items()}
    
    return avg_losses


@torch.no_grad()
def evaluate(model, dataloader, metrics_evaluator, device):
    """
    Evaluate the model
    
    Args:
        model: StyDiff model
        dataloader: Validation dataloader
        metrics_evaluator: Metrics calculator
        device: Device
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_metrics = {
        'ssim': [],
        'gm': [],
        'lpips': [],
        'pd': []
    }
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 20:  # Limit evaluation to save time
            break
        
        content_img = batch['simulation'].to(device)
        style_img = batch['experimental'].to(device)
        
        # Generate stylized image
        generated_img = model.transfer_style(content_img, style_img, num_inference_steps=20)
        
        # Clamp to [0, 1]
        generated_img = torch.clamp(generated_img, 0, 1)
        
        # Calculate metrics
        metrics = metrics_evaluator.evaluate_batch(model.map_content_style(content_img), style_img, generated_img)
        
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {
        key: np.mean(values) for key, values in all_metrics.items()
    }
    
    return avg_metrics


def visualize_results(model, dataloader, device, save_path='stydiff_results.png', num_samples=3):
    """
    Visualize style transfer results
    
    Args:
        model: StyDiff model
        dataloader: Dataloader
        device: Device
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    batch = next(iter(dataloader))
    content_imgs = batch['simulation'][:num_samples].to(device)
    style_imgs = batch['experimental'][:num_samples].to(device)
    
    with torch.no_grad():
        generated_imgs = model.transfer_style(content_imgs, style_imgs, num_inference_steps=20)
        generated_imgs = torch.clamp(generated_imgs, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
    
    for i in range(num_samples):
        # Content
        print(f"content_imgs shape: {content_imgs[i].shape}")
        axes[i, 0].imshow(content_imgs[i].cpu().permute(1, 2, 0)[..., 0].numpy())
        axes[i, 0].set_title('Content' if i == 0 else '')
        axes[i, 0].axis('off')
        # Mapped content
        axes[i, 1].imshow(model.map_content_style(content_imgs[i]).cpu().permute(1, 2, 0).detach().numpy())
        axes[i, 1].set_title('Content' if i == 0 else '')
        axes[i, 1].axis('off')
        # Style
        axes[i, 2].imshow(style_imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 2].set_title('Style' if i == 0 else '')
        axes[i, 2].axis('off')
        
        # Generated
        axes[i, 3].imshow(generated_imgs[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 3].set_title('Generated' if i == 0 else '')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {save_path}")


def main(config_path='config_stydiff.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    print("Creating StyDiff model...")
    model = StyDiff(
        img_size=config['model']['img_size'],
        in_channels_content=config['model']['in_channels_content'],
        in_channels_style=config['model']['in_channels_style'],
        out_channels=config['model']['out_channels'],
        latent_channels=config['model'].get('latent_channels', 4),
        autokl_base_channels=config['model'].get('autokl_base_channels', 128),
        diffusion_model_channels=config['model'].get('diffusion_model_channels', 256),
        num_embeddings=config['model'].get('num_embeddings', 8192),
        diffusion_timesteps=config['model'].get('diffusion_timesteps', 1000)
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create loss function
    criterion = StyDiffLoss(
        content_weight=config['loss'].get('content_weight', 1.0),
        style_weight=config['loss'].get('style_weight', 1.0),
        element_weight=config['loss'].get('element_weight', 1.0),
        diffusion_weight=config['loss'].get('diffusion_weight', 1.0),
        perceptual_weight=config['loss'].get('perceptual_weight', 0.1),
        style_feature_weight=config['loss'].get('style_feature_weight', 1.0)
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['learning_rate'] * 0.01
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = get_real_dataloaders(
        exp_dir=config['data']['exp_dir'],
        sim_dir=config['data']['sim_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['model']['img_size'],
        train_split=config['training']['train_split']
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    # Create metrics evaluator
    metrics_evaluator = StyDiffMetrics(device=device)

    # Training loop
    if config['training'].get('eval_only', False):
        print("Loading best model for evaluation...")
        checkpoint = torch.load(config['checkpoint']['load_path'], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Starting training...")
        best_ssim = float('-inf')
        
        loss_history = {
            'total': [],
            'content': [],
            'style': [],
            'element': [],
            'diffusion': []
        }
        
        for epoch in range(config['training']['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
            
            # Train
            train_losses = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch + 1
            )
            
            print(f"Train losses: {train_losses}")
            
            # Store losses
            for key in loss_history.keys():
                if key in train_losses:
                    loss_history[key].append(train_losses[key])
            
            # Evaluate
            if (epoch + 1) % config['checkpoint']['eval_every'] == 0:
                print("Evaluating...")
                metrics = evaluate(model, val_loader, metrics_evaluator, device)
                print(f"Validation metrics: {metrics}")
                
                # Save best model
                if metrics['ssim'] > best_ssim:
                    best_ssim = metrics['ssim']
                    save_path = Path(config['checkpoint']['save_dir']) / 'best_model.pth'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics,
                        'config': config
                    }, save_path)
                    print(f"Saved best model with SSIM: {best_ssim:.4f}")
            
            # Step scheduler
            scheduler.step()
        
        print("Training complete!")
        
        # Plot loss history
        if len(loss_history['total']) > 0:
            plt.figure(figsize=(12, 5))
            epochs = range(1, len(loss_history['total']) + 1)
            
            plt.plot(epochs, loss_history['total'], label='Total Loss', marker='o')
            plt.plot(epochs, loss_history['content'], label='Content Loss', marker='s')
            plt.plot(epochs, loss_history['style'], label='Style Loss', marker='^')
            plt.plot(epochs, loss_history['element'], label='Element Loss', marker='d')
            plt.plot(epochs, loss_history['diffusion'], label='Diffusion Loss', marker='*')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses Over Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('stydiff_training_losses.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate(model, val_loader, metrics_evaluator, device)
    print(f"Final metrics: {final_metrics}")
    
    # Save final metrics
    metrics_path = Path(config['checkpoint']['save_dir']) / 'final_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(
        model, val_loader, device,
        save_path='stydiff_results.png',
        num_samples=3
    )


if __name__ == '__main__':
    import sys
    
    # Allow optional config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config_stydiff.yaml'
    
    print(f"Loading configuration from: {config_path}")
    main(config_path)

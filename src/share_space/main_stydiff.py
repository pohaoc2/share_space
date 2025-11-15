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
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from share_space.losses import StyDiffLoss
from share_space.metrics import StyDiffMetrics
from share_space.stydiff import StyDiff
from share_space.dataset_real import get_real_dataloaders

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
        # content_img = batch['simulation'].to(device)
        # style_img = batch['experimental'].to(device)
        content_img = batch[0].to(device)
        style_img = batch[1].to(device)
        # Forward pass
        outputs = model(content_img, style_img, return_intermediates=True)
        
        # Get output image and encode it for loss calculation
        output_img = outputs['output']
        output_latent, _ = model.autokl.encode(output_img)
        
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
        
        #content_img = batch['simulation'].to(device)
        #style_img = batch['experimental'].to(device)
        content_img = batch[0].to(device)
        style_img = batch[1].to(device)
        
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

def unnormalize_imagenet(imgs, device='cpu'):
    """
    Reverse ImageNet normalization: T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    Args:
        imgs: Tensor of shape (B, 3, H, W) in normalized range
        device: Device to put tensors on
    
    Returns:
        Tensor in [0, 1] range
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    imgs = imgs * std + mean
    imgs = torch.clamp(imgs, 0, 1)
    
    return imgs

def visualize_results(model, dataloader, num_samples=4, device='cuda', save_path='results.png'):
    """
    Visualize style transfer results in a grid.
    """
    batch = next(iter(dataloader))
    content_imgs = batch[0][:num_samples].to(device)
    style_imgs = batch[1][:num_samples].to(device)
    
    with torch.no_grad():
        generated_imgs = model.transfer_style(content_imgs, style_imgs, num_inference_steps=100)
        mapped_content = torch.cat([
            model.map_content_style(content_imgs[i:i+1]) for i in range(num_samples)
        ], dim=0)
    
    # Unnormalize
    content_vis = unnormalize_imagenet(content_imgs, device)
    style_vis = unnormalize_imagenet(style_imgs, device)
    generated_vis = unnormalize_imagenet(generated_imgs, device)
    mapped_vis = unnormalize_imagenet(mapped_content, device)
    
    # Create grid: each row is [content, mapped, style, generated]
    all_imgs = []
    for i in range(num_samples):
        all_imgs.extend([content_vis[i], mapped_vis[i], style_vis[i], generated_vis[i]])
    
    grid = make_grid(all_imgs, nrow=4, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=(12, 3 * num_samples))
    plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
    plt.axis('off')
    
    # Add column labels
    titles = ['Content', 'Content (mapped)', 'Style', 'Generated']
    for idx, title in enumerate(titles):
        plt.text(
            (idx + 0.5) / 4, 0.02, title,
            transform=plt.gca().transAxes,
            fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return grid


def main(config_path='config_stydiff.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Create model
    print(config['model'].get('unet_config', None))
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
        diffusion_timesteps=config['model'].get('diffusion_timesteps', 1000),
        unet_config=config['model'].get('unet_config', None)
    ).to(device)
    # Print trainable and non-trainable parameters and total parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Non-trainable parameters: {non_trainable_params / 1e6:.2f}M")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
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
    # Visualize the first batch of the training data
    if 0:
        print("Visualizing first batch of the training data...")
        first_batch = next(iter(train_loader))
        n_samples = 8
        fig, ax = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
        for i in range(n_samples):
            # Reverse transform: T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            first_batch[0][i] = first_batch[0][i] * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            first_batch[0][i] = torch.clamp(first_batch[0][i], 0, 1)
            first_batch[1][i] = first_batch[1][i] * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            first_batch[1][i] = torch.clamp(first_batch[1][i], 0, 1)
            ax[0, i].imshow(first_batch[0][i].permute(1, 2, 0).cpu().numpy())
            ax[1, i].imshow(first_batch[1][i].permute(1, 2, 0).cpu().numpy())
            if i == 0:
                # Keep y-axis visible for first column to show labels
                ax[0, i].set_ylabel("COCO", rotation=90, fontsize=12)
                ax[1, i].set_ylabel("WikiArt", rotation=90, fontsize=12)
                ax[0, i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                ax[1, i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            else:
                ax[0, i].axis('off')
                ax[1, i].axis('off')
        plt.tight_layout()
        plt.show()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    # Create metrics evaluator
    metrics_evaluator = StyDiffMetrics(device=device)
    # Training loop
    if config['training'].get('eval_only', False):
        print("Loading best model for evaluation...")
        checkpoint = torch.load(config['checkpoint']['load_path'], weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Starting training...")
        best_ssim = float('-inf')
        
        loss_history = {
            'total': [],
            'content': [],
            'style': [],
            'element': [],
            'diffusion': [],
            'perceptual': [],
            'style_feature': []
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
            plt.plot(epochs, loss_history['perceptual'], label='Perceptual Loss', marker='x')
            plt.plot(epochs, loss_history['style_feature'], label='Style Feature Loss', marker='v')
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses Over Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('stydiff_training_losses.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Final evaluation
    if 0:
        print("\nFinal evaluation...")
        final_metrics = evaluate(model, val_loader, metrics_evaluator, device)
        print(f"Final metrics: {final_metrics}")
        
        # Save final metrics
        metrics_path = Path(config['checkpoint']['save_dir']) / 'final_metrics.json'
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
    # 3. Check actual loss values (not weighted)
    with torch.no_grad():
        batch = next(iter(val_loader))
        content = batch[0][:4].to(device)
        style = batch[1][:4].to(device)
        outputs = model(content, style, return_intermediates=True)
        # Get output image and encode it for loss calculation
        output_img = outputs['output']
        output_latent, _ = model.autokl.encode(output_img)
        
        # Extract VGG features for output
        output_features = model.adain_fusion.vgg_extractor(
            (output_img - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) / 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )
        # Get raw loss values
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
        print(f"Raw content loss: {losses['content']}")
        print(f"Raw style loss: {losses['style']}")
        print(f"Raw element loss: {losses['element']}")
        print(f"Raw diffusion loss: {losses['diffusion']}")
        print(f"Raw perceptual loss: {losses['perceptual']}")
        print(f"Raw style feature loss: {losses['style_feature']}")
        print(f"Raw total loss: {losses['total']}")
    # Visualize results

    visualize_results(
        model, val_loader, device=device,
        save_path='stydiff_results.png',
        num_samples=3
    )

if __name__ == '__main__':
    import sys
    
    # Allow optional config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config_stydiff.yaml'
    
    print(f"Loading configuration from: {config_path}")
    main(config_path)

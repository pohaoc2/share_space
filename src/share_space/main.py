import yaml
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from share_space.models.sim2exp_model import Sim2ExpModel
from share_space.train import TeacherStudentTrainer
from share_space.loss import Sim2ExpLoss
from share_space.dataset import get_dummy_dataloaders, get_real_dataloaders
from share_space.evaluation import MetricsEvaluator

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

def main(config_path='config.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model configuration
    encoder_config = {
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'in_chans': config['model']['in_chans'],
        'embed_dim': config['model']['embed_dim'],
        'depth': config['model']['depth'],
        'num_heads': config['model']['num_heads'],
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0
    }

    decoder_config = {
        'embed_dim': config['model']['embed_dim'],
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'out_chans': config['model']['out_chans']
    }
    
    # Create model
    print("Creating model...")
    model = Sim2ExpModel(
        encoder_config=encoder_config,
        decoder_type=config['model']['decoder_type'],
        decoder_config=decoder_config
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create trainer
    trainer = TeacherStudentTrainer(
        student_model=model,
        device=device,
        teacher_momentum=config['training']['teacher_momentum']
    )
    
    # Create loss function
    loss_fn = Sim2ExpLoss(
        recon_weight=config['loss']['recon_weight'],
        distill_weight=config['loss']['distill_weight'],
        mask_weight=config['loss']['mask_weight']
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
    if 0:#True:
        print("Visualizing first batch of the training data...")
        first_batch = next(iter(train_loader))
        fig, ax = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(5):
            ax[0, i].imshow(first_batch['simulation'][i].permute(1, 2, 0).cpu().numpy())
            ax[1, i].imshow(first_batch['experimental'][i].permute(1, 2, 0).cpu().numpy())
            ax[0, i].axis('off')
            ax[1, i].axis('off')
        plt.show()
        plt.tight_layout()
    
    # Create evaluator
    evaluator = MetricsEvaluator(device=device)
    
    # Training loop
    print("Starting training...")
    best_psnr = float('-inf')
    
    # Track losses across epochs
    loss_history = {
        'recon': [],
        'distill_cls': [],
        'mask_recon': []
    }
    
    if config['training']['eval_only']: # load the best model and evaluate
        print("Loading best model...")
        model.load_state_dict(torch.load(config['checkpoint']['load_path'], weights_only=False)['model_state_dict'])
    else:
        for epoch in range(config['training']['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
            
            # Train
            train_losses = trainer.train_epoch(
                train_loader, optimizer, loss_fn, mask_ratio=config['training']['mask_ratio']
            )
            
            print(f"Train losses: {train_losses}")
            
            # Store losses for plotting
            for key in loss_history.keys():
                if key in train_losses:
                    loss_history[key].append(train_losses[key])
            
            # Validate
            if (epoch + 1) % config['checkpoint']['eval_every'] == 0:
                print("Evaluating...")
                metrics = evaluator.evaluate_model(model, val_loader)
                print(f"Validation metrics: {metrics}")
                
                # Save best model
                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    save_path = Path(config['checkpoint']['save_dir']) / 'best_model.pth'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': metrics
                    }, save_path)
                    print(f"Saved best model with PSNR: {best_psnr:.4f}")
            
            # Step scheduler
            scheduler.step()
        
        print("Training complete!")
    
    # Final evaluation
    if 1:
        print("\nFinal evaluation...")
        final_metrics = evaluator.evaluate_model(model, val_loader)
        print(f"Final metrics: {final_metrics}")
        
        # Save final metrics
        metrics_path = Path(config['checkpoint']['save_dir']) / 'final_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Visualize loss history
        if not config['training']['eval_only'] and len(loss_history['recon']) > 0:
            plt.figure(figsize=(12, 5))
            epochs = range(1, len(loss_history['recon']) + 1)
            plt.plot(epochs, loss_history['recon'], label='Reconstruction Loss', marker='o')
            if len(loss_history['distill_cls']) > 0:
                plt.plot(epochs, loss_history['distill_cls'], label='Distillation Loss', marker='s')
            if len(loss_history['mask_recon']) > 0:
                plt.plot(epochs, loss_history['mask_recon'], label='Masked Reconstruction Loss', marker='^')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Losses Over Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # Visualize the reconstructed images with the original images
    n_viz = 3
    print(f"Visualizing {n_viz} samples of reconstructed images in the first batch of the validation data...")
    first_batch = next(iter(val_loader))
    fig, ax = plt.subplots(n_viz, 10, figsize=(3 * 8, 3 * n_viz))
    for i in range(n_viz):
        exp_img = first_batch['experimental'][i].permute(1, 2, 0).cpu().numpy()
        sim_img = first_batch['simulation'][i]
        pred_imgs = trainer.forward_pass(sim_img[torch.newaxis, ...], None)['student_recon'][0].permute(1, 2, 0).cpu().detach().numpy()
        ax[i, 0].imshow(exp_img)
        ax[i, 1].imshow(pred_imgs)
        ax[i, 1].axis('off')
        if i == 0:
            ax[i, 1].set_title('Reconstructed\n(Output)', loc='center')
        for j in range(1, 9):
            ax[i, j+1].imshow(sim_img.permute(1, 2, 0).cpu().detach().numpy()[..., j-1])
            ax[i, j+1].axis('off')
            if i == 0 and j == 1:
                ax[i, j+1].set_title(f'Cell Count', loc='center')
            elif i == 0:
                ax[i, j+1].set_title(f'{state_names[j-1]}', loc='center')
        if i == 0:
            ax[i, 0].set_title('Experimental\n(Target)', loc='center')
        ax[i, 0].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01, wspace=0.05)
    plt.savefig('reconstructed_images.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    
if __name__ == '__main__':
    import sys
    
    # Allow optional config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    
    print(f"Loading configuration from: {config_path}")
    main(config_path)
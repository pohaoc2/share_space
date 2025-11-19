import yaml
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from share_space.models.sim2exp_model import Sim2ExpModel, StyleTransferModel
from share_space.train import TeacherStudentTrainer
from share_space.loss import Sim2ExpLoss, StyleTransferLoss
from share_space.dataset import get_dummy_dataloaders, get_real_dataloaders
#from share_space.dataset_real import get_real_dataloaders
from share_space.evaluation import MetricsEvaluator
from share_space.train_style import StyleTransferTrainer
from share_space.models.decoders import ConvDecoder, TransformerDecoder
import copy
from share_space.diffusion import DiffusionModel
from share_space.models.adain import AdaINFusion
import os
state_names = {
    0: 'Cell Count',
    1: 'OTHER',
    2: 'INFLAMMATORY',
    3: 'HEALTHY_EPITHELIAL',
    4: 'DYSPLASTIC_MALIGNANT',
    5: 'FIBROBLAST',
    6: 'MUSCLE',
    7: 'ENDOTHELIAL'
}

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_stage(model, trainer, optimizer, scheduler, loss_fn, train_loader, val_loader, 
                evaluator, stage_config, stage_name, best_psnr, device, 
                run_final_eval=False):
    """
    Train a single stage of the model.
    
    Args:
        model: The model to train
        trainer: Trainer instance
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        evaluator: Metrics evaluator
        stage_config: Configuration dictionary for this stage
        stage_name: Name of the stage (e.g., 'stage_1', 'stage_2')
        best_psnr: Current best PSNR value
        device: Device to run on
        run_final_eval: Whether to run final evaluation and visualization
        
    Returns:
        Updated best_psnr value
    """
    loss_history = {}
    if stage_config['eval_only']:  # load the best model and evaluate
        print(f"Loading best model for {stage_name} from {stage_config['load_path']}...")
        model.load_state_dict(torch.load(stage_config['load_path'], map_location=device, weights_only=False)['model_state_dict'])
        model.to(device)  # Ensure model is on the correct device
    else:
        for epoch in range(stage_config['epochs']):
            print(f"\n{stage_name.upper()} - Epoch {epoch + 1}/{stage_config['epochs']}")
            
            # Train
            train_losses = trainer.train_epoch(
                train_loader, optimizer, loss_fn, mask_ratio=stage_config['mask_ratio']
            )
            print(f"Train losses: {train_losses}")
            
            # Store losses for plotting
            for key in train_losses.keys():
                loss_history.setdefault(key, []).append(train_losses[key])
            
            # Validate
            if (epoch + 1) % stage_config['eval_every'] == 0:
                print("Evaluating...")
                metrics = evaluator.evaluate_model(model, val_loader, stage_name)
                print(f"Validation metrics: {metrics}")
                
                # Save best model
                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    save_path = Path(stage_config['save_dir']) / 'best_model.pth'
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
        
        print(f"Training {stage_name} complete!")
    
    # Final evaluation
    if run_final_eval:
        print(f"\nFinal evaluation for {stage_name}...")
        final_metrics = evaluator.evaluate_model(model, val_loader, stage_name=stage_name)
        print(f"Final metrics: {final_metrics}")
        
        # Save final metrics
        metrics_path = Path(stage_config['save_dir']) / f'final_metrics_{stage_name}.json'
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Visualize loss history
        for key, values in loss_history.items():
            if isinstance(values[0], torch.Tensor):
                loss_history[key] = [value.detach().cpu().numpy() for value in values]
            else:
                loss_history[key] = values
        if not stage_config['eval_only']:
            plt.figure(figsize=(8, 2))
            epochs = range(1, len(loss_history[list(loss_history.keys())[0]]) + 1)
            for key in loss_history.keys():
                plt.plot(epochs, loss_history[key], label=key, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{stage_name.upper()} - Training Losses Over Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(Path(stage_config['save_dir']) / f'loss_history_{stage_name}.png', dpi=300, bbox_inches='tight')
    
    return best_psnr, loss_history

def _get_stage_1_model(config, device, stage_name):
    # Model configuration
    encoder_config = {
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'sim_chans': config['model']['sim_chans'],
        'exp_chans': config['model']['exp_chans'],
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
        'sim_chans': config['model']['sim_chans'],
        'exp_chans': config['model']['exp_chans']
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
        teacher_momentum=config['training'][stage_name]['teacher_momentum'],
        diffusion_model=DiffusionModel(
            unet_config=config['diffusion']['unet_config'],
            timesteps=config['diffusion']['timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end']
        )
    )
    
    # Create loss function
    loss_fn = Sim2ExpLoss(
        recon_weight=config['loss']['recon_weight'],
        distill_weight=config['loss']['distill_weight'],
        mask_weight=config['loss']['mask_weight'],
        diffusion_weight=config['loss']['diffusion_weight']
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.decoder.parameters(), # if use_diffusion, also include diffusion model parameters
        lr=config['training'][stage_name]['learning_rate'],
        weight_decay=config['training'][stage_name]['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training'][stage_name]['epochs'],
        eta_min=config['training'][stage_name]['learning_rate'] * 0.01
    )
    evaluator = MetricsEvaluator(device=device)
    return model, trainer, optimizer, scheduler, loss_fn, evaluator

def _get_stage_2_model(config, device, stage_name, feature_extractor):
    decoder_config = {
        'embed_dim': config['model']['embed_dim'],
        'img_size': config['model']['img_size'],
        'patch_size': config['model']['patch_size'],
        'out_chans': config['model']['exp_chans']
    }
    loss_fn = StyleTransferLoss(
        use_diffusion=config['training'][stage_name]['use_diffusion'],
        content_weight=config['loss']['content_weight'],
        style_weight=config['loss']['style_weight'],
        element_weight=config['loss']['element_weight'],
        diffusion_weight=config['loss']['diffusion_weight']
    )
    style_model = StyleTransferModel(
        feature_extractor=feature_extractor,
        decoder=ConvDecoder(**decoder_config) if config['model']['decoder_type'] == 'conv' else TransformerDecoder(**decoder_config),
        adain=AdaINFusion()
    )
    if config['training'][stage_name]['use_diffusion']:
        diffusion_model = DiffusionModel(
            unet_config=config['diffusion']['unet_config'],
            timesteps=config['diffusion']['timesteps'],
            beta_start=config['diffusion']['beta_start'],
            beta_end=config['diffusion']['beta_end']
        )
    else:
        diffusion_model = None
    trainer_style = StyleTransferTrainer(
        model=style_model,
        device=device,
        use_diffusion=config['training'][stage_name]['use_diffusion'],
        diffusion_model=diffusion_model,
    )
    optimizer = optim.AdamW(
        list(style_model.decoder.parameters()) + (list(diffusion_model.parameters()) if config['training'][stage_name]['use_diffusion'] and diffusion_model is not None else []),
        lr=config['training'][stage_name]['learning_rate'],
        weight_decay=config['training'][stage_name]['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training'][stage_name]['epochs'],
        eta_min=config['training'][stage_name]['learning_rate'] * 0.01
    )

    evaluator = MetricsEvaluator(device=device)
    return style_model, trainer_style, optimizer, scheduler, loss_fn, evaluator


def main(config_path='config.yaml'):
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = get_real_dataloaders(
        exp_dir=config['data']['exp_dir'],
        sim_dir=config['data']['sim_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['model']['img_size'],
        train_split=config['data']['train_split']
    )

    # Visualize the first batch of the training data
    if 0:#True:
        print("Visualizing first batch of the training data...")
        first_batch = next(iter(val_loader))
        fig, ax = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(5):
            #ax[0, i].imshow(first_batch['simulation'][i].permute(1, 2, 0).cpu().numpy())
            #ax[1, i].imshow(first_batch['experimental'][i].permute(1, 2, 0).cpu().numpy())
            ax[0, i].imshow(first_batch[0][i].permute(1, 2, 0).cpu().numpy())
            ax[1, i].imshow(first_batch[1][i].permute(1, 2, 0).cpu().numpy())
            ax[0, i].axis('off')
            ax[1, i].axis('off')
        plt.tight_layout()
    
    
    # Train stage 1
    model, trainer, optimizer, scheduler, loss_fn, evaluator = _get_stage_1_model(config, device, "stage_1")
    print("Starting training...")
    best_psnr = float('-inf')
        
    # Train stage 1
    best_psnr, loss_history = train_stage(
        model=model,
        trainer=trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        stage_config=config['training']['stage_1'],
        stage_name='stage_1',
        best_psnr=best_psnr,
        device=device,
        run_final_eval=True  # Set to True to run final evaluation and visualization
    )
    style_model, trainer_style, optimizer, scheduler, loss_fn, evaluator = _get_stage_2_model(config, device, "stage_2", model)
    best_psnr = float('-inf')
    best_psnr = train_stage(
        model=style_model,
        trainer=trainer_style,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        stage_config=config['training']['stage_2'],
        stage_name='stage_2',
        best_psnr=best_psnr,
        device=device,
        run_final_eval=True  # Set to True to run final evaluation and visualization
    )
    visualize_reconstructed_images(model,
        val_loader,
        device,
        save_dir=config['visualization']['reconstructed_images']['save_dir'],
        n_viz=config['visualization']['reconstructed_images']['n_viz'],
        states=list(state_names.keys())
    )
    visualize_style_transfer(style_model,
        val_loader,
        device,
        save_dir=config['visualization']['style_transfer']['save_dir'],
        n_viz=config['visualization']['style_transfer']['n_viz'],
        states=list(state_names.keys())
    )

def visualize_reconstructed_images(model, val_loader, device, save_dir, n_viz=5, states=list(state_names.keys())):
    print(f"Visualizing reconstructed images...")
    model.eval()
    first_batch = next(iter(val_loader))
    fig, ax = plt.subplots(n_viz, 4, figsize=(3 * 4, 3 * n_viz))
    
    exp_imgs = first_batch['experimental']
    sim_imgs = first_batch['simulation']
    pred_sim_imgs, _, _ = model(sim_imgs.to(device))
    pred_exp_imgs, _, _ = model(exp_imgs.to(device))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for state in states:
        for i in range(n_viz):
            ax[i, 0].imshow(torch.clamp(exp_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy())
            ax[i, 1].imshow(torch.clamp(pred_exp_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy()[..., :3])
            ax[i, 2].imshow(torch.clamp(sim_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy()[..., state])
            ax[i, 3].imshow(torch.clamp(pred_sim_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy()[..., state])
            ax[i, 0].axis('off')
            ax[i, 1].axis('off')
            ax[i, 2].axis('off')
            ax[i, 3].axis('off')
            if i == 0:
                ax[i, 0].set_title('Exp (Input)', loc='center')
                ax[i, 1].set_title('Exp Reconstructed (Output)', loc='center')
                ax[i, 2].set_title('Sim (Input)', loc='center')
                ax[i, 3].set_title(f'Sim Reconstructed (Output) {state_names[state]}', loc='center')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.05)
        plt.savefig(Path(save_dir) / f'reconstructed_images_{state_names[state]}.png', dpi=300, bbox_inches='tight', transparent=True)

def visualize_style_transfer(style_model, val_loader, device, save_dir, n_viz=5, states=list(state_names.keys())):
    print(f"Visualizing style transfer...")
    style_model.eval()
    first_batch = next(iter(val_loader))
    fig, ax = plt.subplots(n_viz, 3, figsize=(3 * 3, 3 * n_viz))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for state in states:
        for i in range(n_viz):
            exp_img = first_batch['experimental'][i].unsqueeze(0) # style
            sim_img = first_batch['simulation'][i].unsqueeze(0) # content
            pred_imgs = style_model(sim_img.to(device), exp_img.to(device))[0].permute(1, 2, 0).cpu().detach()

            ax[i, 0].imshow(torch.clamp(exp_img[0].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy())
            ax[i, 1].imshow(torch.clamp(sim_img[0].permute(1, 2, 0)[..., state].cpu().detach(), 0, 1).cpu().numpy())
            ax[i, 2].imshow(torch.clamp(pred_imgs, 0, 1).cpu().numpy())
            ax[i, 0].axis('off')
            ax[i, 1].axis('off')
            ax[i, 2].axis('off')
            if i == 0:
                ax[i, 0].set_title('Exp (Style)', loc='center')
                ax[i, 1].set_title(f'Sim (Content) {state_names[state]}', loc='center')
                ax[i, 2].set_title('Style Transferred\n(Output)', loc='center')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.05)

        plt.savefig(Path(save_dir) / f'style_transfer_{state_names[state]}.png', dpi=300, bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    import sys
    
    # Allow optional config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    
    print(f"Loading configuration from: {config_path}")
    main(config_path)
# main.py
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from share_space.models.sim2exp_model import Sim2ExpModel
from share_space.train import TeacherStudentTrainer
from share_space.loss import Sim2ExpLoss
from share_space.dataset import get_dataloaders
from share_space.evaluation import MetricsEvaluator


def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model configuration
    encoder_config = {
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'in_chans': 3,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0
    }
    
    # Create model
    print("Creating model...")
    model = Sim2ExpModel(
        encoder_config=encoder_config,
        decoder_type=args.decoder_type
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create trainer
    trainer = TeacherStudentTrainer(
        student_model=model,
        device=device,
        teacher_momentum=args.teacher_momentum
    )
    
    # Create loss function
    loss_fn = Sim2ExpLoss(
        recon_weight=args.recon_weight,
        distill_weight=args.distill_weight,
        mask_weight=args.mask_weight
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        img_size=args.img_size
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
    
    if args.eval_only: # load the best model and evaluate
        print("Loading best model...")
        model.load_state_dict(torch.load(args.load_path, weights_only=False)['model_state_dict'])
    else:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_losses = trainer.train_epoch(
                train_loader, optimizer, loss_fn, mask_ratio=args.mask_ratio
            )
            
            print(f"Train losses: {train_losses}")
            
            # Store losses for plotting
            for key in loss_history.keys():
                if key in train_losses:
                    loss_history[key].append(train_losses[key])
            
            # Validate
            if (epoch + 1) % args.eval_every == 0:
                print("Evaluating...")
                metrics = evaluator.evaluate_model(model, val_loader)
                print(f"Validation metrics: {metrics}")
                
                # Save best model
                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    save_path = Path(args.save_dir) / 'best_model.pth'
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
        metrics_path = Path(args.save_dir) / 'final_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        # Visualize loss history
        if not args.eval_only and len(loss_history['recon']) > 0:
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
    fig, ax = plt.subplots(n_viz, 3, figsize=(5, 5 * n_viz / 3))
    for i in range(n_viz):
        sim_imgs = first_batch['simulation'][i]
        exp_imgs = first_batch['experimental'][i]
        pred_imgs = trainer.forward_pass(sim_imgs.unsqueeze(0), None)['student_recon'][0]
        # Remove batch dimension for plotting
        ax[i, 0].imshow(exp_imgs.permute(1, 2, 0).cpu().numpy())
        ax[i, 1].imshow(sim_imgs.permute(1, 2, 0).cpu().numpy())
        ax[i, 2].imshow(pred_imgs.permute(1, 2, 0).cpu().detach().numpy())
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
        if i == 0:
            ax[i, 0].set_title('Experimental\n(Target)', loc='center')
            ax[i, 1].set_title('Simulation\n(Input)', loc='center')
            ax[i, 2].set_title('Reconstructed\n(Output)', loc='center')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Sim2Exp model')
    
    # Model architecture
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--decoder_type', type=str, default='conv', choices=['conv', 'transformer'])
    
    # Training
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--teacher_momentum', type=float, default=0.996)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--mask_weight', type=float, default=1.0)
    
    # Data
    parser.add_argument('--train_samples', type=int, default=50)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--load_path', type=str, default='./checkpoints/best_model.pth')
    
    # Misc
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    main(args)
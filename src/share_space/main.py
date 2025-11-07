# main.py
import torch
import torch.optim as optim
from models.sim2exp_model import Sim2ExpModel
from train import TeacherStudentTrainer
from loss import Sim2ExpLoss
from data.dataset import get_dataloaders
from evaluation import MetricsEvaluator
import argparse
from pathlib import Path
import json

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
        val_samples=args.val_samples
    )
    
    # Create evaluator
    evaluator = MetricsEvaluator(device=device)
    
    # Training loop
    print("Starting training...")
    best_fid = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_losses = trainer.train_epoch(
            train_loader, optimizer, loss_fn, mask_ratio=args.mask_ratio
        )
        
        print(f"Train losses: {train_losses}")
        
        # Validate
        if (epoch + 1) % args.eval_every == 0:
            print("Evaluating...")
            metrics = evaluator.evaluate_model(model, val_loader)
            print(f"Validation metrics: {metrics}")
            
            # Save best model
            if metrics['fid'] < best_fid:
                best_fid = metrics['fid']
                save_path = Path(args.save_dir) / 'best_model.pth'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, save_path)
                print(f"Saved best model with FID: {best_fid:.4f}")
        
        # Step scheduler
        scheduler.step()
    
    print("Training complete!")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluator.evaluate_model(model, val_loader)
    print(f"Final metrics: {final_metrics}")
    
    # Save final metrics
    metrics_path = Path(args.save_dir) / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)


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
    parser.add_argument('--epochs', type=int, default=100)
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
    parser.add_argument('--train_samples', type=int, default=5000)
    parser.add_argument('--val_samples', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Misc
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    main(args)
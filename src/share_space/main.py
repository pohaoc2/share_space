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
from share_space.trainer_diffusion import LatentDiffusionTrainer
from share_space.models.adain import AdaINFusion#, HistoAdaIN
from share_space.adain_histo import HistoAdaINSpatialAware, HistoAdaIN, HistoAdaINHybrid, WeightedAdaIN, LearnableAdaIN, FusionModule
from share_space.latent_adapter import LatentDomainAdapter
from share_space.latent_adapter import train_epoch as train_latent_adapter_epoch
import os
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from share_space.evaluation import frechet_permutation_test, frechet_distance
state_names = {
    0: 'Cell Count',
    # 1: 'OTHER',
    # 2: 'INFLAMMATORY',
    # 3: 'HEALTHY_EPITHELIAL',
    # 4: 'DYSPLASTIC_MALIGNANT',
    # 5: 'FIBROBLAST',
    # 6: 'MUSCLE',
    # 7: 'ENDOTHELIAL'
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
    if stage_config['run_final_eval']:
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
        if not stage_config['eval_only'] and len(loss_history) > 0:
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

def visualize_pca(*feature_sets, save_dir=None, labels=None, colors=None, markers=None):
    """
    Create PCA visualization of multiple feature sets.
    
    Args:
        *feature_sets: Variable number of feature tensors, each of shape [N, D]
        save_dir: Directory to save the visualization
        labels: Optional list of labels for each feature set. If None, uses default labels.
        colors: Optional list of colors for each feature set. If None, uses default colors.
        markers: Optional list of markers for each feature set. If None, uses default markers.
    
    Returns:
        Dictionary containing PCA results:
        - explained_variance_ratio: List of explained variance ratios
        - latents_pca: Dictionary mapping feature set index/label to PCA-projected latents
    
    Examples:
        # Two feature sets (content and style)
        visualize_pca(content_features_cls, style_features_cls, save_dir=save_dir)
        
        # Three feature sets (content, style, reconstructed)
        visualize_pca(content_features_cls, style_features_cls, reconstructed_features_cls, 
                     save_dir=save_dir, labels=['Content', 'Style', 'Reconstructed'])
    """
    if len(feature_sets) == 0:
        raise ValueError("At least one feature set must be provided")
    
    # Default styling
    default_labels = ['Feature Set 1', 'Feature Set 2', 'Feature Set 3', 'Feature Set 4', 'Feature Set 5']
    default_colors = ['blue', 'red', 'green', 'orange', 'purple']
    default_markers = ['o', 's', '^', 'D', 'v']
    
    if labels is None:
        labels = default_labels[:len(feature_sets)]
    if colors is None:
        colors = default_colors[:len(feature_sets)]
    if markers is None:
        markers = default_markers[:len(feature_sets)]
    
    # Ensure we have enough labels, colors, and markers
    while len(labels) < len(feature_sets):
        labels.append(default_labels[len(labels)])
    while len(colors) < len(feature_sets):
        colors.append(default_colors[len(colors)])
    while len(markers) < len(feature_sets):
        markers.append(default_markers[len(markers)])
    
    # Convert all features to numpy and combine for PCA fitting
    feature_arrays = []
    for features in feature_sets:
        if isinstance(features, torch.Tensor):
            feature_arrays.append(features.numpy())
        else:
            feature_arrays.append(features)
    
    all_latents = np.vstack(feature_arrays)  # [Total_N, D]
    
    # Fit PCA on combined latents
    pca = PCA(n_components=2)
    pca.fit(all_latents)
    
    # Project all feature sets to PCA space
    latents_pca = {}
    latents_pca_np = {}
    for i, (features_np, label) in enumerate(zip(feature_arrays, labels)):
        latents_pca_np[label] = pca.transform(features_np)  # [N, 2]
        latents_pca[label] = latents_pca_np[label].tolist()
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    
    # Plot all samples except first (hollow markers)
    for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        latents = latents_pca_np[label]
        if len(latents) > 1:
            ax.scatter(
                latents[1:, 0], latents[1:, 1],
                facecolors='none', edgecolors=color, marker=marker, label=label, s=40
            )
        elif len(latents) == 1:
            # If only one sample, plot it as filled
            ax.scatter(
                latents[0:1, 0], latents[0:1, 1],
                facecolors=color, edgecolors='black', marker=marker, label=label, s=70
            )
    
    # Plot first sample of each set (filled markers with black edge)
    sample_idx = 5
    for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        latents = latents_pca_np[label]
        if len(latents) > 0:
            ax.scatter(
                latents[sample_idx:sample_idx+1, 0], latents[sample_idx:sample_idx+1, 1],
                facecolors=color, edgecolors='black', marker=marker, s=70, zorder=5
            )
    
    # Draw lines connecting the first samples (only if we have exactly 2 feature sets)
    if len(feature_sets) == 2:
        x0, y0 = latents_pca_np[labels[0]][sample_idx:sample_idx+1, 0], latents_pca_np[labels[0]][sample_idx:sample_idx+1, 1]
        x1, y1 = latents_pca_np[labels[1]][sample_idx:sample_idx+1, 0], latents_pca_np[labels[1]][sample_idx:sample_idx+1, 1]
        ax.plot([x0, x1], [y0, y1], 'k--', alpha=0.5, linewidth=1.5)
        
        # Add crosses at 25%, 50%, 75% along the line
        for frac in [0.25, 0.5, 0.75]:
            xc = x0 + (x1 - x0) * frac
            yc = y0 + (y1 - y0) * frac
            ax.scatter(xc, yc, marker='x', color='gray', s=60, linewidths=2, zorder=10)
    
    ax.set_xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = '_'.join(labels)
        plt.savefig(Path(save_dir) / f'pca_visualization_{name}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Return PCA results
    result = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'latents_pca': latents_pca
    }
    
    # Backward compatibility: add old keys for 2-feature case
    if len(feature_sets) == 2:
        result['content_latents_pca'] = latents_pca[labels[0]]
        result['style_latents_pca'] = latents_pca[labels[1]]
    
    return result

def compute_feature_similarity_metrics(model, val_loader, save_dir, device, verbose=True):
    """
    Compute cosine similarity and KL divergence metrics between content and style features.
    
    Args:
        model: Model to extract features from
        val_loader: DataLoader with 'simulation' and 'experimental' keys
        device: Device to run computation on
        verbose: Whether to print results
    
    Returns:
        Dictionary containing all computed metrics:
        - cosine_similarity: dict with 'content_style_paired', 'content_within', 'style_within'
        - kl_divergence: dict with 'content_style_paired', 'content_within', 'style_within'
    """
    model.eval()
    with torch.no_grad():
        # Collect all latents from validation set
        all_content_latents = []
        all_style_latents = []
        
        if verbose:
            print("Collecting latents from all validation samples...")
        all_content_patches = []
        all_style_patches = []
        all_sim = []
        all_exp = []
        for batch in val_loader:
            recon_sim, content_features_cls, content_features_patches = model(batch['simulation'].to(device))
            recon_exp, style_features_cls, style_features_patches = model(batch['experimental'].to(device))
            #content_features_patches = content_features_patches.view(content_features_patches.shape[0], -1)
            #style_features_patches = style_features_patches.view(style_features_patches.shape[0], -1)
            all_sim.append(batch['simulation'].cpu())
            all_exp.append(batch['experimental'].cpu())
            all_content_latents.append(content_features_cls.cpu())
            all_style_latents.append(style_features_cls.cpu())
            all_content_patches.append(content_features_patches.cpu())
            all_style_patches.append(style_features_patches.cpu())
            if 0:
                fig, ax = plt.subplots(2, 2, figsize=(10, 5))
                ax[0, 0].imshow(inverse_transform(batch['simulation'][1].permute(1, 2, 0).cpu().numpy()))
                ax[0, 1].imshow(inverse_transform(batch['experimental'][1].permute(1, 2, 0).cpu().numpy()))
                ax[1, 0].imshow(inverse_transform(recon_sim[1].permute(1, 2, 0).cpu().numpy()))
                ax[1, 1].imshow(inverse_transform(recon_exp[1].permute(1, 2, 0).cpu().numpy()))
                ax[0, 0].axis('off')
                ax[0, 1].axis('off')
                ax[1, 0].axis('off')
                ax[1, 1].axis('off')
                ax[0, 0].set_title('Simulation')
                ax[0, 1].set_title('Experimental')
                ax[1, 0].set_title('Reconstructed Simulation')
                ax[1, 1].set_title('Reconstructed Experimental')
                plt.show()
                asd()
            
        # Concatenate all batches
        content_features_cls = torch.cat(all_content_latents, dim=0)  # [N, D]
        style_features_cls = torch.cat(all_style_latents, dim=0)  # [N, D]
        all_sim = torch.cat(all_sim, dim=0).permute(0, 2, 3, 1)  # [B, N, D]
        all_exp = torch.cat(all_exp, dim=0).permute(0, 2, 3, 1)  # [B, N, D]
        print(f"all_sim shape: {all_sim.shape}, all_exp shape: {all_exp.shape}")
        all_content_patches = torch.cat(all_content_patches, dim=0)  # [B, N, D]
        all_style_patches = torch.cat(all_style_patches, dim=0)  # [B, N, D]
        sample_idx = 3
        results = {}
        if verbose:
            print(f"Total samples - content_features_cls: {content_features_cls.shape}, style_features_cls: {style_features_cls.shape}")
        if 1:
        # Normalize features for cosine similarity (L2 normalize)
            content_norm = F.normalize(content_features_cls, p=2, dim=-1)
            style_norm = F.normalize(style_features_cls, p=2, dim=-1)

            dist, pval = frechet_permutation_test(content_norm.cpu().numpy(), style_norm.cpu().numpy())
            print(f"Frechet distance: {dist}")
            print(f"Frechet permutation test p-value: {pval}")

            # Create figure with gridspec for custom layout
            if 1:
                fig = plt.figure(figsize=(16, 8))
                gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.3)

                # Top: bar plot spanning both columns
                ax_bar = fig.add_subplot(gs[0, :])

                # Get the number of features
                n_features = content_features_cls.shape[1]

                # Get the content and style feature arrays
                content_values = content_features_cls[sample_idx, :].cpu().numpy()
                style_values = style_features_cls[sample_idx, :].cpu().numpy()
                
                # Sort by content_values (you can easily swap for style_values)
                sort_indices = np.argsort(content_values)
                content_values_sorted = content_values[sort_indices]
                style_values_sorted = style_values[sort_indices]
                x = np.arange(n_features)
                width = 0.35

                # Create bars for sorted values, side by side
                bars1 = ax_bar.bar(x - width/2, content_values_sorted, width, label='Sim')
                bars2 = ax_bar.bar(x + width/2, style_values_sorted, width, label='Exp')

                ax_bar.set_xlabel('Feature Index')
                ax_bar.set_ylabel('Value')
                ax_bar.set_title('Sim vs Exp Features')
                ax_bar.set_xticks([i for i in x if i%2 == 0])
                ax_bar.legend()

                # Bottom: reconstructed images
                recon_sim = model.decoder(all_content_patches[sample_idx:sample_idx+1].to(device))
                recon_exp = model.decoder(all_style_patches[sample_idx:sample_idx+1].to(device))


                ax_sim = fig.add_subplot(gs[1, 0])
                ax_sim.imshow(all_sim[sample_idx])
                ax_sim.axis('off')
                ax_sim.set_title('Simulation')

                ax_sim = fig.add_subplot(gs[1, 1])
                ax_sim.imshow(inverse_transform(recon_sim[0].permute(1, 2, 0).cpu().detach().numpy()))
                ax_sim.axis('off')
                ax_sim.set_title('Reconstructed Simulation')

                ax_exp = fig.add_subplot(gs[1, 2])
                ax_exp.imshow(all_exp[sample_idx])
                ax_exp.axis('off')
                ax_exp.set_title('Experiment')
                ax_exp = fig.add_subplot(gs[1, 3])
                ax_exp.imshow(inverse_transform(recon_exp[0].permute(1, 2, 0).cpu().detach().numpy()))
                ax_exp.axis('off')
                ax_exp.set_title('Reconstructed Experiment')

                plt.tight_layout()
                plt.savefig(Path(save_dir) / f'feature_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
                asd()
            # Convert to probability distributions for KL divergence (using softmax)
            content_probs = F.softmax(content_features_cls, dim=-1)
            style_probs = F.softmax(style_features_cls, dim=-1)
            
            # ========== Cosine Similarity ==========
            # Between content[i] and style[i] (paired)
            content_style_cosine_paired = F.cosine_similarity(content_norm, style_norm, dim=-1).mean()
            
            # Within content group: content[i] vs content[j] (all pairs)
            content_cosine = torch.matmul(content_norm, content_norm.t())  # [B, B] - cosine similarity matrix
            mask = torch.triu(torch.ones_like(content_cosine), diagonal=1).bool()
            content_only_cosine = content_cosine[mask].mean()
            
            # Within style group: style[i] vs style[j] (all pairs)
            style_cosine = torch.matmul(style_norm, style_norm.t())  # [B, B] - cosine similarity matrix
            mask = torch.triu(torch.ones_like(style_cosine), diagonal=1).bool()
            style_only_cosine = style_cosine[mask].mean()
            
            # ========== KL Divergence ==========
            # Between content[i] and style[i] (paired)
            # KL(P||Q) = sum(P * log(P/Q)), where P=content, Q=style
            content_log_probs = F.log_softmax(content_features_cls, dim=-1)
            style_log_probs = F.log_softmax(style_features_cls, dim=-1)
            
            content_style_kl_paired = F.kl_div(
                content_log_probs,
                style_probs,
                reduction='batchmean'
            )
            
            # Within content group: content[i] vs content[j] (all pairs)
            # Vectorized: compute all pairwise KL divergences
            # KL(P_i || P_j) = sum(P_i * log(P_i / P_j)) = sum(P_i * (log(P_i) - log(P_j)))
            content_probs_expanded = content_probs.unsqueeze(1)  # [B, 1, D]
            content_log_diff = content_log_probs.unsqueeze(1) - content_log_probs.unsqueeze(0)  # [B, B, D]
            content_kl_matrix = (content_probs_expanded * content_log_diff).sum(dim=-1)  # [B, B]
            mask = torch.triu(torch.ones_like(content_kl_matrix), diagonal=1).bool()
            content_only_kl = content_kl_matrix[mask].mean()
            
            # Within style group: style[i] vs style[j] (all pairs)
            style_probs_expanded = style_probs.unsqueeze(1)  # [B, 1, D]
            style_log_diff = style_log_probs.unsqueeze(1) - style_log_probs.unsqueeze(0)  # [B, B, D]
            style_kl_matrix = (style_probs_expanded * style_log_diff).sum(dim=-1)  # [B, B]
            mask = torch.triu(torch.ones_like(style_kl_matrix), diagonal=1).bool()
            style_only_kl = style_kl_matrix[mask].mean()
            
            # Prepare results dictionary
            results = {
                'cosine_similarity': {
                    'content_style_paired': content_style_cosine_paired.item(),
                    'content_within': content_only_cosine.item(),
                    'style_within': style_only_cosine.item()
                },
                'kl_divergence': {
                    'content_style_paired': content_style_kl_paired.item(),
                    'content_within': content_only_kl.item(),
                    'style_within': style_only_kl.item()
                }
            }
            
            if verbose:
                print("=" * 60)
                print("COSINE SIMILARITY:")
                print(f"  content[i] vs style[i] (paired): {results['cosine_similarity']['content_style_paired']:.6f}")
                print(f"  content[i] vs content[j] (within-group): {results['cosine_similarity']['content_within']:.6f}")
                print(f"  style[i] vs style[j] (within-group): {results['cosine_similarity']['style_within']:.6f}")
                print("=" * 60)
                print("KL DIVERGENCE:")
                print(f"  content[i] vs style[i] (paired): {results['kl_divergence']['content_style_paired']:.6f}")
                print(f"  content[i] vs content[j] (within-group): {results['kl_divergence']['content_within']:.6f}")
                print(f"  style[i] vs style[j] (within-group): {results['kl_divergence']['style_within']:.6f}")
                print("=" * 60)
        if 0: # experiment permutation test -- randomly shuffle the content and style features pairs
            content_features_cls_permuted = content_features_cls.clone()
            style_features_cls_permuted = style_features_cls.clone()
            # Apply DIFFERENT permutations to break the pairing
            perm_content = torch.randperm(content_features_cls.shape[0])
            perm_style = torch.randperm(style_features_cls.shape[0])
            content_features_cls_permuted = content_features_cls_permuted[perm_content]
            style_features_cls_permuted = style_features_cls_permuted[perm_style]
            
            # Normalize permuted features for cosine similarity
            content_norm_permuted = F.normalize(content_features_cls_permuted, p=2, dim=-1)
            style_norm_permuted = F.normalize(style_features_cls_permuted, p=2, dim=-1)
            
            # Convert permuted features to probability distributions for KL divergence
            content_probs_permuted = F.softmax(content_features_cls_permuted, dim=-1)
            style_probs_permuted = F.softmax(style_features_cls_permuted, dim=-1)
            content_log_probs_permuted = F.log_softmax(content_features_cls_permuted, dim=-1)
            style_log_probs_permuted = F.log_softmax(style_features_cls_permuted, dim=-1)
            
            # ========== Cosine Similarity (Permuted) ==========
            # Between content[i] and style[i] (paired, permuted)
            content_style_cosine_paired_permuted = F.cosine_similarity(content_norm_permuted, style_norm_permuted, dim=-1).mean()
            
            # Within content group: content[i] vs content[j] (all pairs, permuted)
            content_cosine_permuted = torch.matmul(content_norm_permuted, content_norm_permuted.t())  # [B, B]
            mask_content_permuted = torch.triu(torch.ones_like(content_cosine_permuted), diagonal=1).bool()
            content_only_cosine_permuted = content_cosine_permuted[mask_content_permuted].mean()
            
            # Within style group: style[i] vs style[j] (all pairs, permuted)
            style_cosine_permuted = torch.matmul(style_norm_permuted, style_norm_permuted.t())  # [B, B]
            mask_style_permuted = torch.triu(torch.ones_like(style_cosine_permuted), diagonal=1).bool()
            style_only_cosine_permuted = style_cosine_permuted[mask_style_permuted].mean()
            
            # ========== KL Divergence (Permuted) ==========
            # Between content[i] and style[i] (paired, permuted)
            content_style_kl_paired_permuted = F.kl_div(
                content_log_probs_permuted,
                style_probs_permuted,
                reduction='batchmean'
            )
            
            # Within content group: content[i] vs content[j] (all pairs, permuted)
            content_probs_expanded_permuted = content_probs_permuted.unsqueeze(1)  # [B, 1, D]
            content_log_diff_permuted = content_log_probs_permuted.unsqueeze(1) - content_log_probs_permuted.unsqueeze(0)  # [B, B, D]
            content_kl_matrix_permuted = (content_probs_expanded_permuted * content_log_diff_permuted).sum(dim=-1)  # [B, B]
            mask_content_kl_permuted = torch.triu(torch.ones_like(content_kl_matrix_permuted), diagonal=1).bool()
            content_only_kl_permuted = content_kl_matrix_permuted[mask_content_kl_permuted].mean()
            
            # Within style group: style[i] vs style[j] (all pairs, permuted)
            style_probs_expanded_permuted = style_probs_permuted.unsqueeze(1)  # [B, 1, D]
            style_log_diff_permuted = style_log_probs_permuted.unsqueeze(1) - style_log_probs_permuted.unsqueeze(0)  # [B, B, D]
            style_kl_matrix_permuted = (style_probs_expanded_permuted * style_log_diff_permuted).sum(dim=-1)  # [B, B]
            mask_style_kl_permuted = torch.triu(torch.ones_like(style_kl_matrix_permuted), diagonal=1).bool()
            style_only_kl_permuted = style_kl_matrix_permuted[mask_style_kl_permuted].mean()
            
            results['cosine_similarity']['content_style_paired_permuted'] = content_style_cosine_paired_permuted.item()
            results['cosine_similarity']['content_within_permuted'] = content_only_cosine_permuted.item()
            results['cosine_similarity']['style_within_permuted'] = style_only_cosine_permuted.item()
            results['kl_divergence']['content_style_paired_permuted'] = content_style_kl_paired_permuted.item()
            results['kl_divergence']['content_within_permuted'] = content_only_kl_permuted.item()
            results['kl_divergence']['style_within_permuted'] = style_only_kl_permuted.item()
            print("=" * 60)
            print("COSINE SIMILARITY PERMUTED:")
            print(f"  content[i] vs style[i] (paired): {results['cosine_similarity']['content_style_paired_permuted']:.6f}")
            print(f"  content[i] vs content[j] (within-group): {results['cosine_similarity']['content_within_permuted']:.6f}")
            print(f"  style[i] vs style[j] (within-group): {results['cosine_similarity']['style_within_permuted']:.6f}")
            print("=" * 60)
            print("KL DIVERGENCE PERMUTED:")
            print(f"  content[i] vs style[i] (paired): {results['kl_divergence']['content_style_paired_permuted']:.6f}")
            print(f"  content[i] vs content[j] (within-group): {results['kl_divergence']['content_within_permuted']:.6f}")
            print(f"  style[i] vs style[j] (within-group): {results['kl_divergence']['style_within_permuted']:.6f}")
            print("=" * 60)
            asd()

        # ========== PCA Visualization ==========
        pca_results = visualize_pca(
            content_features_cls, 
            style_features_cls, 
            save_dir=save_dir,
            labels=['Simulation', 'Experiment']
        )
        results['pca'] = pca_results
        
        return results

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
        diffusion_weight=config['loss']['diffusion_weight'],
        image_weight=config['loss']['image_weight'],
        fused_patches_weight=config['loss']['fused_patches_weight']
    )
    style_model = StyleTransferModel(
        feature_extractor=feature_extractor,
        #decoder=feature_extractor.decoder,
        decoder=ConvDecoder(**decoder_config) if config['model']['decoder_type'] == 'conv' else TransformerDecoder(**decoder_config),
        #adain=AdaINFusion()
        #adain=HistoAdaIN(embed_dim=config['model']['embed_dim'])
        adain=FusionModule(embed_dim=config['model']['embed_dim'], mode='cross_attn')
        #adain=WeightedAdaIN(embed_dim=config['model']['embed_dim'])
        #adain=LearnableAdaIN(embed_dim=config['model']['embed_dim'])
        #adain=HistoAdaINSpatialAware(embed_dim=config['model']['embed_dim'], use_content_residual=config['training'][stage_name]['use_content_residual'])
        #adain=HistoAdaINHybrid(patch_size=config['model']['patch_size'])
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
        use_all_pairs=config['training'][stage_name]['use_all_pairs'],
    )
    param_groups = [
        {
            'params': style_model.decoder.parameters(),
            'lr': config['training'][stage_name]['learning_rate']*0.01,
            'weight_decay': config['training'][stage_name]['weight_decay'],
        },
        {
            'params': style_model.adain.parameters(),
            'lr': config['training'][stage_name]['learning_rate'],
            'weight_decay': config['training'][stage_name]['weight_decay'],
        }
    ]
    if config['training'][stage_name]['use_diffusion'] and diffusion_model is not None:
        param_groups.append({
            'params': diffusion_model.parameters(),
            'lr': config['training'][stage_name]['learning_rate'],
            'weight_decay': config['training'][stage_name]['weight_decay'],
        })
    optimizer = optim.AdamW(param_groups)
    if 0:
        optimizer = optim.AdamW(
            list(style_model.decoder.parameters())+ list(style_model.adain.parameters()) + (list(diffusion_model.parameters()) if config['training'][stage_name]['use_diffusion'] and diffusion_model is not None else []),
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

def inverse_transform(img):
    return (img*0.5) + 0.5

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
        train_split=config['data']['train_split'],
        seed=config['data']['seed']
    )

    # Visualize the first batch of the training data
    if 0:#True:
        print("Visualizing first batch of the training data...")
        n_viz = 5
        sub_fig_size = 3
        val_iter = iter(val_loader)  # Create iterator once
        for _ in range(1):
            batch = next(val_iter)  # Get next batch from the same iterator
            fig, ax = plt.subplots(3, n_viz, figsize=(sub_fig_size * n_viz, sub_fig_size * 3))
            for i in range(n_viz):
                sim_img = batch['simulation'][i].permute(1, 2, 0).cpu().numpy()
                exp_img = batch['experimental'][i].permute(1, 2, 0).cpu().numpy()
                sim_img = inverse_transform(sim_img)
                exp_img = inverse_transform(exp_img)
                ax[2, i].hist(sim_img[..., 0].flatten(), bins=10)
                ax[1, i].imshow(sim_img[..., 0], vmin=0, vmax=1, cmap='gray')
                ax[0, i].imshow(exp_img)
                ax[0, i].axis('off')
                ax[1, i].axis('off')
            plt.tight_layout()
            plt.show()
        plt.savefig(f'first_batch_visualization.png', dpi=300, bbox_inches='tight')
        asd()
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
    # Compute feature similarity metrics
    metrics = compute_feature_similarity_metrics(model, val_loader, save_dir=config['training']['stage_1']['save_dir'], device=device)
    
    asd()
    if 0: # latent adapter training
        latent_adapter = LatentDomainAdapter(
            feature_extractor=model.encoder,
            n_patches=(config['model']['img_size'] // config['model']['patch_size']) ** 2,
            latent_dim=config['model']['embed_dim'],
        )
        for _ in range(1):
            loss = train_latent_adapter_epoch(latent_adapter, train_loader, optimizer, device=device)
            print(f"Loss dictionary: {loss}")
                # Compare with ground truth
        test_batch = next(iter(val_loader))
        sim_images_test = test_batch['simulation'].to(device)
        exp_images_test = test_batch['experimental'].to(device)
        with torch.no_grad():
            _, sim_latents = model.encoder(sim_images_test)
            _, target_latents = model.encoder(exp_images_test)
        mse_original = torch.nn.functional.mse_loss(sim_latents, target_latents)
        print(f"MSE between original and target: {mse_original.item():.4f}")
        generated_latents, _ = latent_adapter(sim_images_test, exp_images_test)
        mse = torch.nn.functional.mse_loss(generated_latents, target_latents)
        print(f"MSE between generated and target: {mse.item():.4f}")
        if 0:
            # Train diffusion model
            diffusion_model = DiffusionModel(
                unet_config=config['diffusion']['unet_config'],
                timesteps=config['diffusion']['timesteps'],
                beta_start=config['diffusion']['beta_start'],
                beta_end=config['diffusion']['beta_end']
            )
            trainer_diffusion = LatentDiffusionTrainer(
                diffusion_model=diffusion_model,
                sim_feature_extractor=model.encoder,
                exp_feature_extractor=model.encoder,
                train_loader=train_loader,
                val_loader=val_loader,
                embedding_dim=config['model']['embed_dim'],
                device=device
            )
            trainer_diffusion.train(num_epochs=2)
            test_batch = next(iter(val_loader))
            sim_images_test = test_batch['simulation'][:4].to(device)
            exp_images_test = test_batch['experimental'][:4].to(device)
            
            # Generate experimental latents from simulation images
            with torch.no_grad():
                generated_latents = trainer_diffusion.sample_translations(
                    sim_images_test,
                    num_inference_steps=200,
                    use_ema=True
                )
            
            print(f"Generated latents shape: {generated_latents.shape}")
            
            # Compare with ground truth
            with torch.no_grad():
                sim_latents, target_latents = trainer_diffusion.extract_latents(sim_images_test, exp_images_test)
            mse_original = torch.nn.functional.mse_loss(sim_latents, target_latents)
            print(f"MSE between original and target: {mse_original.item():.4f}")
            mse = torch.nn.functional.mse_loss(generated_latents, target_latents)
            print(f"MSE between generated and target: {mse.item():.4f}")
            
            print("\nTraining completed successfully!")
        recon_sim = model.decoder(sim_latents.to(device))
        recon_target = model.decoder(target_latents.to(device))
        recon_generated = model.decoder(generated_latents.to(device))
        print(f"Recon generated shape: {recon_generated.shape}")
        fig, ax = plt.subplots(1, 5, figsize=(3 * 5, 3))
        ax[0].imshow(inverse_transform(sim_images_test[0].permute(1, 2, 0).cpu().detach().numpy()))
        ax[1].imshow(inverse_transform(recon_sim[0].permute(1, 2, 0).cpu().detach().numpy()))
        ax[2].imshow(inverse_transform(exp_images_test[0].permute(1, 2, 0).cpu().detach().numpy()))
        ax[3].imshow(inverse_transform(recon_target[0].permute(1, 2, 0).cpu().detach().numpy()))
        ax[4].imshow(inverse_transform(recon_generated[0].permute(1, 2, 0).cpu().detach().numpy()))
        ax[0].set_title('Sim (Input)', loc='center')
        ax[1].set_title('Sim Reconstructed (Output)', loc='center')
        ax[2].set_title('Exp (Input)', loc='center')
        ax[3].set_title('Exp Reconstructed (Output)', loc='center')
        ax[4].set_title('Generated (Output)', loc='center')
        for i in range(5):
            ax[i].axis('off')
        plt.tight_layout()
        plt.savefig(Path(config['training']['stage_1']['save_dir']) / f'generated_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    if 1:
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
    if 1:
        visualize_reconstructed_images(model,
            val_loader,
            device,
            save_dir=config['visualization']['reconstructed_images']['save_dir'],
            n_viz=config['visualization']['reconstructed_images']['n_viz'],
            states=list(state_names.keys())
        )
    if 1:
        visualize_style_transfer(style_model,
            val_loader,
            device,
            save_dir=config['visualization']['style_transfer']['save_dir'],
            n_viz=config['visualization']['style_transfer']['n_viz'],
            states=list(state_names.keys())
        )
    style_model.eval()
    first_batch = next(iter(val_loader))
    all_content_features_cls = []
    all_style_features_cls = []
    all_recon_features_cls = []
    for batch in val_loader:
        content_features_cls = model.encoder(batch['simulation'].to(device))[0]
        style_features_cls = model.encoder(batch['experimental'].to(device))[0]
        recon = style_model(batch['simulation'].to(device), batch['experimental'].to(device))
        recon_features_cls = model.encoder(recon.to(device))[0]
        all_content_features_cls.append(content_features_cls)
        all_style_features_cls.append(style_features_cls)
        all_recon_features_cls.append(recon_features_cls.detach())
    all_content_features_cls = torch.cat(all_content_features_cls, dim=0)
    all_style_features_cls = torch.cat(all_style_features_cls, dim=0)
    all_recon_features_cls = torch.cat(all_recon_features_cls, dim=0)
    pca_results = visualize_pca(
        all_content_features_cls.detach().cpu().numpy(), 
        all_style_features_cls.detach().cpu().numpy(), 
        all_recon_features_cls.detach().cpu().numpy(), 
        labels=['Simulation', 'Experiment', 'Reconstructed'],
        save_dir=config['training']['stage_2']['save_dir']
    )

def visualize_reconstructed_images(model, val_loader, device, save_dir, n_viz=5, states=list(state_names.keys())):
    print(f"Visualizing reconstructed images...")
    model.eval()
    first_batch = next(iter(val_loader))
    
    
    exp_imgs = first_batch['experimental']
    sim_imgs = first_batch['simulation']
    pred_sim_imgs, sim_cls_token, sim_patch_tokens = model(sim_imgs.to(device))
    pred_exp_imgs, exp_cls_token, exp_patch_tokens = model(exp_imgs.to(device))
    exp_idx = 0
    fig, ax = plt.subplots(1, 5, figsize=(3 * 5, 3))
    for i in range(5):
        fused_patches = i * 0.25 * sim_patch_tokens + (4 - i) * 0.25 * exp_patch_tokens
        pred_fused_recon = model.decoder(fused_patches.to(device))[exp_idx]
        ax[i].imshow(torch.clamp(inverse_transform(pred_fused_recon.permute(1, 2, 0).cpu().detach()), 0, 1).numpy())
        ax[i].axis('off')
        ax[i].set_title(f'{i * 25}% Sim + {100 - i * 25}% Exp')
    
    # fit the exp distribution and sample from it
    # Convert to numpy and get the shape
    exp_patch_tokens_np = exp_patch_tokens.cpu().detach().numpy()
    B, num_patches, embed_dim = exp_patch_tokens_np.shape
    
    # Fit a separate multivariate Gaussian distribution for each patch position
    patch_means = []
    patch_covs = []
    
    for patch_idx in range(num_patches):
        # Collect all tokens for this patch position across all batches
        # Shape: (B, embed_dim)
        patch_tokens = exp_patch_tokens_np[:, patch_idx, :]
        
        # Fit multivariate Gaussian for this patch position
        mean = np.mean(patch_tokens, axis=0)  # (embed_dim,)
        cov = np.cov(patch_tokens.T)  # (embed_dim, embed_dim)
        
        # Add small regularization to ensure covariance is positive definite
        cov += np.eye(embed_dim) * 1e-6
        
        patch_means.append(mean)
        patch_covs.append(cov)
    
    # Sample from each patch's distribution with the same shape as exp_patch_tokens
    sampled_patch_tokens = np.zeros((B, num_patches, embed_dim))
    for patch_idx in range(num_patches):
        # Sample B vectors from this patch's distribution
        sampled_patch_tokens[:, patch_idx, :] = np.random.multivariate_normal(
            patch_means[patch_idx], 
            patch_covs[patch_idx], 
            size=B
        )
    
    sampled_patch_tokens = torch.from_numpy(sampled_patch_tokens).float().to(device)
    
    for i in range(0):
        # Use the sampled patch tokens (you can index by i if you want different samples)
        fused_patches = sampled_patch_tokens[i:i+1] if i < B else sampled_patch_tokens[0:1]
        pred_fused_recon = model.decoder(fused_patches.to(device))[0]
        ax[i].imshow(torch.clamp(inverse_transform(pred_fused_recon.permute(1, 2, 0).cpu().detach()), 0, 1).numpy())
        ax[i].axis('off')
        #ax[i].set_title(f'Sample from Exp Distribution')
    #fig, ax = plt.subplots(2, 5, figsize=(3 * 5, 3 * 2))
    for i in range(0):
        fused_patches = i * 0.25 * exp_patch_tokens[exp_idx:exp_idx+1] + (4 - i) * 0.25 * exp_patch_tokens[exp_idx+1:exp_idx+2]
        fused_patches_sim = i * 0.25 * sim_patch_tokens[exp_idx:exp_idx+1] + (4 - i) * 0.25 * sim_patch_tokens[exp_idx+1:exp_idx+2]
        #pred_fused_recon = model.decoder(random_fused_patches.to(device))[0]
        pred_fused_recon = model.decoder(fused_patches.to(device))[0]
        pred_fused_recon_sim = model.decoder(fused_patches_sim.to(device))[0]
        ax[0, i].imshow(torch.clamp(inverse_transform(pred_fused_recon.permute(1, 2, 0).cpu().detach()), 0, 1).numpy())
        ax[1, i].imshow(torch.clamp(inverse_transform(pred_fused_recon_sim.permute(1, 2, 0).cpu().detach()), 0, 1).numpy())
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        ax[0, i].set_title(f'{i * 25}% Exp1 + {100 - i * 25}% Exp2')
        ax[1, i].set_title(f'{i * 25}% Sim + {100 - i * 25}% Exp')
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(Path(save_dir) / f'fused_reconstructed_images.png', dpi=300, bbox_inches='tight', transparent=True)

    exp_imgs = inverse_transform(exp_imgs)
    pred_exp_imgs = inverse_transform(pred_exp_imgs)
    sim_imgs = inverse_transform(sim_imgs)
    pred_sim_imgs = inverse_transform(pred_sim_imgs)

    fig, ax = plt.subplots(n_viz, 4, figsize=(3 * 4, 3 * n_viz))
    for state in states:
        for i in range(n_viz):
            ax[i, 0].imshow(torch.clamp(exp_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy())
            ax[i, 1].imshow(torch.clamp(pred_exp_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy())
            ax[i, 2].imshow(torch.clamp(sim_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy())#[..., state])
            ax[i, 3].imshow(torch.clamp(pred_sim_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1).cpu().numpy())#[..., state])
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
    fig, ax = plt.subplots(n_viz, 5, figsize=(3 * 5, 3 * n_viz))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for state in states:
        for i in range(n_viz):
            exp_img = first_batch['experimental'][i].unsqueeze(0) # style
            shuffled_exp_img = first_batch['shuffled_exp'][i].unsqueeze(0) # style
            sim_img = first_batch['simulation'][i].unsqueeze(0) # content
            pred_imgs = style_model(sim_img.to(device), exp_img.to(device))[0].permute(1, 2, 0).cpu().detach()
            shuffled_pred_imgs = style_model(sim_img.to(device), shuffled_exp_img.to(device))[0].permute(1, 2, 0).cpu().detach()
            print(f"pred_img[0] and shuffled_pred_imgs[0] is the same: {torch.allclose(pred_imgs[0], shuffled_pred_imgs[0])}")
            ax[i, 0].imshow(np.clip(inverse_transform(exp_img[0].permute(1, 2, 0).cpu().detach().numpy()), 0, 1))
            ax[i, 1].imshow(np.clip(inverse_transform(sim_img[0].permute(1, 2, 0)[..., state].cpu().detach().numpy()), 0, 1), cmap='gray')
            ax[i, 2].imshow(np.clip(inverse_transform(pred_imgs.numpy()), 0, 1))
            ax[i, 3].imshow(np.clip(inverse_transform(shuffled_exp_img[0].permute(1, 2, 0).cpu().detach().numpy()), 0, 1))
            ax[i, 4].imshow(np.clip(inverse_transform(shuffled_pred_imgs.numpy()), 0, 1))
            ax[i, 0].axis('off')
            ax[i, 1].axis('off')
            ax[i, 2].axis('off')
            ax[i, 3].axis('off')
            ax[i, 4].axis('off')
            if i == 0:
                ax[i, 0].set_title('Exp (Style)', loc='center')
                ax[i, 1].set_title(f'Sim (Content) {state_names[state]}', loc='center')
                ax[i, 2].set_title('Style Transferred\n(Output)', loc='center')
                ax[i, 3].set_title('Shuffled Style\n(Input)', loc='center')
                ax[i, 4].set_title('Shuffled Style Transferred\n(Output)', loc='center')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.05)

        plt.savefig(Path(save_dir) / f'style_transfer_{state_names[state]}.png', dpi=300, bbox_inches='tight')#, transparent=True)


if __name__ == '__main__':
    import sys
    
    # Allow optional config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    
    print(f"Loading configuration from: {config_path}")
    main(config_path)
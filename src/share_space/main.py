import yaml
import os
import copy
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from share_space.models.sim2exp_model import Sim2ExpModel, StyleTransferModel
from share_space.train import TeacherStudentTrainer
from share_space.loss import Sim2ExpLoss, StyleTransferLoss
from share_space.dataset import get_dummy_dataloaders, get_real_dataloaders
from share_space.evaluation import MetricsEvaluator, compute_feature_similarity_metrics
from share_space.train_style import StyleTransferTrainer
from share_space.models.decoders import ConvDecoder, TransformerDecoder
from share_space.diffusion import DiffusionModel
from share_space.trainer_diffusion import LatentDiffusionTrainer
from share_space.adain_histo import (
    HistoAdaINSpatialAware,
    HistoAdaIN,
    HistoAdaINHybrid,
    WeightedAdaIN,
    LearnableAdaIN,
    FusionModule,
)
from share_space.latent_adapter import LatentDomainAdapter
from share_space.latent_adapter import train_epoch as train_latent_adapter_epoch
from share_space.permutation_test import (
    frechet_permutation_test_animated,
    create_static_summary_plot,
)
from share_space.get_models import get_stage_1_model, get_stage_2_model
from share_space.visualizer import (
    visualize_reconstructed_images,
    visualize_images_sampled_from_distribution,
    visualize_linearly_fused_images,
    visualize_pca,
    visualize_first_batch,
    visualize_style_transfer,
)
from share_space.utils import (
    inverse_transform,
    inverse_transform_batch,
    get_all_samples_from_loader,
)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_stage(
    model,
    trainer,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    val_loader,
    evaluator,
    stage_config,
    stage_name,
    best_psnr,
    device,
    run_final_eval=False,
):
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
    if stage_config["eval_only"]:  # load the best model and evaluate
        print(f"Loading best model for {stage_name} from {stage_config['load_path']}...")
        model.load_state_dict(
            torch.load(stage_config["load_path"], map_location=device, weights_only=False)[
                "model_state_dict"
            ]
        )
        model.to(device)  # Ensure model is on the correct device
    else:
        for epoch in range(stage_config["epochs"]):
            print(f"\n{stage_name.upper()} - Epoch {epoch + 1}/{stage_config['epochs']}")

            # Train
            train_losses = trainer.train_epoch(
                train_loader, optimizer, loss_fn, mask_ratio=stage_config["mask_ratio"]
            )

            # Store losses for plotting
            for key in train_losses.keys():
                loss_history.setdefault(key, []).append(train_losses[key])

            # Validate
            if (epoch + 1) % stage_config["eval_every"] == 0:
                print("Evaluating...")
                metrics = evaluator.evaluate_model(model, val_loader, stage_name)
                print(f"Validation metrics: {metrics}")

                # Save best model
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    save_path = Path(stage_config["save_dir"]) / "best_model.pth"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": metrics,
                        },
                        save_path,
                    )
                    print(f"Saved best model with PSNR: {best_psnr:.4f}")

            # Step scheduler
            scheduler.step()

        print(f"Training {stage_name} complete!")
    # Final evaluation
    if stage_config["run_final_eval"]:
        print(f"\nFinal evaluation for {stage_name}...")
        final_metrics = evaluator.evaluate_model(model, val_loader, stage_name=stage_name)
        print(f"Final metrics: {final_metrics}")

        # Save final metrics
        metrics_path = Path(stage_config["save_dir"]) / f"final_metrics_{stage_name}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        # Visualize loss history
        for key, values in loss_history.items():
            if isinstance(values[0], torch.Tensor):
                loss_history[key] = [value.detach().cpu().numpy() for value in values]
            else:
                loss_history[key] = values
        if not stage_config["eval_only"] and len(loss_history) > 0:
            plt.figure(figsize=(8, 2))
            epochs = range(1, len(loss_history[list(loss_history.keys())[0]]) + 1)
            for key in loss_history.keys():
                plt.plot(epochs, loss_history[key], label=key, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{stage_name.upper()} - Training Losses Over Epochs")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                Path(stage_config["save_dir"]) / f"loss_history_{stage_name}.png",
                dpi=300,
                bbox_inches="tight",
            )

    return best_psnr, loss_history


def save_patch_tokens(model, train_loader, config, device):
    """
    Extracts content and style patch tokens from the entire train_loader using the model,
    and saves them as .npy files in the specified save_dir.
    """
    content_patch_tokens_list = []
    style_patch_tokens_list = []
    for batch in train_loader:
        sim_images = batch["simulation"].to(device)
        exp_images = batch["experimental"].to(device)
        _, _, content_patch_tokens = model(sim_images)
        _, _, style_patch_tokens = model(exp_images)
        content_patch_tokens_list.append(content_patch_tokens.cpu().detach().numpy())
        style_patch_tokens_list.append(style_patch_tokens.cpu().detach().numpy())
    content_patch_tokens = np.concatenate(content_patch_tokens_list, axis=0)
    style_patch_tokens = np.concatenate(style_patch_tokens_list, axis=0)
    save_dir = Path(config["training"]["stage_1"]["save_dir"])
    np.save(save_dir / f"content_patch_tokens.npy", content_patch_tokens)
    np.save(save_dir / f"style_patch_tokens.npy", style_patch_tokens)
    return content_patch_tokens, style_patch_tokens


def main(config_path="config.yaml"):
    # Load configuration
    config = load_config(config_path)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = get_real_dataloaders(
        exp_dir=config["data"]["exp_dir"],
        sim_dir=config["data"]["sim_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        img_size=config["model"]["img_size"],
        train_split=config["data"]["train_split"],
        seed=config["data"]["seed"],
    )

    if 0:
        visualize_first_batch(train_loader)
    # Train stage 1
    model, trainer, optimizer, scheduler, loss_fn, evaluator = get_stage_1_model(
        config, device, "stage_1"
    )
    print("Starting training...")
    best_psnr = float("-inf")

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
        stage_config=config["training"]["stage_1"],
        stage_name="stage_1",
        best_psnr=best_psnr,
        device=device,
        run_final_eval=True,
    )
    # Compute feature similarity metrics
    if 0:
        metrics = compute_feature_similarity_metrics(model, val_loader, device=device, verbose=True)
    if 0:
        if config["training"]["stage_1"]["eval_only"]:
            content_patch_tokens = np.load(
                Path(config["training"]["stage_1"]["save_dir"]) / f"content_patch_tokens.npy"
            )
            style_patch_tokens = np.load(
                Path(config["training"]["stage_1"]["save_dir"]) / f"style_patch_tokens.npy"
            )
        else:
            content_patch_tokens, style_patch_tokens = save_patch_tokens(
                model, train_loader, config, device
            )
        print(f"Content patch tokens shape: {content_patch_tokens.shape}")
        print(f"Style patch tokens shape: {style_patch_tokens.shape}")
    if 0:  # latent adapter training
        latent_adapter = LatentDomainAdapter(
            feature_extractor=model.encoder,
            n_patches=(config["model"]["img_size"] // config["model"]["patch_size"]) ** 2,
            latent_dim=config["model"]["embed_dim"],
        )
        for _ in range(1):
            loss = train_latent_adapter_epoch(
                latent_adapter, train_loader, optimizer, device=device
            )
            print(f"Loss dictionary: {loss}")
            # Compare with ground truth
        test_batch = next(iter(val_loader))
        sim_images_test = test_batch["simulation"].to(device)
        exp_images_test = test_batch["experimental"].to(device)
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
                unet_config=config["diffusion"]["unet_config"],
                timesteps=config["diffusion"]["timesteps"],
                beta_start=config["diffusion"]["beta_start"],
                beta_end=config["diffusion"]["beta_end"],
            )
            trainer_diffusion = LatentDiffusionTrainer(
                diffusion_model=diffusion_model,
                sim_feature_extractor=model.encoder,
                exp_feature_extractor=model.encoder,
                train_loader=train_loader,
                val_loader=val_loader,
                embedding_dim=config["model"]["embed_dim"],
                device=device,
            )
            trainer_diffusion.train(num_epochs=2)
            test_batch = next(iter(val_loader))
            sim_images_test = test_batch["simulation"][:4].to(device)
            exp_images_test = test_batch["experimental"][:4].to(device)

            # Generate experimental latents from simulation images
            with torch.no_grad():
                generated_latents = trainer_diffusion.sample_translations(
                    sim_images_test, num_inference_steps=200, use_ema=True
                )

            print(f"Generated latents shape: {generated_latents.shape}")

            # Compare with ground truth
            with torch.no_grad():
                sim_latents, target_latents = trainer_diffusion.extract_latents(
                    sim_images_test, exp_images_test
                )
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
        ax[0].set_title("Sim (Input)", loc="center")
        ax[1].set_title("Sim Reconstructed (Output)", loc="center")
        ax[2].set_title("Exp (Input)", loc="center")
        ax[3].set_title("Exp Reconstructed (Output)", loc="center")
        ax[4].set_title("Generated (Output)", loc="center")
        for i in range(5):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(
            Path(config["training"]["stage_1"]["save_dir"]) / f"generated_images.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    style_model, trainer_style, optimizer, scheduler, loss_fn, evaluator = get_stage_2_model(
        config, device, "stage_2", model
    )
    best_psnr = float("-inf")
    best_psnr = train_stage(
        model=style_model,
        trainer=trainer_style,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        stage_config=config["training"]["stage_2"],
        stage_name="stage_2",
        best_psnr=best_psnr,
        device=device,
        run_final_eval=True,
    )
    if 1:
        model.eval()
        all_sim_imgs, all_exp_imgs = get_all_samples_from_loader(val_loader)
        pred_sim_imgs, sim_cls_token, sim_patch_tokens = model(all_sim_imgs.to(device))
        pred_exp_imgs, exp_cls_token, exp_patch_tokens = model(all_exp_imgs.to(device))
        pred_exp_imgs = inverse_transform_batch(pred_exp_imgs)
        pred_sim_imgs = inverse_transform_batch(pred_sim_imgs)
        all_sim_imgs = inverse_transform_batch(all_sim_imgs)
        all_exp_imgs = inverse_transform_batch(all_exp_imgs)
        n_viz = 4
        sample_idx = 5
        visualize_linearly_fused_images(
            model,
            sim_patch_tokens[sample_idx : sample_idx + 1],
            exp_patch_tokens[sample_idx : sample_idx + 1],
            device,
            save_dir=config["visualization"]["reconstructed_images"]["save_dir"],
        )
        visualize_reconstructed_images(all_exp_imgs, pred_exp_imgs, all_sim_imgs, pred_sim_imgs, save_dir=config['visualization']['reconstructed_images']['save_dir'])
        visualize_images_sampled_from_distribution(model, exp_patch_tokens, device, save_dir=config['visualization']['reconstructed_images']['save_dir'])

        # ========== PCA Visualization ==========
        if 1:
            pca_results = visualize_pca(
                sim_cls_token,
                exp_cls_token,
                save_dir=config["visualization"]["reconstructed_images"]["save_dir"],
                labels=["Simulation", "Experiment"],
            )
            content_latents = np.array(pca_results["content_latents_pca"])
            style_latents = np.array(pca_results["style_latents_pca"])
            obs, pval = frechet_permutation_test_animated(
                content_latents,
                style_latents,
                n_perms=100,
                seed=42,
                figsize=(8, 3.75),
                save_dir=config["visualization"]["reconstructed_images"]["save_dir"],
                fps=15,
                n_bins=20,
            )
            create_static_summary_plot(
                content_latents,
                style_latents,
                n_perms=100,
                seed=42,
                figsize=(8, 3.75),
                save_dir=config["visualization"]["reconstructed_images"]["save_dir"],
                n_bins=20,
            )

    if 1:
        visualize_style_transfer(
            style_model,
            val_loader,
            device,
            save_dir=config["visualization"]["style_transfer"]["save_dir"],
            n_viz=config["visualization"]["style_transfer"]["n_viz"],
        )
    if 0:
        style_model.eval()
        first_batch = next(iter(val_loader))
        all_content_features_cls = []
        all_style_features_cls = []
        all_recon_features_cls = []
        all_content_features_patches = []
        all_style_features_patches = []
        all_recon_features_patches = []
        sample_idx = 5
        for batch in val_loader:
            content_features_cls, content_features_patches = model.encoder(
                batch["simulation"].to(device)
            )
            style_features_cls, style_features_patches = model.encoder(
                batch["experimental"].to(device)
            )
            recon = style_model(batch["simulation"].to(device), batch["experimental"].to(device))
            recon_features_cls, recon_features_patches = model.encoder(recon.to(device))
            all_content_features_cls.append(content_features_cls)
            all_style_features_cls.append(style_features_cls)
            all_recon_features_cls.append(recon_features_cls)
            all_content_features_patches.append(content_features_patches)
            all_style_features_patches.append(style_features_patches)
            all_recon_features_patches.append(recon_features_patches)
        all_content_features_cls = torch.cat(all_content_features_cls, dim=0)
        all_style_features_cls = torch.cat(all_style_features_cls, dim=0)
        all_recon_features_cls = torch.cat(all_recon_features_cls, dim=0)
        all_content_features_patches = torch.cat(all_content_features_patches, dim=0)
        all_style_features_patches = torch.cat(all_style_features_patches, dim=0)
        all_recon_features_patches = torch.cat(all_recon_features_patches, dim=0)
        recon_style = model.decoder(all_style_features_patches.to(device))
        recon_style_cls, _ = model.encoder(recon_style.to(device))
        pca_results = visualize_pca(
            all_content_features_cls.detach().cpu().numpy(),
            all_style_features_cls.detach().cpu().numpy(),
            recon_style_cls.detach().cpu().numpy(),
            labels=["Simulation", "Experiment", "Reconstructed"],
            save_dir=config["training"]["stage_2"]["save_dir"],
        )
        fused_latents = np.array(pca_results["reconstructed_latents_pca"])
        experimental_latents = np.array(pca_results["style_latents_pca"])
        obs, pval = frechet_permutation_test_animated(
            fused_latents,
            experimental_latents,
            n_perms=300,
            seed=42,
            figsize=(8, 3.75),
            output_path="frechet_permutation_test_recon_style_style.gif",
            fps=15,
            n_bins=20,
        )
        create_static_summary_plot(
            fused_latents,
            experimental_latents,
            n_perms=300,
            seed=42,
            figsize=(8, 3.75),
            output_path="frechet_permutation_summary_recon_style_style.png",
            n_bins=20,
        )

        fig, ax = plt.subplots(1, 3)  # Sim, Exp, Reconstructed
        recon_sim = model.decoder(
            all_content_features_patches[sample_idx : sample_idx + 1].to(device)
        )[0]
        recon_exp = model.decoder(
            all_style_features_patches[sample_idx : sample_idx + 1].to(device)
        )[0]
        recon_recon = model.decoder(
            all_recon_features_patches[sample_idx : sample_idx + 1].to(device)
        )[0]
        ax[0].imshow(inverse_transform(recon_sim.permute(1, 2, 0).cpu().detach().numpy()))
        ax[1].imshow(inverse_transform(recon_exp.permute(1, 2, 0).cpu().detach().numpy()))
        ax[2].imshow(inverse_transform(recon_recon.permute(1, 2, 0).cpu().detach().numpy()))
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        plt.tight_layout()
        plt.savefig(
            Path(config["training"]["stage_2"]["save_dir"])
            / f"pca_visualization_sample_{sample_idx}.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        # plt.show()


if __name__ == "__main__":
    import sys

    # Allow optional config path as command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    print(f"Loading configuration from: {config_path}")
    main(config_path)

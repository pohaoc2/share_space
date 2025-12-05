import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from share_space.utils import inverse_transform, inverse_transform_batch

state_names = {
    0: "Cell Count",
    # 1: 'OTHER',
    # 2: 'INFLAMMATORY',
    # 3: 'HEALTHY_EPITHELIAL',
    # 4: 'DYSPLASTIC_MALIGNANT',
    # 5: 'FIBROBLAST',
    # 6: 'MUSCLE',
    # 7: 'ENDOTHELIAL'
}
def visualize_first_batch(val_loader, save_dir=None):
    print("Visualizing first batch of the training data...")
    n_viz = 5
    sub_fig_size = 3
    val_iter = iter(val_loader)  # Create iterator once
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
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(Path(save_dir) / f'first_batch_visualization.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    return fig, ax

def visualize_pca(*feature_sets, sample_idx=5, save_dir=None, labels=None, colors=None, markers=None):
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
    default_labels = [
        "Feature Set 1",
        "Feature Set 2",
        "Feature Set 3",
        "Feature Set 4",
        "Feature Set 5",
    ]
    default_colors = ["#AAAAAA", "#993F71", "green", "orange", "purple"]
    default_markers = ["o", "s", "^", "D", "v"]

    if labels is None:
        labels = default_labels[: len(feature_sets)]
    if colors is None:
        colors = default_colors[: len(feature_sets)]
    if markers is None:
        markers = default_markers[: len(feature_sets)]

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
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.75))

    for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        latents = latents_pca_np[label]
        if len(latents) > 1:
            ax.scatter(
                latents[: sample_idx - 1, 0],
                latents[: sample_idx - 1, 1],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                label=label,
                s=40,
            )
            ax.scatter(
                latents[sample_idx + 1 :, 0],
                latents[sample_idx + 1 :, 1],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=40,
            )
        elif len(latents) == 1:
            ax.scatter(
                latents[0:1, 0],
                latents[0:1, 1],
                facecolors=color,
                edgecolors="black",
                marker=marker,
                label=label,
                s=70,
            )

    # Plot first sample of each set (filled markers with black edge)
    for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        latents = latents_pca_np[label]
        if len(latents) > 0:
            ax.scatter(
                latents[sample_idx : sample_idx + 1, 0],
                latents[sample_idx : sample_idx + 1, 1],
                facecolors=color,
                edgecolors="black",
                marker=marker,
                s=70,
                zorder=5,
            )

    # Draw lines connecting the first samples (only if we have exactly 2 feature sets)
    if len(feature_sets) == 2:
        x0, y0 = (
            latents_pca_np[labels[0]][sample_idx : sample_idx + 1, 0],
            latents_pca_np[labels[0]][sample_idx : sample_idx + 1, 1],
        )
        x1, y1 = (
            latents_pca_np[labels[1]][sample_idx : sample_idx + 1, 0],
            latents_pca_np[labels[1]][sample_idx : sample_idx + 1, 1],
        )
        ax.plot([x0, x1], [y0, y1], "k--", linewidth=1.5)

        # Add crosses at 25%, 50%, 75% along the line
        for frac in [0.25, 0.5, 0.75]:
            xc = x0 + (x1 - x0) * frac
            yc = y0 + (y1 - y0) * frac
            ax.scatter(xc, yc, marker="x", color="k", s=60, linewidths=2, zorder=10)

    ax.set_xlabel(
        f"PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})", fontsize=12
    )
    ax.set_ylabel(
        f"PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})", fontsize=12
    )
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = "_".join(labels)
        plt.savefig(
            Path(save_dir) / f"pca_visualization_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()

    # Return PCA results
    result = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "latents_pca": latents_pca,
    }
    plt.close()
    if len(feature_sets) == 2:
        result["content_latents_pca"] = latents_pca[labels[0]]
        result["style_latents_pca"] = latents_pca[labels[1]]
    elif len(feature_sets) == 3:
        result["content_latents_pca"] = latents_pca[labels[0]]
        result["style_latents_pca"] = latents_pca[labels[1]]
        result["reconstructed_latents_pca"] = latents_pca[labels[2]]
    return result


def visualize_linearly_fused_images(
    model, sim_patch_token, exp_patch_token, device, save_dir=None
):
    print(f"Visualizing linearly fused images...")
    model.eval()
    fig, ax = plt.subplots(1, 5, figsize=(3 * 5, 3))
    for i in range(5):
        fused_patches = (4-i) * 0.25 * sim_patch_token + i * 0.25 * exp_patch_token
        pred_fused_recon = model.decoder(fused_patches.to(device))[0]
        ax[i].imshow(
            torch.clamp(
                inverse_transform(pred_fused_recon.permute(1, 2, 0).cpu().detach()),
                0,
                1,
            ).numpy()
        )
        ax[i].axis("off")
        ax[i].set_title(f"{100 - i * 25}% Sim + {i * 25}% Exp")

    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            Path(save_dir) / f"linearly_fused_images.png",
            dpi=300,
            bbox_inches="tight",
            transparent=False,
        )
    else:
        plt.show()
    plt.close()
    return fig, ax


def visualize_images_sampled_from_distribution(
    model, patch_tokens, device, save_dir=None, n_samples=5
):
    print(f"Visualizing images sampled from distribution...")

    model.eval()
    patch_tokens_np = patch_tokens.cpu().detach().numpy()
    B, num_patches, embed_dim = patch_tokens_np.shape

    patch_means = []
    patch_covs = []

    for patch_idx in range(num_patches):
        # Shape: (B, embed_dim)
        patch_tokens_i = patch_tokens_np[:, patch_idx, :]
        mean = np.mean(patch_tokens_i, axis=0)  # (embed_dim,)
        cov = np.cov(patch_tokens_i.T)  # (embed_dim, embed_dim)
        cov += np.eye(embed_dim) * 1e-6

        patch_means.append(mean)
        patch_covs.append(cov)

    # Sample from each patch's distribution with the same shape as patch_tokens
    sampled_patch_tokens = np.zeros((n_samples, num_patches, embed_dim))
    for patch_idx in range(num_patches):
        # Sample B vectors from this patch's distribution
        sampled_patch_tokens[:, patch_idx, :] = np.random.multivariate_normal(
            patch_means[patch_idx], patch_covs[patch_idx], size=n_samples
        )

    sampled_patch_tokens = torch.from_numpy(sampled_patch_tokens).float().to(device)
    pred_fused_recons = model.decoder(sampled_patch_tokens.to(device))
    pred_fused_recons = inverse_transform_batch(pred_fused_recons)
    fig, ax = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    for sample_idx in range(n_samples):
        ax[sample_idx].imshow(
            torch.clamp(
                pred_fused_recons[sample_idx].permute(1, 2, 0).cpu().detach(),
                0,
                1,
            ).numpy()
        )
        ax[sample_idx].axis("off")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            Path(save_dir) / f"images_sampled_from_distribution.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    else:
        plt.show()
    plt.close()
    return fig, ax


def visualize_reconstructed_images(
    exp_imgs,
    pred_exp_imgs,
    sim_imgs,
    pred_sim_imgs,
    save_dir=None,
    n_viz=4,
    states=list(state_names.keys()),
):
    print(f"Visualizing reconstructed images...")
    fig, ax = plt.subplots(n_viz, 4, figsize=(3 * 4, 3 * n_viz))
    for state in states:
        for i in range(n_viz):
            ax[i, 0].imshow(
                torch.clamp(exp_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1)
                .cpu()
                .numpy()
            )
            ax[i, 1].imshow(
                torch.clamp(pred_exp_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1)
                .cpu()
                .numpy()
            )
            ax[i, 2].imshow(
                torch.clamp(sim_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1)
                .cpu()
                .numpy()
            )
            ax[i, 3].imshow(
                torch.clamp(pred_sim_imgs[i].permute(1, 2, 0).cpu().detach(), 0, 1)
                .cpu()
                .numpy()
            )
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")
            if i == 0:
                ax[i, 0].set_title("Exp (Input)", loc="center")
                ax[i, 1].set_title("Exp Reconstructed (Output)", loc="center")
                ax[i, 2].set_title("Sim (Input)", loc="center")
                ax[i, 3].set_title(
                    f"Sim Reconstructed (Output) {state_names[state]}", loc="center"
                )
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.05)
        if save_dir is not None:
            plt.savefig(
                Path(save_dir) / f"reconstructed_images_{state_names[state]}.png",
                dpi=300,
                bbox_inches="tight",
                transparent=True,
            )
        else:
            plt.show()
    plt.close()
    return fig, ax


def visualize_style_transfer(
    style_model,
    val_loader,
    device,
    save_dir,
    n_viz=5,
    states=list(state_names.keys()),
    sample_idx=5,
):
    print(f"Visualizing style transfer...")
    style_model.eval()
    first_batch = next(iter(val_loader))
    fig, ax = plt.subplots(n_viz, 5, figsize=(3 * 5, 3 * n_viz))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for state in states:
        for i in range(n_viz):
            exp_img = first_batch["experimental"][sample_idx + i].unsqueeze(0)  # style
            shuffled_exp_img = first_batch["shuffled_exp"][sample_idx + i].unsqueeze(
                0
            )  # style
            sim_img = first_batch["simulation"][sample_idx + i].unsqueeze(0)  # content
            pred_imgs = (
                style_model(sim_img.to(device), exp_img.to(device))[0]
                .permute(1, 2, 0)
                .cpu()
                .detach()
            )
            shuffled_pred_imgs = (
                style_model(sim_img.to(device), shuffled_exp_img.to(device))[0]
                .permute(1, 2, 0)
                .cpu()
                .detach()
            )
            print(
                f"pred_img[0] and shuffled_pred_imgs[0] is the same: {torch.allclose(pred_imgs[0], shuffled_pred_imgs[0])}"
            )
            ax[i, 0].imshow(
                np.clip(
                    inverse_transform(
                        exp_img[0].permute(1, 2, 0).cpu().detach().numpy()
                    ),
                    0,
                    1,
                )
            )
            ax[i, 1].imshow(
                np.clip(
                    inverse_transform(
                        sim_img[0].permute(1, 2, 0)[..., state].cpu().detach().numpy()
                    ),
                    0,
                    1,
                ),
                cmap="gray",
            )
            ax[i, 2].imshow(np.clip(inverse_transform(pred_imgs.numpy()), 0, 1))
            ax[i, 3].imshow(
                np.clip(
                    inverse_transform(
                        shuffled_exp_img[0].permute(1, 2, 0).cpu().detach().numpy()
                    ),
                    0,
                    1,
                )
            )
            ax[i, 4].imshow(
                np.clip(inverse_transform(shuffled_pred_imgs.numpy()), 0, 1)
            )
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")
            ax[i, 4].axis("off")
            if i == 0:
                ax[i, 0].set_title("Exp (Style)", loc="center")
                ax[i, 1].set_title(f"Sim (Content) {state_names[state]}", loc="center")
                ax[i, 2].set_title("Style Transferred\n(Output)", loc="center")
                ax[i, 3].set_title("Shuffled Style\n(Input)", loc="center")
                ax[i, 4].set_title("Shuffled Style Transferred\n(Output)", loc="center")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.05)

        plt.savefig(
            Path(save_dir) / f"style_transfer_{state_names[state]}.png",
            dpi=300,
            bbox_inches="tight",
        )  # , transparent=True)

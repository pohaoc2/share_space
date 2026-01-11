# evaluation.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights
import numpy as np
from scipy import linalg
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
import copy
import glob
import os
from PIL import Image
from share_space.utils import (
    get_all_samples_from_loader
)

class FIDScore:
    """Frechet Inception Distance for evaluating image quality"""
    def __init__(self, device: str = 'cuda'):
        from torchvision import models
        
        # Use InceptionV3 for FID calculation
        inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
        inception.eval()
        
        for param in inception.parameters():
            param.requires_grad = False
        
        self.inception = inception
        self.device = device
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract InceptionV3 features"""
        # Convert uint8 [0, 255] to float32 [0, 1] if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        images = images.to(self.device)
        
        # Resize to 299x299 (InceptionV3 input size)
        if images.shape[-1] != 299:
            images = torch.nn.functional.interpolate(images, size=(299, 299), 
                                                    mode='bilinear', align_corners=False)
        
        # Normalize for InceptionV3 (ImageNet stats)
        # InceptionV3 expects inputs normalized to [-1, 1] or [0, 1] depending on implementation
        # Standard torchvision InceptionV3 expects [0, 1] range
        
        # Get features
        features = self.inception(images)
        return features.cpu().numpy()
    
    def calculate_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """
        Calculate FID score between real and fake features
        
        Args:
            real_features: features from real images (N, D)
            fake_features: features from generated images (N, D)
        
        Returns:
            FID score (lower is better)
        """
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)
    
    def compute_fid_from_loader(self, model: nn.Module, dataloader: DataLoader) -> float:
        """
        Compute FID score between generated and real experimental images
        
        Args:
            model: trained Sim2Exp model
            dataloader: dataloader with sim-exp pairs
        
        Returns:
            FID score
        """
        model.eval()
        
        real_features_list = []
        fake_features_list = []
        
        for batch in tqdm(dataloader, desc="Computing FID"):
            #sim_imgs = batch['simulation'].to(self.device)
            sim_imgs = batch[0].to(self.device)
            exp_imgs = batch[1].to(self.device)
            exp_imgs = copy.deepcopy(sim_imgs)
            # Generate fake experimental images
            with torch.no_grad():
                fake_imgs, _, _ = model(sim_imgs)
            
            # Extract features
            real_feat = self.extract_features(exp_imgs)
            fake_feat = self.extract_features(fake_imgs)
            
            real_features_list.append(real_feat)
            fake_features_list.append(fake_feat)
        
        # Concatenate all features
        real_features = np.concatenate(real_features_list, axis=0)
        fake_features = np.concatenate(fake_features_list, axis=0)
        
        # Calculate FID
        fid = self.calculate_fid(real_features, fake_features)
        
        return fid


class MetricsEvaluator:
    """Evaluate multiple metrics for image translation"""
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.fid_calculator = FIDScore(device=device)
    
    @torch.no_grad()
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    @torch.no_grad()
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
        """Structural Similarity Index"""
        from pytorch_msssim import ssim
        return ssim(pred, target, data_range=1.0, size_average=True).item()
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader, stage_name: str = 'stage_1') -> dict:
        """
        Comprehensive evaluation of model
        
        Returns:
            dict with FID, PSNR, SSIM scores
        """
        model.eval()
        
        psnr_scores = []
        ssim_scores = []
        
        print("Computing PSNR and SSIM...")
        for batch in tqdm(dataloader):
            sim_imgs = batch['simulation'].to(self.device)
            exp_imgs = batch['experimental'].to(self.device)
            
            with torch.no_grad():
                if stage_name == 'stage_2':
                    pred_imgs = model(sim_imgs, exp_imgs)  
                    # Compute metrics
                    psnr = self.compute_psnr(pred_imgs, exp_imgs)
                    ssim_val = self.compute_ssim(pred_imgs, exp_imgs)
                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim_val)
                else:
                    pred_imgs_sim, _, _ = model(sim_imgs)
                    pred_imgs_exp, _, _ = model(exp_imgs)
                    psnr_val_sim = self.compute_psnr(pred_imgs_sim, sim_imgs)
                    psnr_val_exp = self.compute_psnr(pred_imgs_exp, exp_imgs)
                    ssim_val_sim = self.compute_ssim(pred_imgs_sim, sim_imgs)
                    ssim_val_exp = self.compute_ssim(pred_imgs_exp, exp_imgs)
                    psnr_scores.append(psnr_val_sim)
                    psnr_scores.append(psnr_val_exp)
                    ssim_scores.append(ssim_val_sim)
                    ssim_scores.append(ssim_val_exp)
            
        
        results = {
            #'fid': fid_score,
            'psnr': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores)
        }
        
        return results


def compute_feature_similarity_metrics(model, val_loader, device, verbose=True):
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
    all_sim, all_exp = get_all_samples_from_loader(val_loader)

    with torch.no_grad():
        _, content_features_cls, all_content_patches = model(all_sim.to(device))
        _, style_features_cls, all_style_patches = model(all_exp.to(device))
        results = {}
        content_norm = F.normalize(content_features_cls, p=2, dim=-1)
        style_norm = F.normalize(style_features_cls, p=2, dim=-1)
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
    return results

def main():
    generated_exp_folder = "../../results/all_outputs_val_gt/generated_images"
    real_exp_folder = "../../data/exp"

    generated_images_paths = glob.glob(os.path.join(generated_exp_folder, "*.png"))
    generated_images = np.stack([np.array(Image.open(path)) for path in generated_images_paths[:50]])
    generated_images = generated_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    real_images_paths = glob.glob(os.path.join(real_exp_folder, "*.png"))
    real_images = np.stack([np.array(Image.open(path)) for path in real_images_paths[:50]])
    real_images = real_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    FID_score = FIDScore(device=device)

    # Ensure uint8 dtype
    generated_images_tensor = torch.from_numpy(generated_images.astype(np.uint8))
    real_images_tensor = torch.from_numpy(real_images.astype(np.uint8))

    generated_features = FID_score.extract_features(generated_images_tensor)
    real_features = FID_score.extract_features(real_images_tensor)
    fid_score = FID_score.calculate_fid(generated_features, real_features)
    print(f"FID score: {fid_score}")

if __name__ == "__main__":
    main()
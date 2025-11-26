# evaluation.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights
import numpy as np
from scipy import linalg
from typing import Tuple
from tqdm import tqdm
import copy

def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """Fréchet distance between two Gaussians (vectors: mu, cov)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)
    diff = mu1 - mu2
    # sqrt of product might be numerical; add eps to diag
    cov_prod_sqrt, info = None, None
    try:
        cov_prod_sqrt = linalg.sqrtm(cov1.dot(cov2))
    except Exception:
        cov_prod_sqrt = None

    if np.iscomplexobj(cov_prod_sqrt):
        # numerical noise -> take real part
        cov_prod_sqrt = cov_prod_sqrt.real

    # If still has NaNs or so, regularize covariances
    if cov_prod_sqrt is None or not np.isfinite(cov_prod_sqrt).all():
        cov1 += np.eye(cov1.shape[0]) * eps
        cov2 += np.eye(cov2.shape[0]) * eps
        cov_prod_sqrt = linalg.sqrtm(cov1.dot(cov2)).real

    tr_term = np.trace(cov1 + cov2 - 2.0 * cov_prod_sqrt)
    return float(diff.dot(diff) + tr_term)

def compute_frechet_from_samples(X, Y, eps=1e-6):
    """
    X, Y: arrays shape (N, D) and (M, D)
    """
    muX = np.mean(X, axis=0)
    muY = np.mean(Y, axis=0)
    covX = np.cov(X, rowvar=False)
    covY = np.cov(Y, rowvar=False)
    return frechet_distance(muX, covX, muY, covY, eps=eps)

def frechet_permutation_test(X, Y, n_perms=200, seed=0):
    np.random.seed(seed)
    combined = np.vstack([X, Y])
    n = X.shape[0]
    obs = compute_frechet_from_samples(X, Y)
    count = 0
    for _ in range(n_perms):
        perm = np.random.permutation(combined.shape[0])
        A = combined[perm[:n]]
        B = combined[perm[n:]]
        val = compute_frechet_from_samples(A, B)
        if val >= obs:
            count += 1
    pval = (count + 1) / (n_perms + 1)
    return obs, pval



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
        images = images.to(self.device)
        
        # Resize to 299x299 (InceptionV3 input size)
        if images.shape[-1] != 299:
            images = torch.nn.functional.interpolate(images, size=(299, 299), 
                                                    mode='bilinear', align_corners=False)
        
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

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
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from share_space.utils import (
    get_all_samples_from_loader
)
from scipy.optimize import linear_sum_assignment
from scipy.io import loadmat

class FIDScore:
    """Frechet Distance for evaluating image quality with multiple encoders"""
    def __init__(self, encoder: str = 'inception', device: str = 'cuda'):
        """
        Initialize feature extractor
        
        Args:
            encoder: One of ['inception', 'hoptimus', 'virchow', 'uni']
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.encoder_name = encoder
        
        if encoder == 'inception':
            self.model, self.input_size, self.feature_dim = self._load_inception()
        elif encoder == 'hoptimus':
            self.model, self.input_size, self.feature_dim = self._load_hoptimus()
        elif encoder == 'virchow':
            self.model, self.input_size, self.feature_dim = self._load_virchow()
        elif encoder == 'uni':
            self.model, self.input_size, self.feature_dim = self._load_uni()
        else:
            raise ValueError(f"Unknown encoder: {encoder}. Choose from ['inception', 'hoptimus', 'virchow', 'uni']")
        
        self.model.eval()
        self.model.to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"Loaded {encoder} encoder: input_size={self.input_size}, feature_dim={self.feature_dim}")
    
    def _load_inception(self):
        """Load InceptionV3"""
        from torchvision import models
        from torchvision.models import Inception_V3_Weights
        
        inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.fc = torch.nn.Identity()
        
        return inception, 299, 2048
    
    def _load_hoptimus(self):
        """Load Hoptimus-1"""
        import timm
        
        try:
            model = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0",  # Actual model name
                pretrained=True,
                dynamic_img_size=True
            )
            # Remove classification head
            if hasattr(model, 'head'):
                model.head = torch.nn.Identity()
            
            return model, 224, 768  # Adjust feature_dim based on actual model
        except Exception as e:
            raise ValueError(f"Failed to load Hoptimus-1: {e}. Make sure it's available via timm/HuggingFace")
    
    def _load_virchow(self):
        """Load Virchow-2"""
        import timm
        from timm.layers import SwiGLUPacked
        
        try:
            # Virchow2 requires specific MLP layer and activation function
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU
            )
            
            # Virchow2 outputs (batch, 261, 1280)
            # We need to extract features properly, so wrap it
            class Virchow2FeatureExtractor(torch.nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.model = base_model
                
                def forward(self, x):
                    output = self.model(x)  # (batch, 261, 1280)
                    
                    class_token = output[:, 0]          # (batch, 1280)
                    patch_tokens = output[:, 5:]        # (batch, 256, 1280), skip register tokens 1-4
                    
                    # Concatenate class token and average pool of patch tokens
                    embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # (batch, 2560)
                    
                    return embedding
            
            wrapped_model = Virchow2FeatureExtractor(model)
            
            # Virchow2 uses 224x224 input, outputs 2560-dim features
            return wrapped_model, 224, 2560
            
        except Exception as e:
            raise ValueError(f"Failed to load Virchow-2: {e}. Make sure timm and huggingface-hub are installed")
    
    def _load_uni(self):
        """Load UNI"""
        import timm
        
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True
        )
        # Remove classification head
        if hasattr(model, 'head'):
            model.head = torch.nn.Identity()
        
        return model, 224, 1024
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """Extract features using the selected encoder"""
        # Convert uint8 [0, 255] to float32 [0, 1] if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        all_features = []
        
        # Process in batches to avoid memory issues
        total_batches = (len(images) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(images), batch_size), total=total_batches, desc=f"Extracting {self.encoder_name} features"):
            print(f"Extracting {self.encoder_name} features from batch {i//batch_size} of {total_batches}")
            batch = images[i:i+batch_size].to(self.device)
            
            # Resize to model's expected input size
            if batch.shape[-1] != self.input_size or batch.shape[-2] != self.input_size:
                batch = torch.nn.functional.interpolate(
                    batch, size=(self.input_size, self.input_size), 
                    mode='bilinear', align_corners=False
                )
            
            # Normalize with ImageNet mean and std (standard for most models)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            batch = (batch - mean) / std
            
            # Extract features
            features = self.model(batch)
            all_features.append(features.cpu())
            
            # Clear GPU memory
            del batch, features
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        all_features = torch.cat(all_features, dim=0).numpy()
        
        return all_features
    
    def calculate_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """
        Calculate Frechet Distance between real and fake features
        
        Args:
            real_features: features from real images (N, D)
            fake_features: features from generated images (N, D)
        
        Returns:
            Frechet Distance score (lower is better)
        """
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate Frechet Distance
        diff = mu1 - mu2
        
        # Product of covariances
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fd)
    
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
            sim_imgs = batch[0].to(self.device)
            exp_imgs = batch[1].to(self.device)
            
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

def extract_features_from_paths(image_paths, fid_calculator, batch_size=50):
    """
    Extract features from image paths in batches to avoid memory issues
    
    Args:
        image_paths: list of image file paths
        fid_calculator: FIDScore instance
        batch_size: number of images to process at once
    
    Returns:
        features: np.ndarray of shape (N, 2048)
    """
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load batch
        batch_images = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img_arr = np.array(img)
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr]*3, axis=-1)
            elif img_arr.shape[2] == 4:
                img_arr = img_arr[..., :3]
            batch_images.append(img_arr)
        
        batch_images = np.stack(batch_images)
        batch_images = batch_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        
        # Extract features for this batch
        batch_tensor = torch.from_numpy(batch_images.astype(np.uint8))
        batch_features = fid_calculator.extract_features(batch_tensor)
        
        all_features.append(batch_features)
        
        # Free memory
        del batch_images, batch_tensor, batch_features
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    return all_features


def dice2(gt_masks, pred_masks):
    """
    Ensemble Dice (DICE2) - average Dice coefficient per nucleus.
    
    Args:
        gt_masks: Ground truth binary masks (H, W) with unique labels per nucleus
        pred_masks: Predicted binary masks (H, W) with unique labels per nucleus
    
    Returns:
        float: DICE2 score
    """
    gt_ids = np.unique(gt_masks)[1:]  # exclude background (0)
    
    dice_scores = []
    for gt_id in gt_ids:
        gt_mask = (gt_masks == gt_id)
        
        # Find overlapping predicted nucleus
        overlapping_ids = np.unique(pred_masks[gt_mask])
        # exclude background (0)
        overlapping_ids = overlapping_ids[overlapping_ids != 0]
        if len(overlapping_ids) == 0:
            dice_scores.append(0.0)
            continue
        
        # Use the prediction with maximum overlap
        best_dice = 0.0
        for pred_id in overlapping_ids:
            pred_mask = (pred_masks == pred_id)
            intersection = np.sum(gt_mask & pred_mask)
            dice = 2.0 * intersection / (np.sum(gt_mask) + np.sum(pred_mask))
            best_dice = max(best_dice, dice)
        
        dice_scores.append(best_dice)
    
    return np.mean(dice_scores) if dice_scores else 0.0


def aji(gt_masks, pred_masks):
    """
    Aggregated Jaccard Index (AJI).
    
    Args:
        gt_masks: Ground truth binary masks (H, W) with unique labels per nucleus
        pred_masks: Predicted binary masks (H, W) with unique labels per nucleus
    
    Returns:
        float: AJI score
    """
    gt_ids = np.unique(gt_masks)[1:]
    pred_ids = np.unique(pred_masks)[1:]
    
    # Calculate intersection for all GT-pred pairs
    intersection_sum = 0.0
    gt_used = set()
    
    for pred_id in pred_ids:
        pred_mask = (pred_masks == pred_id)
        overlapping_gt_ids = np.unique(gt_masks[pred_mask])
        # exclude background (0)
        overlapping_gt_ids = overlapping_gt_ids[overlapping_gt_ids != 0]
        
        if len(overlapping_gt_ids) > 0:
            # Find GT with maximum intersection
            max_intersection = 0
            best_gt_id = None
            for gt_id in overlapping_gt_ids:
                gt_mask = (gt_masks == gt_id)
                intersection = np.sum(gt_mask & pred_mask)
                if intersection > max_intersection:
                    max_intersection = intersection
                    best_gt_id = gt_id
            
            intersection_sum += max_intersection
            gt_used.add(best_gt_id)
    
    # Calculate union: all GT pixels + all pred pixels - matched intersections
    gt_area = np.sum(gt_masks > 0)
    pred_area = np.sum(pred_masks > 0)
    union = gt_area + pred_area - intersection_sum
    
    return intersection_sum / union if union > 0 else 0.0


def panoptic_quality(gt_masks, pred_masks, iou_threshold=0.5):
    """
    Panoptic Quality (PQ) = Detection Quality (DQ) × Segmentation Quality (SQ).
    
    Args:
        gt_masks: Ground truth binary masks (H, W) with unique labels per nucleus
        pred_masks: Predicted binary masks (H, W) with unique labels per nucleus
        iou_threshold: IoU threshold for matching (default: 0.5)
    
    Returns:
        dict: {'pq': PQ score, 'dq': DQ score, 'sq': SQ score}
    """
    gt_ids = np.unique(gt_masks)[1:]
    pred_ids = np.unique(pred_masks)[1:]
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gt_id in enumerate(gt_ids):
        gt_mask = (gt_masks == gt_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_masks == pred_id)
            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)
            iou_matrix[i, j] = intersection / union if union > 0 else 0.0
    
    # Find matches with IoU > threshold
    matched_pairs = []
    iou_sum = 0.0
    
    # Greedy matching: match highest IoU pairs first
    matched_gt = set()
    matched_pred = set()
    
    for i in range(len(gt_ids)):
        for j in range(len(pred_ids)):
            if iou_matrix[i, j] > iou_threshold:
                if i not in matched_gt and j not in matched_pred:
                    matched_pairs.append((i, j))
                    iou_sum += iou_matrix[i, j]
                    matched_gt.add(i)
                    matched_pred.add(j)
    
    tp = len(matched_pairs)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    
    # Detection Quality (DQ)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    
    # Segmentation Quality (SQ)
    sq = iou_sum / tp if tp > 0 else 0.0
    
    # Panoptic Quality (PQ)
    pq = dq * sq
    
    return {'pq': pq, 'dq': dq, 'sq': sq}


def fid_evaluation():
    folder_path = "../../results/all_outputs_gt/"
    #folder_path = "../../data/tiles/HE/test/"
    generated_exp_folder = f"{folder_path}/generated_images"
    #real_exp_folder = f"{folder_path}/exp_images"
    real_exp_folder = generated_exp_folder
    # Get image paths
    generated_images_paths = np.array(glob.glob(os.path.join(generated_exp_folder, "*.png")))
    real_images_paths = np.array(glob.glob(os.path.join(real_exp_folder, "*.png")))
    random_idx = np.random.permutation(len(generated_images_paths))
    generated_images_paths = generated_images_paths[random_idx]
    real_images_paths = real_images_paths[random_idx]

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = 'virchow'
    FID_score = FIDScore(device=device, encoder=encoder)
    #generated_images_paths = generated_images_paths[:20]
    #real_images_paths = real_images_paths[:20]
    # Extract features in batches
    mid = 200
    print("\nExtracting features from generated images...")
    generated_features = extract_features_from_paths(
        generated_images_paths[:mid], 
        FID_score, 
        batch_size=100
    )
    
    print("\nExtracting features from real images...")
    real_features = extract_features_from_paths(
        real_images_paths[mid:2*mid], 
        FID_score, 
        batch_size=100
    )
    
    # Calculate FID
    print(f"\nCalculating FID with {encoder} encoder...")
    fid_score = FID_score.calculate_fid(generated_features, real_features)
    print(f"FID score (generated vs real): {fid_score:.2f}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(Image.open(generated_images_paths[i]).convert('RGB'))
        axes[0, i].set_title(f"Generated {i}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(Image.open(real_images_paths[len(real_images_paths)-i-1]).convert('RGB'))
        axes[1, i].set_title(f"Real {i}")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig("fid_comparison.png")
    plt.show()

def segmentation_evaluation():
    guidance_scale = 6
    batch_id = 0
    
    for sample_id in range(1, 5):
        fig, ax = plt.subplots(1, 5, figsize=(30, 6))

        original_images_pattern = f"../../results/all_outputs_w{guidance_scale}_match/exp_images/{batch_id}_{sample_id}*.png"
        original_images_paths = glob.glob(original_images_pattern)
        if not original_images_paths:
            raise FileNotFoundError(f"No files found matching pattern: {original_images_pattern}")
        original_image_path = original_images_paths[0]
        original_image = np.array(Image.open(original_image_path))

        generated_images_pattern = f"../../results/all_outputs_w{guidance_scale}_match/generated_images/{batch_id}_{sample_id}*.png"
        generated_images_paths = glob.glob(generated_images_pattern)
        if not generated_images_paths:
            raise FileNotFoundError(f"No files found matching pattern: {generated_images_pattern}")
        generated_image_path = generated_images_paths[0]
        generated_image = np.array(Image.open(generated_image_path))

        gt_mask_pattern = f"../../results/all_outputs_w{guidance_scale}_match/masks/{batch_id}_{sample_id}*.png"
        gt_mask_paths = glob.glob(gt_mask_pattern)
        if not gt_mask_paths:
            raise FileNotFoundError(f"No files found matching pattern: {gt_mask_pattern}")
        gt_mask_path = gt_mask_paths[0]  # Take the first matching file
        mask_batch_id, mask_sample_id = gt_mask_path.split('/')[-1].split('_')[-2], gt_mask_path.split('/')[-1].split('_')[-1].split('.')[0]
        correct_gt_mask_path = f"../../data/sim/train_{mask_batch_id}_{mask_sample_id}.mask.png"
        gt_mask = np.array(Image.open(correct_gt_mask_path))
        # Resize gt mask to generated image size
        gt_mask = cv2.resize(gt_mask, (generated_image.shape[1], generated_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Convert gt mask to binary
        #gt_mask = (gt_mask > 0).astype(np.uint8)
        # Hovernet output
        mat_path = f"../../results/seg_outputs_w{guidance_scale}_match/mat/{batch_id}_{sample_id}_generated.mat"
        mat_data = loadmat(mat_path)
        hovernet_mask = mat_data['inst_map']
        overlay_image_path = f"../../results/seg_outputs_w{guidance_scale}_match/overlay/{batch_id}_{sample_id}_generated.png"
        overlay_image = np.array(Image.open(overlay_image_path))
        ax[0].imshow(original_image)
        ax[2].imshow(generated_image)
        ax[1].imshow(gt_mask, cmap='gray')
        ax[3].imshow(hovernet_mask, cmap='jet')
        ax[4].imshow(overlay_image)
        # Add title
        ax[0].set_title("Original Image")
        ax[1].set_title("GT Mask")
        ax[2].set_title("Generated Image")
        ax[3].set_title("Hovernet Mask")
        ax[4].set_title("Overlay Image")
        dice_score = dice2(gt_mask, hovernet_mask)
        aji_score = aji(gt_mask, hovernet_mask)
        pq_score = panoptic_quality(gt_mask, hovernet_mask)
        print(f"DICE score: {dice_score}")
        print(f"AJI score: {aji_score}")
        print(f"PQ score: {pq_score}")
        # show metrics in the top right corner
        text = f"DICE: {dice_score:.2f}\nAJI: {aji_score:.2f}\nPQ: {pq_score['pq']:.2f}"
        ax[4].text(
            0.98, 0.02, text,
            fontsize=12,
            ha='right', va='top',
            color='white',
            backgroundcolor='black',
            transform=ax[4].transAxes
        )
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(f"./viz/segmentation_evaluation_{sample_id}.png", dpi=300)
        plt.show()

if __name__ == "__main__":
    #fid_evaluation()
    segmentation_evaluation()
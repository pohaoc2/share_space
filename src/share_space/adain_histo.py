import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
class HistoAdaIN(nn.Module):
    """
    AdaIN for histopathology style transfer with ViT features
    
    Handles both:
    - CLS tokens: (B, D) - global image representation
    - Patch tokens: (B, N, D) - spatial patch embeddings
    
    Strategy:
    - Content: preserve structure (cell locations)
    - Style: transfer appearance (stain, texture)
    """
    def __init__(self, embed_dim, mode='learnable', eps=1e-5):
        """
        Args:
            embed_dim: Embedding dimension
            mode: 'learnable' or 'standard'
                - 'learnable': Learn affine parameters from style features
                - 'standard': Direct statistical matching (classic AdaIN)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        self.eps = eps
        
        if mode == 'learnable':
            # Learn to map style features to affine parameters
            self.style_encoder = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            
            self.gamma_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
                nn.Softplus()  # Ensure positive scale
            )
            
            self.beta_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
    def forward(self, content_features, style_features):
        """
        Args:
            content_features: (B, D) for CLS or (B, N, D) for patches
            style_features: (B, D) for CLS or (B, N, D) for patches
        
        Returns:
            fused_features: same shape as content_features
        """
        is_patches = content_features.dim() == 3
        
        # === 1. Normalize content (preserve structure) ===
        if is_patches:
            # (B, N, D) - normalize across patches, per feature dimension
            content_mean = content_features.mean(dim=-1, keepdim=True)  # (B, N, 1)
            content_std = content_features.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        else:
            # (B, D) - normalize across feature dimension
            content_mean = content_features.mean(dim=1, keepdim=True)  # (B, 1)
            content_std = content_features.std(dim=1, keepdim=True, unbiased=False) + self.eps
        
        content_normalized = (content_features - content_mean) / content_std
        
        # === 2. Extract style statistics ===
        if self.mode == 'learnable':
            # Learn affine parameters from style
            if is_patches:
                # For patches: aggregate style across all patches first
                style_encoded = self.style_encoder(style_features)  # (B, N, D)
                style_global = style_encoded.mean(dim=1)  # (B, N, 1)
                
                # Predict affine parameters
                gamma = self.gamma_net(style_global).unsqueeze(1)  # (B, 1, D)
                beta = self.beta_net(style_global).unsqueeze(1)  # (B, 1, D)
            else:
                # For CLS: use directly
                style_encoded = self.style_encoder(style_features)  # (B, D)
                gamma = self.gamma_net(style_encoded)  # (B, D)
                beta = self.beta_net(style_encoded)  # (B, D)
        
        else:  # standard AdaIN
            # Use style statistics directly
            if is_patches:
                style_mean = style_features.mean(dim=-1, keepdim=True)  # (B, N, 1)
                style_std = style_features.std(dim=-1, keepdim=True, unbiased=False) + self.eps
                gamma = style_std
                beta = style_mean
            else:
                style_mean = style_features.mean(dim=-1, keepdim=True)  # (B, 1)
                style_std = style_features.std(dim=-1, keepdim=True, unbiased=False) + self.eps
                gamma = style_std
                beta = style_mean
        
        # === 3. Apply style transfer ===
        fused_features = content_normalized * gamma + beta
        #fused_features = self.test_net(style_features) + self.test_net2(content_features)
        return fused_features

class WeightedAdaIN(nn.Module):
    """
    Learns weights to combine content and style statistics
    Handles both (B, D) and (B, N, D) inputs
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        
        # Learn dimension-wise weights for combining statistics
        # Input: [content_mean, content_std, style_mean, style_std] = 4*D
        self.alpha_net = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()  # Weight between 0 and 1
        )
    
    def calc_mean_std(self, features):
        """
        Calculate mean and std across the FEATURE dimension (last dim)
        This preserves spatial/patch structure
        
        Args:
            features: (B, D) or (B, N, D)
        Returns:
            mean: (B, 1) or (B, N, 1)
            std: (B, 1) or (B, N, 1)
        """
        # Always calculate along the last dimension (feature dimension)
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        return mean, std
    
    def forward(self, content_features, style_features):
        """
        Args:
            content_features: (B, D) or (B, N, D)
            style_features: (B, D) or (B, N, D)
        Returns:
            fused_features: same shape as content_features
        """
        # Handle 2D case by adding dummy dimension
        is_2d = content_features.dim() == 2
        if is_2d:
            content_features = content_features.unsqueeze(1)  # (B, D) -> (B, 1, D)
            style_features = style_features.unsqueeze(1)  # (B, D) -> (B, 1, D)
        
        B, N, D = content_features.shape
        
        # Get statistics (normalized across feature dimension D)
        content_mean, content_std = self.calc_mean_std(content_features)  # (B, N, 1)
        style_mean, style_std = self.calc_mean_std(style_features)  # (B, N, 1)
        
        # Normalize content (preserve patch structure)
        content_normalized = (content_features - content_mean) / content_std  # (B, N, D)
        
        # Aggregate statistics across patches for the decision network
        # We want to learn a global policy based on overall content/style
        content_mean_global = content_mean.mean(dim=1)  # (B, 1)
        content_std_global = content_std.mean(dim=1)  # (B, 1)
        style_mean_global = style_mean.mean(dim=1)  # (B, 1)
        style_std_global = style_std.mean(dim=1)  # (B, 1)
        
        # Expand to feature dimension for the network
        # Repeat each statistic D times so the network can learn per-dimension weights
        content_mean_expanded = content_mean_global.expand(B, D)  # (B, D)
        content_std_expanded = content_std_global.expand(B, D)  # (B, D)
        style_mean_expanded = style_mean_global.expand(B, D)  # (B, D)
        style_std_expanded = style_std_global.expand(B, D)  # (B, D)
        
        # Concatenate all statistics
        all_stats = torch.cat([
            content_mean_expanded,
            content_std_expanded,
            style_mean_expanded,
            style_std_expanded
        ], dim=-1)  # (B, 4*D)
        
        # Learn per-dimension weights
        alpha = self.alpha_net(all_stats)  # (B, D)
        alpha = alpha.unsqueeze(1)  # (B, 1, D) to broadcast across patches
        
        # Weighted combination of statistics
        target_mean = alpha * style_mean + (1 - alpha) * content_mean  # (B, N, 1)
        target_std = alpha * style_std + (1 - alpha) * content_std  # (B, N, 1)
        
        # Apply learned statistics
        fused_features = content_normalized * target_std + target_mean  # (B, N, D)
        
        # Remove dummy dimension if input was 2D
        if is_2d:
            fused_features = fused_features.squeeze(1)  # (B, 1, D) -> (B, D)
        
        return fused_features

class LearnableAdaIN(nn.Module):
    """
    Advanced learnable AdaIN with both global and local information
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        
        # Networks that see both global (across patches) and local (per feature) statistics
        # Input: [global_mean, global_std, has_local_info] where has_local_info indicates variance in patches
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),  # More capacity
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Softplus()
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def calc_statistics(self, features):
        """
        Calculate comprehensive statistics
        
        Returns:
            global_mean: (B, D)
            global_std: (B, D)
            local_variance: (B, D) - how much variance across patches
        """
        if features.dim() == 2:
            B, D = features.shape
            global_mean = features
            global_std = features.std(dim=0, keepdim=True).expand(B, D) + self.eps
            local_variance = torch.zeros(B, D, device=features.device)
        else:
            B, N, D = features.shape
            global_mean = features.mean(dim=1)  # (B, D)
            global_std = features.std(dim=1, unbiased=False) + self.eps  # (B, D)
            
            # Measure how much patches vary from each other
            patch_means = features.mean(dim=-1)  # (B, N)
            local_variance = patch_means.std(dim=1, unbiased=False)  # (B,)
            local_variance = local_variance.unsqueeze(-1).expand(B, D)  # (B, D)
        
        return global_mean, global_std, local_variance
    
    def calc_local_mean_std(self, features):
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True, unbiased=False) + self.eps
        return mean, std
    
    def forward(self, content_features, style_features):
        is_2d = content_features.dim() == 2
        
        # Get comprehensive style statistics
        style_mean, style_std, style_variance = self.calc_statistics(style_features)
        
        # Normalize content locally
        content_mean_local, content_std_local = self.calc_local_mean_std(content_features)
        content_normalized = (content_features - content_mean_local) / content_std_local
        
        # Predict affine parameters from rich style information
        style_stats = torch.cat([style_mean, style_std, style_variance], dim=-1)  # (B, 3*D)
        gamma = self.gamma_net(style_stats)  # (B, D)
        beta = self.beta_net(style_stats)  # (B, D)
        
        # Apply transformation
        if is_2d:
            fused_features = content_normalized * gamma + beta
        else:
            gamma = gamma.unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)  # (B, 1, D)
            fused_features = content_normalized * gamma + beta
        
        return fused_features

class HistoAdaINSpatialAware (nn.Module):
    """
    Spatially-aware AdaIN with patch-level attention
    
    Better when you want to:
    - Preserve spatial relationships between patches
    - Allow different patches to receive different amounts of style
    - Use cross-attention between content and style patches
    """
    def __init__(self, embed_dim, num_heads=8, use_cross_attention=True, use_content_residual=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cross_attention = use_cross_attention
        self.use_content_residual = use_content_residual

        # Style encoding
        self.style_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        if use_cross_attention:
            # Cross-attention: content queries style
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.1
            )
            self.norm_attn = nn.LayerNorm(embed_dim)
        if use_content_residual:
            # Learnable weight for content preservation
            self.content_weight = nn.Parameter(torch.tensor(0.5))
            # Or per-dimension weights:
            # self.content_weight = nn.Parameter(torch.ones(embed_dim) * 0.5)

        # Affine transformation networks
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Softplus()
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.norm_out = nn.LayerNorm(embed_dim)
    
    def forward(self, content_features, style_features):
        """
        Args:
            content_features: (B, N, D) - content patch embeddings
            style_features: (B, N, D) - style patch embeddings
        
        Returns:
            fused_features: (B, N, D)
        """

        # Handle CLS tokens (expand to single patch)
        if content_features.dim() == 2:
            content_features = content_features.unsqueeze(1)
        if style_features.dim() == 2:
            style_features = style_features.unsqueeze(1)
        
        B, N, D = content_features.shape
        
        # === 1. Normalize content ===
        content_mean = content_features.mean(dim=1, keepdim=True)
        content_std = content_features.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        content_normalized = (content_features - content_mean) / content_std
        
        # === 2. Encode style ===
        style_encoded = self.style_encoder(style_features)  # (B, N, D)
        
        # === 3. Cross-attention (optional) ===
        if self.use_cross_attention:
            # Each content patch attends to relevant style patches
            style_attended, attn_weights = self.cross_attn(
                query=content_normalized,
                key=style_encoded,
                value=style_encoded
            )
            style_for_affine = self.norm_attn(style_attended)
        else:
            # Simple average pooling
            style_for_affine = style_encoded.mean(dim=1, keepdim=True).expand(-1, N, -1)
        
        # === 4. Compute affine parameters (per-patch or global) ===
        gamma = self.gamma_net(style_for_affine)  # (B, N, D)
        beta = self.beta_net(style_for_affine)    # (B, N, D)
        
        # === 5. Apply style transfer ===
        fused_features = content_normalized * gamma + beta
        
        if self.use_content_residual:
            # Add weighted content residual
            fused_features = fused_features + self.content_weight * content_features
        
        fused_features = self.norm_out(fused_features)
        return fused_features


class HistoAdaINHybrid(nn.Module):
    """
    Hybrid AdaIN: Combines CLS-guided global style with patch-level refinement
    
    Use when you have both CLS and patch tokens and want:
    - Global consistency from CLS tokens
    - Local spatial details from patch tokens
    """
    def __init__(self, embed_dim, use_cls_guidance=True, alpha=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cls_guidance = use_cls_guidance
        self.alpha = alpha  # Blending factor
        
        # Global style from CLS
        self.cls_gamma = nn.Linear(embed_dim, embed_dim)
        self.cls_beta = nn.Linear(embed_dim, embed_dim)
        
        # Local style from patches
        self.patch_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        self.patch_gamma = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Softplus()
        )
        self.patch_beta = nn.Linear(embed_dim, embed_dim)
        
        # Gating mechanism to balance global vs local
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, content_patches, style_cls, style_patches):
        """
        Args:
            content_patches: (B, N, D)
            style_cls: (B, D)
            style_patches: (B, N, D)
        
        Returns:
            fused_patches: (B, N, D)
        """
        B, N, D = content_patches.shape
        
        # === 1. Normalize content ===
        content_mean = content_patches.mean(dim=1, keepdim=True)
        content_std = content_patches.std(dim=1, keepdim=True, unbiased=False) + 1e-5
        content_normalized = (content_patches - content_mean) / content_std
        
        # === 2. Global style from CLS ===
        gamma_global = self.cls_gamma(style_cls).unsqueeze(1)  # (B, 1, D)
        beta_global = self.cls_beta(style_cls).unsqueeze(1)
        
        # === 3. Local style from patches ===
        style_encoded = self.patch_encoder(style_patches)  # (B, N, D)
        gamma_local = self.patch_gamma(style_encoded)  # (B, N, D)
        beta_local = self.patch_beta(style_encoded)
        
        # === 4. Blend global and local ===
        # Option A: Learnable gating
        gate_input = torch.cat([
            gamma_global.expand(-1, N, -1),
            gamma_local
        ], dim=-1)
        gate_weights = self.gate(gate_input)  # (B, N, D)
        
        gamma_combined = gate_weights * gamma_global + (1 - gate_weights) * gamma_local
        beta_combined = gate_weights * beta_global + (1 - gate_weights) * beta_local
        
        # Option B: Fixed blending (simpler)
        # gamma_combined = self.alpha * gamma_global + (1 - self.alpha) * gamma_local
        # beta_combined = self.alpha * beta_global + (1 - self.alpha) * beta_local
        
        # === 5. Apply style transfer ===
        fused_patches = content_normalized * gamma_combined + beta_combined
        
        return fused_patches
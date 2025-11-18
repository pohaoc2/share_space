# models/sim2exp_model.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
from share_space.models.vit import VisionTransformer
from share_space.models.decoders import ConvDecoder, TransformerDecoder
from share_space.models.adain import AdaINFusion


class Sim2ExpModel(nn.Module):
    """Complete model for simulation to experimental image translation"""
    def __init__(self, encoder_config: dict, decoder_type: str = 'conv', decoder_config: dict = None):
        super().__init__()
        
        self.encoder = VisionTransformer(**encoder_config)
        
        if decoder_config is None:
            decoder_config = {
                'embed_dim': encoder_config['embed_dim'],
                'img_size': encoder_config['img_size'],
                'patch_size': encoder_config['patch_size'],
                'out_chans': encoder_config['out_chans']
            }
        
        if decoder_type == 'conv':
            self.decoder = ConvDecoder(**decoder_config)
        elif decoder_type == 'transformer':
            self.decoder = TransformerDecoder(**decoder_config)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input image (B, C, H, W)
            mask: optional mask for masked image modeling (B, num_patches)
        Returns:
            reconstructed image, cls_token features, patch_token features
        """
        cls_token, patch_tokens = self.encoder(x)
        
        if isinstance(self.decoder, TransformerDecoder):
            reconstructed = self.decoder(patch_tokens, mask)
        else:
            reconstructed = self.decoder(patch_tokens)
        
        return reconstructed, cls_token, patch_tokens

class StyleTransferModel(nn.Module):
    """Complete model for style transfer"""
    def __init__(self, feature_extractor: nn.Module, decoder: nn.Module, adain: AdaINFusion):
        super().__init__()
        self.encoder = feature_extractor # frozen
        self.decoder = decoder # trainable
        self.adain = adain


    def forward(self, x_content: torch.Tensor, x_style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_content: content image (B, C, H, W)
            x_style: style image (B, C, H, W)
        Returns:
            reconstructed image (B, C, H, W)
        """
        content_features_cls, content_features_patches = self.encoder(x_content)
        style_features_cls, style_features_patches = self.encoder(x_style)
        fused_features_patches = self.adain(content_features_patches, style_features_patches)
        fused_features_cls = self.adain(content_features_cls, style_features_cls)
        reconstructed = self.decoder(fused_features_patches)
        return reconstructed
        
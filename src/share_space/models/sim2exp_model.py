# models/sim2exp_model.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
from share_space.models.vit import VisionTransformer
from share_space.models.decoders import ConvDecoder, TransformerDecoder

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
                'out_chans': encoder_config['in_chans']
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

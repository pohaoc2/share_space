# models/sim2exp_model.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
from share_space.models.vit import VisionTransformer
from share_space.models.decoders import ConvDecoder, TransformerDecoder
import torch.nn.functional as F

class Sim2ExpModel(nn.Module):
    """Complete model with channel projection layers"""
    def __init__(self, encoder_config: dict, decoder_type: str = 'conv', decoder_config: dict = None):
        super().__init__()
        
        # Use a fixed internal channel representation (e.g., 8 channels)
        self.sim_chans = encoder_config['sim_chans']
        self.exp_chans = encoder_config['exp_chans']
        internal_chans = min(self.sim_chans, self.exp_chans)
        
        # Projection layers to convert inputs to internal representation
        if internal_chans == self.sim_chans:
            self.proj_sim_to_internal = nn.Identity()
        else:
            self.proj_sim_to_internal = nn.Conv2d(self.sim_chans, internal_chans, kernel_size=1)
        if internal_chans == self.exp_chans:
            self.proj_exp_to_internal = nn.Identity()
        else:
            self.proj_exp_to_internal = nn.Conv2d(self.exp_chans, internal_chans, kernel_size=1)
        
        # Single encoder for internal representation
        vit_allowed_keys = {
            'img_size',
            'patch_size',
            'embed_dim',
            'depth',
            'num_heads',
            'mlp_ratio',
            'qkv_bias',
            'drop_rate',
            'attn_drop_rate'
        }
        vit_config = {k: v for k, v in encoder_config.items() if k in vit_allowed_keys}
        self.encoder = VisionTransformer(internal_chans=internal_chans, **vit_config)
        
        # Single decoder that outputs internal representation
        if decoder_config is None:
            decoder_config = {
                'embed_dim': encoder_config['embed_dim'],
                'img_size': encoder_config['img_size'],
                'patch_size': encoder_config['patch_size'],
            }
        decoder_config['out_chans'] = internal_chans
        if decoder_type == 'conv':
            decoder_allowed_keys = {
                'embed_dim',
                'img_size',
                'patch_size',
                'out_chans',
                'hidden_dims'
            }
            decoder_config = {k: v for k, v in decoder_config.items() if k in decoder_allowed_keys}
            self.decoder = ConvDecoder(**decoder_config)
        elif decoder_type == 'transformer':
            decoder_allowed_keys = {
                'embed_dim',
                'depth',
                'num_heads',
                'img_size',
                'patch_size',
                'out_chans'
            }
            decoder_config = {k: v for k, v in decoder_config.items() if k in decoder_allowed_keys}
            self.decoder = TransformerDecoder(**decoder_config)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        
        # Projection layers to convert back to original channels
        if internal_chans == self.sim_chans:
            self.proj_internal_to_sim = nn.Identity()
        else:
            self.proj_internal_to_sim = nn.Conv2d(internal_chans, self.sim_chans, kernel_size=1)
        if internal_chans == self.exp_chans:
            self.proj_internal_to_exp = nn.Identity()
        else:
            self.proj_internal_to_exp = nn.Conv2d(internal_chans, self.exp_chans, kernel_size=1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        in_chans = x.shape[1]
        
        # Project to internal representation
        if in_chans == self.sim_chans:
            x_internal = self.proj_sim_to_internal(x)
        elif in_chans == self.exp_chans:
            x_internal = self.proj_exp_to_internal(x)
        else:
            raise ValueError(f"Unsupported input channels: {in_chans}")
        
        # Encode and decode
        cls_token, patch_tokens = self.encoder(x_internal)
        
        if isinstance(self.decoder, TransformerDecoder):
            reconstructed_internal = self.decoder(patch_tokens, mask)
        else:
            reconstructed_internal = self.decoder(patch_tokens)
        
        # Project back to original channels
        if in_chans == self.sim_chans:
            reconstructed = self.proj_internal_to_sim(reconstructed_internal)
        elif in_chans == self.exp_chans:
            reconstructed = self.proj_internal_to_exp(reconstructed_internal)
        else:
            raise ValueError(f"Unsupported input channels: {in_chans}")
        
        return reconstructed, cls_token, patch_tokens

class StyleTransferModel(nn.Module):
    """Complete model for style transfer"""
    def __init__(self, feature_extractor: nn.Module, decoder: nn.Module, adain):
        super().__init__()
        self.encoder = feature_extractor # frozen
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = decoder
        self.decoder.load_state_dict(self.encoder.decoder.state_dict())
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.adain = adain
        print(f"Adain is trainable: {any(p.requires_grad for p in self.adain.parameters())}")
        print(f"Decoder is trainable: {any(p.requires_grad for p in self.decoder.parameters())}")
        print(f"Encoder is trainable: {any(p.requires_grad for p in self.encoder.parameters())}")

    def forward(self, x_content: torch.Tensor, x_style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_content: content image (B, C, H, W)
            x_style: style image (B, C, H, W)
        Returns:
            reconstructed image (B, C, H, W)
        """
        _, content_features_cls, content_features_patches = self.encoder(x_content)
        _, style_features_cls, style_features_patches = self.encoder(x_style)
        
        #_, _, mixed_features_patches = self.encoder(0.5*x_content+x_style)
        fused_features_patches = self.adain(content_features_patches, style_features_patches)
        reconstructed = self.decoder(fused_features_patches)
        #reconstructed = self.decoder(0.5*content_features_patches+style_features_patches)
        return reconstructed
        
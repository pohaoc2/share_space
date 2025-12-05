# models/__init__.py
from models.vit import VisionTransformer

def get_encoder(encoder_type: str, **kwargs):
    """
    Factory function to get encoder by name
    
    Args:
        encoder_type: one of ['vit']
        **kwargs: encoder-specific arguments
    
    Returns:
        encoder model
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 'vit':
        return VisionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_encoder_info():
    """Print available encoders and their parameter counts"""
    import torch
    
    encoders = {
        'ViT-Base': ('vit', {'img_size': 224, 'embed_dim': 768, 'depth': 12, 'num_heads': 12}),
        'ViT-Small': ('vit', {'img_size': 224, 'embed_dim': 384, 'depth': 12, 'num_heads': 6}),
        'ViT-Tiny': ('vit', {'img_size': 224, 'embed_dim': 192, 'depth': 12, 'num_heads': 3}),
    }
    
    print("\nAvailable Encoders:")
    print("-" * 60)
    print(f"{'Name':<25} {'Parameters (M)':<20} {'Type':<15}")
    print("-" * 60)
    
    for name, (encoder_type, config) in encoders.items():
        try:
            model = get_encoder(encoder_type, pretrained=False, **config)
            num_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"{name:<25} {num_params:<20.2f} {encoder_type:<15}")
        except Exception as e:
            print(f"{name:<25} {'Error':<20} {encoder_type:<15}")
    
    print("-" * 60)

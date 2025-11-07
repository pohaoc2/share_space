# models/__init__.py
from models.vit import VisionTransformer
from models.efficientnet import EfficientNetBackbone
from models.mobilenet import MobileNetBackbone
from models.resnet import ResNetBackbone

def get_encoder(encoder_type: str, **kwargs):
    """
    Factory function to get encoder by name
    
    Args:
        encoder_type: one of ['vit', 'efficientnet_b0', 'efficientnet_b3', 
                              'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                              'resnet18', 'resnet34', 'resnet50', 'resnet101']
        **kwargs: encoder-specific arguments
    
    Returns:
        encoder model
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 'vit':
        return VisionTransformer(**kwargs)
    
    elif encoder_type.startswith('efficientnet'):
        variant = encoder_type.split('_')[1] if '_' in encoder_type else 'b0'
        return EfficientNetBackbone(variant=variant, **kwargs)
    
    elif encoder_type.startswith('mobilenet'):
        if 'v3_small' in encoder_type:
            variant = 'v3_small'
        elif 'v3_large' in encoder_type:
            variant = 'v3_large'
        else:
            variant = 'v2'
        return MobileNetBackbone(variant=variant, **kwargs)
    
    elif encoder_type.startswith('resnet'):
        variant = encoder_type.replace('resnet', '')
        return ResNetBackbone(variant=variant, **kwargs)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def get_encoder_info():
    """Print available encoders and their parameter counts"""
    import torch
    
    encoders = {
        'ViT-Base': ('vit', {'img_size': 224, 'embed_dim': 768, 'depth': 12, 'num_heads': 12}),
        'ViT-Small': ('vit', {'img_size': 224, 'embed_dim': 384, 'depth': 12, 'num_heads': 6}),
        'ViT-Tiny': ('vit', {'img_size': 224, 'embed_dim': 192, 'depth': 12, 'num_heads': 3}),
        'EfficientNet-B0': ('efficientnet_b0', {'img_size': 224, 'embed_dim': 768}),
        'EfficientNet-B3': ('efficientnet_b3', {'img_size': 224, 'embed_dim': 768}),
        'MobileNet-V2': ('mobilenet_v2', {'img_size': 224, 'embed_dim': 768}),
        'MobileNet-V3-Small': ('mobilenet_v3_small', {'img_size': 224, 'embed_dim': 768}),
        'MobileNet-V3-Large': ('mobilenet_v3_large', {'img_size': 224, 'embed_dim': 768}),
        'ResNet-18': ('resnet18', {'img_size': 224, 'embed_dim': 768}),
        'ResNet-34': ('resnet34', {'img_size': 224, 'embed_dim': 768}),
        'ResNet-50': ('resnet50', {'img_size': 224, 'embed_dim': 768}),
        'ResNet-101': ('resnet101', {'img_size': 224, 'embed_dim': 768}),
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

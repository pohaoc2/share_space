"""
Test script for StyDiff implementation
Verifies that all components work correctly
"""

import torch
from stydiff.models import (
    AutoKL, 
    AdaINFeatureFusion, 
    DiffusionModel,
    StyDiff,
    StyDiffLoss,
    StyDiffMetrics
)

def test_autokl():
    """Test AutoKL encoder/decoder"""
    print("\n=== Testing AutoKL ===")
    
    model = AutoKL(in_channels=3, latent_channels=4, base_channels=64, num_embeddings=512)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    
    print(f"Input shape: {x.shape}")
    
    # Encode
    z, indices = model.encode(x)
    print(f"Latent shape: {z.shape}")
    print(f"Indices shape: {indices.shape}")
    
    # Decode
    x_recon = model.decode(z)
    print(f"Reconstructed shape: {x_recon.shape}")
    
    # Full forward
    x_recon, z_q, idx = model(x)
    print(f"Full forward - recon: {x_recon.shape}, latent: {z_q.shape}")
    
    print("✓ AutoKL test passed")


def test_adain():
    """Test AdaIN feature fusion"""
    print("\n=== Testing AdaIN ===")
    
    model = AdaINFeatureFusion()
    
    content = torch.randn(2, 3, 256, 256)
    style = torch.randn(2, 3, 256, 256)
    
    print(f"Content shape: {content.shape}")
    print(f"Style shape: {style.shape}")
    
    adapted_features, content_features, style_features = model(content, style)
    
    print(f"Number of adapted feature levels: {len(adapted_features)}")
    for i, feat in enumerate(adapted_features):
        print(f"  Level {i}: {feat.shape}")
    
    print("✓ AdaIN test passed")


def test_diffusion():
    """Test Diffusion Model"""
    print("\n=== Testing Diffusion Model ===")
    
    unet_config = {
        'in_channels': 4,
        'out_channels': 4,
        'model_channels': 128,
        'num_res_blocks': 2,
        'attention_resolutions': [8, 16],
        'channel_mult': (1, 2, 4, 4),
        'num_heads': 4
    }
    
    model = DiffusionModel(unet_config=unet_config, timesteps=1000)
    
    # Test forward (training)
    x0 = torch.randn(2, 4, 64, 64)  # Latent size
    print(f"Input latent shape: {x0.shape}")
    
    noise_pred, noise_target, xt, t = model(x0)
    print(f"Predicted noise shape: {noise_pred.shape}")
    print(f"Target noise shape: {noise_target.shape}")
    print(f"Noisy latent (xt) shape: {xt.shape}")
    print(f"Timesteps shape: {t.shape}")
    
    # Test sampling
    print("Testing sampling...")
    sample = model.sample((2, 4, 64, 64), num_inference_steps=10)
    print(f"Generated sample shape: {sample.shape}")
    
    print("✓ Diffusion Model test passed")


def test_stydiff():
    """Test complete StyDiff model"""
    print("\n=== Testing StyDiff ===")
    
    model = StyDiff(
        img_size=256,
        in_channels=3,
        latent_channels=4,
        autokl_base_channels=64,
        diffusion_model_channels=128,
        num_embeddings=512,
        diffusion_timesteps=100
    )
    
    content = torch.randn(2, 3, 256, 256)
    style = torch.randn(2, 3, 256, 256)
    
    print(f"Content shape: {content.shape}")
    print(f"Style shape: {style.shape}")
    
    # Test training forward pass
    print("\nTesting training forward pass...")
    outputs = model(content, style, return_intermediates=True)
    
    print(f"Output image shape: {outputs['output'].shape}")
    print(f"Content latent shape: {outputs['content_latent'].shape}")
    print(f"Style latent shape: {outputs['style_latent'].shape}")
    print(f"Fused latent shape: {outputs['fused_latent'].shape}")
    print(f"Noise pred shape: {outputs['noise_pred'].shape}")
    
    # Test inference
    print("\nTesting style transfer (inference)...")
    with torch.no_grad():
        output = model.transfer_style(content, style, num_inference_steps=5)
    print(f"Transfer output shape: {output.shape}")
    
    print("✓ StyDiff test passed")


def test_losses():
    """Test loss functions"""
    print("\n=== Testing Loss Functions ===")
    
    criterion = StyDiffLoss(
        content_weight=1.0,
        style_weight=1.0,
        element_weight=1.0,
        diffusion_weight=1.0
    )
    
    # Create dummy tensors
    content_latent = torch.randn(2, 4, 64, 64)
    style_latent = torch.randn(2, 4, 64, 64)
    output_latent = torch.randn(2, 4, 64, 64)
    fused_latent = torch.randn(2, 4, 64, 64)
    noise_pred = torch.randn(2, 4, 64, 64)
    noise_target = torch.randn(2, 4, 64, 64)
    
    losses = criterion(
        content_latent=content_latent,
        style_latent=style_latent,
        output_latent=output_latent,
        fused_latent=fused_latent,
        noise_pred=noise_pred,
        noise_target=noise_target
    )
    
    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("✓ Loss functions test passed")


def test_metrics():
    """Test evaluation metrics"""
    print("\n=== Testing Metrics ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics_eval = StyDiffMetrics(device=device)
    
    # Create dummy images (normalized to [0, 1])
    content = torch.rand(2, 3, 256, 256).to(device)
    style = torch.rand(2, 3, 256, 256).to(device)
    generated = torch.rand(2, 3, 256, 256).to(device)
    
    metrics = metrics_eval.evaluate_batch(content, style, generated)
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("✓ Metrics test passed")


def main():
    """Run all tests"""
    print("="*60)
    print("StyDiff Implementation Test Suite")
    print("="*60)
    
    try:
        test_autokl()
        test_adain()
        test_diffusion()
        test_stydiff()
        test_losses()
        test_metrics()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

# StyDiff: Style Transfer with Diffusion Models

This is an implementation of **StyDiff** based on the paper:
> "StyDiff: a refined style transfer method based on diffusion models" by Yanming Sun & He Meng (2025)

## Architecture Overview

StyDiff combines three key components:

1. **AutoKL Encoder/Decoder**: Encodes images to latent space using a VQ-VAE-like architecture
2. **AdaIN (Adaptive Instance Normalization)**: Fuses content and style features using VGG-16
3. **Diffusion Model**: Generates high-quality stylized images through iterative denoising

### Key Features

- Multi-scale feature extraction using VGG-16
- AdaIN-based style-content fusion
- U-Net diffusion model with attention mechanisms
- Comprehensive loss function with 4 main components

## Project Structure

```
stydiff/
├── models/
│   ├── __init__.py
│   ├── autokl.py          # AutoKL encoder/decoder with vector quantization
│   ├── adain.py           # AdaIN module and VGG feature extractor
│   ├── diffusion.py       # Diffusion model (DDPM-based)
│   ├── stydiff.py         # Complete StyDiff model
│   ├── losses.py          # Loss functions (4 components)
│   └── metrics.py         # Evaluation metrics (SSIM, GM, LPIPS, PD)
├── main.py                # Training and evaluation script
├── config_stydiff.yaml    # Configuration file
└── README.md              # This file
```

## Installation

```bash
pip install torch torchvision pyyaml matplotlib pillow numpy
```

## Usage

### 1. Prepare Data

Organize your data in the following structure:
```
data/
├── content/    # Content images
└── style/      # Style images
```

### 2. Configure

Edit `config_stydiff.yaml` to set your data paths and hyperparameters.

### 3. Train

```bash
python main.py config_stydiff.yaml
```

### 4. Evaluate Only

Set `eval_only: true` in the config and run:
```bash
python main.py config_stydiff.yaml
```

## Model Components

### AutoKL (Latent Space Encoder/Decoder)

- Encodes images to a compressed latent representation
- Uses residual blocks and downsampling/upsampling
- Includes vector quantization (Q function from paper)

### AdaIN (Style-Content Fusion)

- Extracts multi-scale features from VGG-16 (conv1_2, conv2_2, conv3_2, conv4_2)
- Normalizes content features
- Applies style statistics (mean and variance)
- Formula: `F_adapted = γ(F_c) · σ(F_s) + μ(F_s)`

### Diffusion Model

- U-Net architecture with timestep embedding
- Attention blocks at multiple resolutions
- DDPM-based forward and reverse diffusion
- Conditioned on style features from AdaIN

## Loss Functions

The total loss combines four components (from paper Equations 6-9):

1. **Content Loss (L_ImageLatent)**: `||AutoKL(content) - AutoKL(output)||²`
   - Preserves content structure in latent space

2. **Style Loss (L_StyleLatent)**: `||AutoKL(style) - AutoKL(output)||²`
   - Ensures style consistency

3. **Element Loss (L_Element)**: `||A(style, content) - AutoKL(output)||²`
   - Fine-grained element matching
   - A is the AdaIN-fused representation

4. **Diffusion Loss (L_diff)**: `-log P(x_t|X_0, t, A(style, content))`
   - Optimizes noise prediction

Total: `L_total = α·L_ImageLatent + β·L_StyleLatent + γ·L_Element + δ·L_diff`

Default weights (from paper's baseline): α=β=γ=δ=1.0

## Evaluation Metrics

Following the paper, we use 4 metrics:

1. **SSIM** (Structural Similarity): Measures structural preservation
2. **GM** (Gram Matrix distance): Measures style fidelity
3. **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual similarity
4. **PD** (Perceptual Dissimilarity): High-level perceptual difference

## Key Differences from Original Paper

This implementation:
- Uses a simpler AutoKL compared to VDVAE
- Implements standard DDPM instead of more complex variants
- Adds optional perceptual and style feature losses for better results
- Provides configurable loss weights for ablation studies

## Hyperparameter Tuning

To reproduce paper's ablation study (Table 4), modify loss weights in config:

```yaml
# Content focus
loss:
  content_weight: 1.5
  style_weight: 0.7
  element_weight: 0.5
  diffusion_weight: 0.3

# Style focus  
loss:
  content_weight: 0.7
  style_weight: 1.5
  element_weight: 0.5
  diffusion_weight: 0.3
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{sun2025stydiff,
  title={StyDiff: a refined style transfer method based on diffusion models},
  author={Sun, Yanming and Meng, He},
  journal={Scientific Reports},
  volume={15},
  pages={33521},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## Notes

- The model requires significant GPU memory (recommended: 16GB+)
- Training time depends on dataset size and number of diffusion steps
- For faster inference, reduce `num_inference_steps` in `transfer_style()`
- The diffusion model can be replaced with other variants (e.g., Latent Diffusion)

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Reduce image size
- Reduce model channels

**Poor Style Transfer:**
- Increase style_weight
- Adjust style_feature_weight
- Increase number of inference steps

**Poor Content Preservation:**
- Increase content_weight
- Increase perceptual_weight

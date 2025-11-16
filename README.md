[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pohaoc2/share_space/blob/dev/src/share_space/main_colab.ipynb)

# Python project template repository

[![Build status](https://bagherilab.github.io/python_project_template/_badges/build.svg)](https://github.com/bagherilab/python_project_template/actions?query=workflow%3Abuild)
[![Lint status](https://bagherilab.github.io/python_project_template/_badges/lint.svg)](https://github.com/bagherilab/python_project_template/actions?query=workflow%3Alint)
[![Documentation](https://bagherilab.github.io/python_project_template/_badges/documentation.svg)](https://bagherilab.github.io/python_project_template/)
[![Coverage](https://bagherilab.github.io/python_project_template/_badges/coverage.svg)](https://bagherilab.github.io/python_project_template/_coverage/)
[![Code style](https://bagherilab.github.io/python_project_template/_badges/style.svg)](https://github.com/psf/black)
[![Version](https://bagherilab.github.io/python_project_template/_badges/version.svg)](https://pypi.org/project/python_project_template/)
[![License](https://bagherilab.github.io/python_project_template/_badges/license.svg)](https://github.com/bagherilab/python_project_template/blob/main/LICENSE)

# Simulation to Experiment Image Transfer using ViT

This implementation uses a Vision Transformer-based teacher-student framework inspired by DINOv2 to transfer simulation images to experiment-like images.

## Architecture

### Key Components:

1. **ViT Encoder**: Extracts features using Vision Transformer (pretrained on ImageNet)
2. **Generator Head**: Transforms features back to images using transformer decoder
3. **Teacher-Student Framework**: 
   - Student generates exp-like images from simulation
   - Teacher processes real experiment images
   - Teacher updated via EMA of student
4. **Multi-component Loss**:
   - Feature alignment (distillation loss)
   - Pixel reconstruction (L1 loss)
   - Perceptual loss (patch-wise comparison)

## Setup
```bash
# Install dependencies
pip install torch torchvision timm scipy pillow matplotlib tqdm numpy

# Create dummy data (or use your own)
python example_usage.py

# Train model
python sim2exp_transfer.py
```

## Using Your Own Data

Organize your data as:
```
data/
├── simulation/
│   ├── sim_0001.png
│   ├── sim_0002.png
│   └── ...
└── experiment/
    ├── exp_0001.png
    ├── exp_0002.png
    └── ...
```

Images should be paired (same filename index).

## Training

The model trains using:
- **Pretrained ViT**: Better initialization (ImageNet weights)
- **Teacher-Student**: Teacher guides student via feature matching
- **EMA Update**: Teacher = exponential moving average of student
- **Multi-loss**: Balances feature alignment and image quality

## Evaluation

- **FID Score**: Measures distribution similarity between generated and real images
  - FID < 10: Excellent
  - FID < 50: Good
  - FID < 100: Acceptable
  
- **Visual Comparison**: Side-by-side comparison of sim/generated/real

## Customization

Adjust hyperparameters in `main()`:
- `batch_size`: Batch size (default: 16)
- `num_epochs`: Training epochs (default: 50)
- `lr`: Learning rate (default: 1e-4)
- Loss weights in `Sim2ExpLoss`

## Results

After training, you'll get:
- `checkpoints/best_model.pth`: Best model weights
- `training_curves.png`: Training progress
- `comparison_results.png`: Visual comparison
- FID score printed to console
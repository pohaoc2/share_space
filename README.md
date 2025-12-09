# Python project template repository

[![Build status](https://bagherilab.github.io/python_project_template/_badges/build.svg)](https://github.com/bagherilab/python_project_template/actions?query=workflow%3Abuild)
[![Lint status](https://bagherilab.github.io/python_project_template/_badges/lint.svg)](https://github.com/bagherilab/python_project_template/actions?query=workflow%3Alint)
[![Documentation](https://bagherilab.github.io/python_project_template/_badges/documentation.svg)](https://bagherilab.github.io/python_project_template/)
[![Coverage](https://bagherilab.github.io/python_project_template/_badges/coverage.svg)](https://bagherilab.github.io/python_project_template/_coverage/)
[![Code style](https://bagherilab.github.io/python_project_template/_badges/style.svg)](https://github.com/psf/black)
[![Version](https://bagherilab.github.io/python_project_template/_badges/version.svg)](https://pypi.org/project/python_project_template/)
[![License](https://bagherilab.github.io/python_project_template/_badges/license.svg)](https://github.com/bagherilab/python_project_template/blob/main/LICENSE)

## Installation
```bash
# Install dependencies
poetry install

# Activate environments
source $(poetry env info --path)/bin/activate
```

## Architecture
### Stage 1
- Encoder: Vision transformer
- Decoder: Convolutional neural network
### Stage 2
- Style transfer: Adain

## Data availability
### Use example dataset
1. Download [exp.zip](https://drive.google.com/file/d/1rAd7nMFZQJce3i_0IJL4dvClRiWeFAm7/view?usp=sharing) and [sim.zip](https://drive.google.com/file/d/1aOJuoAfKQjQwc5d4s_nRV6T67pHZqyJG/view?usp=sharing)
2. Unzip them in the `data/` folder
3. Ensure the data path in the `src/share_space/config.yaml` pointing to the correct location
### Or, use your own data
Organize your data as:
- TODO: make the naming convention more generalizable
```
data/
├── simulation/
│   ├── train_<wsl_index_1>_<patch_index_1>.png
│   ├── train_<wsl_index_1>_<patch_index_2>.png
│   ├── ...
│   ├── train_<wsl_index_2>_<patch_index_1>.png
│   └── ...
└── experiment/
│   ├── train_<wsl_index_1>_<patch_index_1>.png
│   ├── train_<wsl_index_1>_<patch_index_2>.png
│   ├── ...
│   ├── train_<wsl_index_2>_<patch_index_1>.png
│   └── ...
```

## Running the code

1. Adjust hyperparameters in `src/share_space/config.yaml`:
- `batch_size`: Batch size (default: 8)
- `num_epochs`: Training epochs (default: 5)
- `lr`: Learning rate (default: 1e-4)

2. Run the pipeline:
```
$ python3 main.py
```


## Example output Results
- TODO: Paste example output from console
- `checkpoints/best_model.pth`: Best model weights (TODO: modify for stage 1 and 2)
- `visualization/reconstructed.png`: (TODO: modify for stage 1 and 2)
- Hand-on starter code notebook: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pohaoc2/share_space/blob/main/src/share_space/main_colab.ipynb) (Suggested using GPUs runtime to save training time)
# kaslite

This repo demonstrates a morphogenetic architecture with "soft-landing" seeds. Each sentinel seed now awakens gradually: it shadow-trains as an auto-encoder, blends its output into the trunk using a ramped alpha parameter, then becomes fully active.

## Usage

Run the training script with various configuration options for different dataset types:

```bash
# Basic spirals dataset (2D, default)
python scripts/run_spirals.py

# Two moons dataset with custom parameters
python scripts/run_spirals.py --problem_type moons --n_samples 800 --noise 0.15

# Gaussian clusters dataset
python scripts/run_spirals.py --problem_type clusters --n_samples 600 --n_centers 3 --cluster_std 1.2

# Spherical shell dataset in higher dimensions
python scripts/run_spirals.py --problem_type spheres --input_dim 4 --n_samples 1000 --noise 0.1

# Legacy complex_moons dataset with higher dimensions
python scripts/run_spirals.py --problem_type complex_moons --input_dim 5

# Use CUDA if available
python scripts/run_spirals.py --device cuda --problem_type clusters

# Full configuration example with custom training parameters
python scripts/run_spirals.py --problem_type moons --input_dim 3 --device cuda --blend_steps 200 --shadow_lr 0.002 --batch_size 64 --train_frac 0.8
```

### CLI Arguments

#### General Configuration

- `--problem_type`: Dataset type (`spirals`, `moons`, `clusters`, `spheres`, or `complex_moons`, default: `spirals`)
- `--input_dim`: Input dimensionality (default: `2`)
- `--device`: Device for computation (`cpu` or `cuda`, default: `cpu`)
- `--batch_size`: Training batch size (default: `32`)
- `--train_frac`: Fraction of data used for training (default: `0.7`)

#### Model Training Parameters

- `--blend_steps`: Blend duration for soft-landing (default: `100`)
- `--shadow_lr`: Shadow learning rate (default: `0.001`)
- `--progress_thresh`: Training progress threshold (default: `0.6`)

#### Dataset-Specific Parameters

- `--n_samples`: Number of samples to generate (default: `500`)
- `--noise`: Noise level for moons/spheres datasets (default: `0.1`)
- `--n_centers`: Number of cluster centers for clusters dataset (default: `2`)
- `--cluster_std`: Standard deviation for clusters dataset (default: `1.0`)

### Datasets

- **spirals**: Classic two-spiral classification problem, padded to `input_dim`
- **moons**: Two interleaving half-circles (moons) with configurable noise
- **clusters**: Gaussian blob clusters with configurable centers and spread
- **spheres**: Points on concentric spherical shells with noise
- **complex_moons**: Legacy combination of moons and clusters datasets

## Changelog

- Added support for multiple problem types and arbitrary input dimensions
- Added device selection (CPU/CUDA)
- Hardened soft-landing controller:
  - buffer sampling guard
  - blocked gradient leakage
  - synced status with state
  - CLI-tunable drift warnings

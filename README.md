# kaslite

This repo demonstrates a morphogenetic architecture with "soft-landing" seeds. Each sentinel seed now awakens gradually: it shadow-trains as an auto-encoder, blends its output into the trunk using a ramped alpha parameter, then becomes fully active.

## Usage

Run the training script with various configuration options:

```bash
# Basic spirals dataset (2D, default)
python scripts/run_spirals.py

# Complex moons dataset with higher dimensions
python scripts/run_spirals.py --problem_type complex_moons --input_dim 5

# Use CUDA if available
python scripts/run_spirals.py --device cuda

# Full configuration example
python scripts/run_spirals.py --problem_type complex_moons --input_dim 4 --device cuda --blend_steps 200 --shadow_lr 0.002
```

### CLI Arguments

- `--problem_type`: Dataset type (`spirals` or `complex_moons`, default: `spirals`)
- `--input_dim`: Input dimensionality (default: `2`)
- `--device`: Device for computation (`cpu` or `cuda`, default: `cpu`)
- `--blend_steps`: Blend duration for soft-landing (default: `100`)
- `--shadow_lr`: Shadow learning rate (default: `0.001`)
- `--progress_thresh`: Training progress threshold (default: `0.6`)

### Datasets

- **spirals**: Classic two-spiral classification problem, padded to `input_dim`
- **complex_moons**: Combination of moons and clusters datasets, padded to `input_dim`

## Changelog

- Added support for multiple problem types and arbitrary input dimensions
- Added device selection (CPU/CUDA)
- Hardened soft-landing controller:
  - buffer sampling guard
  - blocked gradient leakage
  - synced status with state
  - CLI-tunable drift warnings

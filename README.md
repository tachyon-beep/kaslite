# kaslite

This repo demonstrates a morphogenetic architecture with "soft-landing" seeds. Each sentinel seed now awakens gradually: it shadow-trains as an auto-encoder, blends its output into the trunk using a ramped alpha parameter, then becomes fully active.

## Usage

### Single Experiment Mode

Run the training script with various configuration options for different dataset types:

```bash
# Basic spirals dataset (2D, default)
python scripts/run_morphogenetic_experiment.py

# Two moons dataset with custom parameters
python scripts/run_morphogenetic_experiment.py --problem_type moons --n_samples 800 --noise 0.15

# Gaussian clusters dataset
python scripts/run_morphogenetic_experiment.py --problem_type clusters --n_samples 600 --n_centers 3 --cluster_std 1.2

# Spherical shell dataset in higher dimensions
python scripts/run_morphogenetic_experiment.py --problem_type spheres --input_dim 4 --n_samples 1000 --noise 0.1

# Legacy complex_moons dataset with higher dimensions
python scripts/run_morphogenetic_experiment.py --problem_type complex_moons --input_dim 5

# Use CUDA if available
python scripts/run_morphogenetic_experiment.py --device cuda --problem_type clusters

# Full configuration example with custom training parameters
python scripts/run_morphogenetic_experiment.py --problem_type moons --input_dim 3 --device cuda --blend_steps 200 --shadow_lr 0.002 --batch_size 64 --train_frac 0.8
```

### Parameter Sweep Mode

Run hyperparameter sweeps using YAML configuration files to automatically test multiple parameter combinations:

```bash
# Run a basic hyperparameter sweep
python scripts/run_morphogenetic_experiment.py --sweep_config examples/basic_sweep.yaml

# Using short flag
python scripts/run_morphogenetic_experiment.py -s examples/architecture_search.yaml

# Run sweep from a directory of YAML files
python scripts/run_morphogenetic_experiment.py --sweep_config examples/

# Combine CLI flags with sweep config (CLI flags become defaults)
python scripts/run_morphogenetic_experiment.py -s examples/quick_test.yaml --device cuda --batch_size 128
```

#### Sweep Configuration Format

YAML sweep configs support multiple value formats:

```yaml
# YAML arrays
num_layers: [4, 8, 16]
seeds_per_layer: [1, 2, 4, 12, 16]

# Comma-separated strings
lr: "0.001,0.003,0.01"
problem_type: "moons,spirals,clusters"

# Single values (no sweep)
device: cuda
batch_size: 64

# Mixed formats work together
hidden_dim: [64, 128]        # Array format
shadow_lr: 1e-4,5e-4,1e-3    # Comma-separated
n_samples: 2000              # Single value
```

#### Grid Expansion

The system creates a Cartesian product of all parameters with multiple values:

- `num_layers: [4, 8]` and `lr: [0.001, 0.003]` → 4 experiments
- `num_layers: [4, 8, 16]`, `seeds_per_layer: [1, 2]`, `lr: [0.001, 0.003]` → 12 experiments

#### CLI Override Behavior

- CLI flags provide default values for all experiments
- YAML parameters override CLI defaults
- Example: `--lr 0.001 -s config.yaml` where config has `lr: [0.003, 0.01]` will test lr=0.003 and lr=0.01

#### Results Organization

Sweep results are organized under `results/sweeps/YYYYMMDD_HHMMSS/`:

```text
results/sweeps/20250616_143022/
├── run_001_a1b2c3d4/
│   └── results_*.log
├── run_002_e5f6g7h8/
│   └── results_*.log
├── ...
└── sweep_summary.csv
```

The `sweep_summary.csv` contains all parameters and results for easy analysis.

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

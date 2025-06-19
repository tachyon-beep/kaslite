# Phase 3: Hyperparameter Sweeps & Automated Optimization - User Guide

## Overview

Phase 3 introduces comprehensive hyperparameter optimization capabilities to the Morphogenetic Engine, including:

- **Enhanced Grid Search**: Parallel execution with Rich progress tracking
- **Bayesian Optimization**: Intelligent parameter search using Optuna
- **Dedicated CLI Tools**: User-friendly command-line interfaces
- **Rich Reporting**: Interactive dashboards and analysis tools
- **CI/CD Automation**: GitHub Actions for automated testing and sweeps

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt

# For Bayesian optimization (optional):
pip install optuna
```

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

### 1. Grid Search

Run a basic grid search using the provided example configuration:

```bash
# Run grid search with parallel execution
morphogenetic-sweep grid --config examples/quick_sweep.yaml --parallel 2

# Or use the Python module
python -m morphogenetic_engine.cli.sweep grid --config examples/quick_sweep.yaml
```

### 2. Quick Validation

For rapid testing and validation:

```bash
# Quick test sweep with minimal configuration
morphogenetic-sweep quick --problem spirals --trials 6
```

### 3. Bayesian Optimization

If you have Optuna installed:

```bash
# Run intelligent parameter search
morphogenetic-sweep bayesian --config examples/bayesian_sweep.yaml --trials 20
```

### 4. Results Analysis

Generate reports and analyze results:

```bash
# Find your sweep results directory
ls results/sweeps/

# Generate summary report
morphogenetic-reports summary --sweep-dir results/sweeps/grid_20250616_143022

# Detailed parameter analysis
morphogenetic-reports analysis --sweep-dir results/sweeps/grid_20250616_143022
```

## Configuration Files

### Grid Search Configuration

```yaml
# examples/basic_grid_search.yaml
sweep_type: "grid"

# Fixed experiment parameters
experiment:
  problem_type: "spirals"
  input_dim: 3
  n_samples: 2000

# Parameter grid to explore
parameters:
  num_layers: [4, 8, 16]
  hidden_dim: [64, 128, 256]
  lr: [1e-4, 1e-3, 1e-2]
  warm_up_epochs: [25, 50]

# Execution settings
execution:
  max_parallel: 4
  timeout_per_trial: 1800  # 30 minutes

# Optimization settings
optimization:
  target_metric: "val_acc"
  direction: "maximize"
```

### Bayesian Optimization Configuration

```yaml
# examples/bayesian_search.yaml
sweep_type: "bayesian"

experiment:
  problem_type: "spirals"
  input_dim: 3

# Parameter search space for Optuna
parameters:
  num_layers:
    type: "categorical"
    choices: [4, 8, 16, 32]
  
  hidden_dim:
    type: "int"
    low: 32
    high: 512
    log: true  # Log-uniform distribution
  
  lr:
    type: "float"
    low: 1e-5
    high: 1e-1
    log: true

execution:
  timeout_per_trial: 1800

optimization:
  target_metric: "val_acc"
  direction: "maximize"
  sampler: "TPE"          # Tree-structured Parzen Estimator
  pruner: "MedianPruner"  # Early stopping for poor trials
```

## Command-Line Interface

### Sweep Commands

```bash
# Grid search
morphogenetic-sweep grid --config CONFIG --parallel N --timeout SECONDS

# Bayesian optimization  
morphogenetic-sweep bayesian --config CONFIG --trials N --timeout SECONDS

# Quick test
morphogenetic-sweep quick --problem TYPE --trials N

# Resume interrupted sweep (planned feature)
morphogenetic-sweep resume --sweep-id ID
```

### Reporting Commands

```bash
# Summary report with top results
morphogenetic-reports summary --sweep-dir PATH --metric METRIC --top N

# Detailed parameter analysis
morphogenetic-reports analysis --sweep-dir PATH --metric METRIC

# Compare multiple sweeps (planned feature)
morphogenetic-reports compare --sweep-dirs PATH1 PATH2 ...

# Export results (planned feature)
morphogenetic-reports export --sweep-dir PATH --format FORMAT
```

## Parameter Space Definition

### Grid Search Parameters

For grid search, parameters can be specified as:

```yaml
parameters:
  # List of values
  lr: [1e-4, 1e-3, 1e-2]
  
  # Comma-separated string
  hidden_dim: "64,128,256"
  
  # Single value (fixed)
  batch_size: 32
  
  # Mixed types
  device: ["cpu", "cuda"]
```

### Bayesian Search Parameters

For Bayesian optimization, use structured parameter definitions:

```yaml
parameters:
  # Categorical choice
  activation:
    type: "categorical"
    choices: ["relu", "tanh", "sigmoid"]
  
  # Integer range
  num_layers:
    type: "int"
    low: 2
    high: 20
    log: false  # Linear scale
  
  # Float range  
  lr:
    type: "float"
    low: 1e-5
    high: 1e-1
    log: true   # Log scale
  
  # Discrete integer
  hidden_dim:
    type: "int"
    low: 32
    high: 512
    log: true   # Powers of 2 preferred
```

## Results and Analysis

### Results Structure

Sweep results are organized as:

```
results/sweeps/
├── grid_20250616_143022/          # Timestamped sweep directory
│   ├── sweep_summary.csv          # CSV summary of all runs
│   ├── sweep_summary.json         # Complete JSON results
│   ├── analysis_report.txt        # Statistical analysis
│   ├── result_run_001_abc123.json # Individual run results
│   └── result_run_002_def456.json
└── bayesian_20250616_150315/
    ├── sweep_summary.csv
    ├── optuna_study.json          # Optuna-specific results
    └── ...
```

### Summary Reports

The summary report shows:

- Top performing configurations
- Parameter importance analysis  
- Performance statistics
- Runtime information

Example output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                    Top 10 Results by val_acc                                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Rank │ Run           │ val_acc │ num_layers │ hidden_dim │ lr     │
├──────┼───────────────┼─────────┼────────────┼────────────┼────────┤
│ 1    │ run_042_a1b2c │ 0.9543  │ 16         │ 256        │ 0.001  │
│ 2    │ run_018_d3e4f │ 0.9521  │ 8          │ 128        │ 0.003  │
│ 3    │ run_095_g5h6i │ 0.9498  │ 16         │ 128        │ 0.001  │
└──────┴───────────────┴─────────┴────────────┴────────────┴────────┘
```

### Parameter Importance

Correlation analysis shows which parameters most affect performance:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                Parameter Correlations with val_acc                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Parameter    │ Correlation │ Strength │
├──────────────┼─────────────┼──────────┤
│ hidden_dim   │ 0.782       │ Strong   │
│ num_layers   │ 0.541       │ Moderate │
│ lr           │ -0.234      │ Weak     │
└──────────────┴─────────────┴──────────┘
```

## Advanced Features

### Parallel Execution

Grid searches support parallel execution:

```bash
# Use 4 parallel workers
morphogenetic-sweep grid --config config.yaml --parallel 4

# Auto-detect CPU count
morphogenetic-sweep grid --config config.yaml --parallel 0
```

### Early Stopping

Configure early stopping for long-running sweeps:

```yaml
execution:
  early_stopping:
    metric: "val_acc"
    patience: 10      # Stop if no improvement for 10 trials
    min_trials: 20    # Minimum trials before early stopping
```

### Custom Metrics

Track additional metrics in your experiments:

```yaml
optimization:
  target_metric: "val_acc"
  secondary_metrics:
    - "runtime"
    - "memory_usage"
    - "convergence_speed"
```

## Integration with Existing Tools

### MLflow Integration

Sweep results automatically integrate with MLflow:

- Each experiment run becomes an MLflow run
- Parameters and metrics are logged
- Artifacts are stored and versioned
- Experiment comparison in MLflow UI

### DVC Integration

Data versioning works seamlessly with sweeps:

- Dataset generation tracked in DVC
- Reproducible data pipelines
- Automatic data dependency management

### TensorBoard Integration

Training metrics available in TensorBoard:

- Real-time training progress
- Loss curves and accuracy plots
- Model architecture visualization

## Troubleshooting

### Common Issues

**"No sweep results found"**
- Check that experiments completed successfully
- Verify the sweep directory path
- Look for error messages in individual result files

**"Optuna not available"**
- Install Optuna: `pip install optuna`
- Optuna is optional for grid search only

**Slow sweep execution**
- Reduce `timeout_per_trial` for faster iterations
- Use fewer parallel workers if memory constrained
- Consider smaller parameter spaces for initial exploration

**Memory issues with parallel execution**
- Reduce `max_parallel` setting
- Monitor system resources during sweeps
- Use `timeout_per_trial` to limit run duration

### Performance Tips

1. **Start small**: Begin with quick sweeps to validate configurations
2. **Use parallel execution**: Leverage multiple CPU cores for grid search
3. **Smart sampling**: Use Bayesian optimization for large parameter spaces
4. **Early stopping**: Configure timeouts to avoid stuck experiments
5. **Resource monitoring**: Watch memory and disk usage during large sweeps

## CI/CD Integration

The provided GitHub Actions workflow (`.github/workflows/ci.yml`) includes:

- **Code quality**: Black formatting, pylint, mypy
- **Unit tests**: pytest with coverage
- **Quick validation**: Fast sweeps on pull requests
- **Nightly sweeps**: Comprehensive parameter exploration
- **Performance monitoring**: Regression detection

To enable:

1. Copy `.github/workflows/ci.yml` to your repository
2. Ensure test data and configurations are available
3. Configure secrets for any external services (MLflow, etc.)
4. Customize sweep configurations for your use case

## What's Next

Phase 3 provides a solid foundation for hyperparameter optimization. Future enhancements may include:

- **Multi-objective optimization**: Optimize multiple metrics simultaneously
- **Distributed execution**: Scale sweeps across multiple machines
- **Advanced pruning**: Smarter early stopping strategies
- **Ensemble selection**: Automatically identify diverse high-performing models
- **Interactive dashboards**: Web-based exploration of results

For more examples and advanced usage, see the `examples/` directory and the comprehensive test suite in `tests/`.

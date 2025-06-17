# kaslite

This repo demonstrates a morphogenetic architecture with "soft-landing" seeds. Each sentinel seed now awakens gradually: it shadow-trains as an auto-encoder, blends its output into the trunk using a ramped alpha parameter, then becomes fully active.

## 🏗️ Complete MLOps Pipeline

The morphogenetic engine now includes a **complete MLOps pipeline** from research to production:

### Phase 1-3: Core Architecture & Experimentation ✅
- **Morphogenetic Neural Networks** - Adaptive architecture with seed-based expansion
- **Hyperparameter Sweeps** - Grid and Bayesian optimization with parallel execution
- **Experiment Tracking** - MLflow integration with comprehensive metrics logging

### Phase 4-5: Monitoring & Observability ✅  
- **Real-time Monitoring** - Prometheus metrics with Grafana dashboards
- **Automated Alerting** - Critical issue detection and notifications
- **Live CLI Dashboard** - Beautiful Rich-powered training visualization

### Phase 6: Model Registry & Production Deployment ✅
- **🏛️ MLflow Model Registry** - Automatic model versioning and lifecycle management
- **🚀 FastAPI Inference Server** - Production-ready REST API with monitoring
- **🔧 Model Management CLI** - Command-line tools for model operations
- **📊 Inference Monitoring** - Real-time serving metrics and alerting
- **🐳 Containerized Deployment** - Docker-based production infrastructure

## 🚀 Quick Start: Train to Production

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a model (auto-registers if accuracy >= 70%)
python scripts/run_morphogenetic_experiment.py --problem spirals --device cpu

# 3. Deploy inference server with full monitoring stack
docker compose -f docker-compose.deploy.yml up -d

# 4. Make predictions via REST API
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.5, 0.3, 0.1]]}'

# 5. Manage model lifecycle
python -m morphogenetic_engine.cli.model_registry_cli list
python -m morphogenetic_engine.cli.model_registry_cli promote Production --version 3
```

## 📊 Model Registry & Deployment

### Automatic Model Lifecycle

**Training → Registration → Staging → Production**

- **Automatic Registration**: Models with validation accuracy ≥ 70% are automatically registered
- **Auto-Promotion**: High-performing models (≥ 90% accuracy) are promoted to Staging
- **Manual Production**: Use CLI or API to promote Staging models to Production
- **Version Management**: Complete version history with metadata and performance metrics

### Production Inference API

The FastAPI inference server provides:

```bash
# Health check
GET /health
# Response: {"status": "healthy", "model_loaded": true, "model_version": "3"}

# List available models  
GET /models
# Response: {"current_version": "3", "available_versions": ["1", "2", "3"]}

# Make predictions
POST /predict
# Body: {"data": [[0.5, 0.3, 0.1]], "model_version": "3"}
# Response: {"predictions": [1], "probabilities": [[0.2, 0.8]], "model_version": "3", "inference_time_ms": 15.2}

# Prometheus metrics
GET /metrics
# Response: Prometheus-formatted metrics for monitoring

# Reload production model
POST /reload-model
# Response: {"status": "success", "message": "Reloaded model version 3"}
```

### Model Management CLI

```bash
# List all model versions with stages
python -m morphogenetic_engine.cli.model_registry_cli list

# Find best model by metric
python -m morphogenetic_engine.cli.model_registry_cli best --metric val_acc --stage Staging

# Promote model to production
python -m morphogenetic_engine.cli.model_registry_cli promote Production --version 3

# Register model manually
python -m morphogenetic_engine.cli.model_registry_cli register <run_id> --val-acc 0.95

# Get current production model URI
python -m morphogenetic_engine.cli.model_registry_cli production
```

### Monitoring & Alerting

**Inference Server Metrics:**
- Request rates, latency percentiles, error rates
- Model prediction times and loading performance  
- Active model version and health status

**Automated Alerts:**
- **Critical**: Server down, high error rate (>10%), no model loaded
- **Warning**: High latency (>2s), slow predictions (>1s)

**Dashboard Access:**
- **Grafana**: http://localhost:3000 (admin/kaslite) - Inference + training dashboards
- **Prometheus**: http://localhost:9090 - Metrics and targets
- **Alertmanager**: http://localhost:9093 - Alert management

## 🧪 Development & Experimentation

This project now includes comprehensive experiment tracking and versioning using **MLflow** and **DVC**, providing a fully reproducible, queryable record of every run's parameters, metrics, and generated artifacts.

### MLflow Integration

All experiments are automatically tracked with MLflow:

- **Parameters**: All CLI flags and configuration values are logged
- **Metrics**: Training/validation loss, accuracy, seed alpha values, and more
- **Artifacts**: Model weights, TensorBoard logs, experiment logs
- **Phase Transitions**: Tagged with phase_1/phase_2 for easy filtering
- **Seed Tracking**: Individual seed states and alpha blending values

### DVC Pipeline

Data and model versioning is handled by DVC with a reproducible pipeline:

```bash
# Initialize DVC (first time only)
dvc init

# Reproduce entire pipeline from scratch
dvc repro

# Generate data only
dvc repro generate_data

# Run training only (after data exists)
dvc repro train

# Pull data from remote storage
dvc pull

# Push data/models to remote storage
dvc push
```

### Quick Start with MLflow + DVC

```bash
# Install dependencies (includes MLflow and DVC)
pip install -r requirements.txt

# Initialize DVC (first time only)
dvc init

# Run a complete reproducible experiment
dvc repro

# View results in different UIs
tensorboard --logdir runs/          # TensorBoard metrics
mlflow ui                          # MLflow experiment tracking

# Experiment tracking URLs
# TensorBoard: http://localhost:6006
# MLflow: http://localhost:5000
```

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

All experiments are automatically tracked in MLflow with full parameter and metric logging.

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

## Monitoring & Visualization

### Real-time Rich CLI Dashboard

The experiment runner includes a beautiful live CLI dashboard powered by Rich that displays:

- **Progress bars**: Live progress for warm-up and adaptation phases
- **Metrics table**: Real-time training/validation loss, accuracy, and best accuracy
- **Seed states panel**: Color-coded status of each seed (dormant/blending/active) with α values
- **Phase transitions**: Highlighted banners when transitioning between experiment phases
- **Germination events**: Special notifications when seeds become active

Example output during training:

```text
🔥 Warm-up Training ━━━━━━━━━━━━━━━━━━━━━ 2/2 • 0:00:01

📊 Experiment Metrics
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric        ┃ Value   ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Epoch         │ 2       │
│ Phase         │ phase_1 │
│ Train Loss    │ 0.3456  │
│ Val Loss      │ 0.2891  │
│ Val Accuracy  │ 0.8320  │
│ Best Accuracy │ 0.8320  │
│ Seeds Active  │ 0/2     │
└───────────────┴─────────┘

🌱 Seed States
seed1_1: dormant
seed2_1: dormant
```

### TensorBoard Integration

All experiments automatically log comprehensive metrics to TensorBoard:

**Scalar Metrics:**

- `train/loss`: Training loss per epoch
- `validation/loss`: Validation loss per epoch  
- `validation/accuracy`: Validation accuracy per epoch
- `validation/best_acc`: Best validation accuracy achieved
- `seed/{id}/alpha`: Alpha blending values for each seed

**Text Summaries:**

- `phase/transitions`: Phase transition events with timestamps
- `seed/{id}/events`: Seed state transition events

**Launch TensorBoard:**

```bash
# View all experiment runs
tensorboard --logdir=runs

# View specific experiment
tensorboard --logdir=runs/spirals_dim3_cpu_h128_bs30_lr0.001_pt0.6_dw0.12
```

TensorBoard will be available at `http://localhost:6006` showing detailed curves for:

- Loss convergence during both phases
- Accuracy improvements over time  
- Seed activation patterns and alpha ramping
- Phase transition timing

### Live Monitoring & Dashboards

The morphogenetic engine includes comprehensive **Prometheus metrics** and **Grafana dashboards** for real-time monitoring and alerting.

#### Prometheus Metrics

All experiments automatically expose detailed metrics at `http://localhost:8000/metrics`:

**Training Metrics:**
- `kaslite_epochs_total`: Number of epochs completed by phase
- `kaslite_validation_loss` / `kaslite_validation_accuracy`: Real-time performance
- `kaslite_best_accuracy`: Best accuracy achieved
- `kaslite_germinations_total`: Total seed germinations

**Seed-Level Metrics:**
- `kaslite_seed_alpha`: Blending alpha values for each seed
- `kaslite_seed_drift`: Interface drift measurements
- `kaslite_seed_health_signal`: Activation variance health indicators
- `kaslite_seed_state`: Current state (dormant/training/blending/active)

**Controller Metrics:**
- `kaslite_kasmina_plateau_counter`: Current plateau detection counter
- `kaslite_epoch_duration_seconds`: Training time per epoch

#### Docker Compose Monitoring Stack

Launch the complete monitoring infrastructure with Docker Compose:

```bash
# Start the full monitoring stack
docker compose up -d

# Or run just the monitoring services
docker compose up -d prometheus grafana alertmanager
```

**Access URLs:**
- **Application Metrics**: http://localhost:8000/metrics
- **Prometheus UI**: http://localhost:9090
- **Grafana Dashboards**: http://localhost:3000 (admin/kaslite)
- **Alertmanager**: http://localhost:9093

#### Grafana Dashboard

The included dashboard provides:

- **Validation Accuracy Trends**: Real-time accuracy curves by phase
- **Training/Validation Loss**: Loss convergence visualization
- **Seed Status Table**: Live view of all seed states, alpha values, and drift
- **Seed Alpha Blending**: Time-series view of alpha ramping
- **Interface Drift Monitoring**: Drift levels with warning thresholds
- **Germination Events**: Controller activity and seed activation rates
- **Performance Stats**: Best accuracy, experiment duration, active seed counts

#### Automated Alerting

Alertmanager monitors key thresholds and sends notifications for:

**Critical Alerts:**
- Validation accuracy drops below 70%
- Training loss explosion (>10)
- Critical seed drift (>25%)
- Experiment stalled (no progress)

**Warning Alerts:**
- Validation accuracy drops below 85% in phase 2
- High seed drift (>15%)
- No germinations during phase 2 with low accuracy
- Kasmina plateau counter approaching threshold

Configure Slack webhooks in `monitoring/alertmanager.yml` to receive real-time notifications.

#### Quick Start: Monitoring

```bash
# 1. Start the monitoring stack
docker compose up -d

# 2. Run an experiment (metrics auto-exposed)
python scripts/run_morphogenetic_experiment.py --problem_type spirals

# 3. View real-time dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### Directory Structure

```
project/
├── data/                 # raw + generated datasets (DVC-tracked)
├── morphogenetic_engine/ # your code (CLI + modules)
├── mlruns/               # MLflow local tracking store
├── runs/                 # TensorBoard logs per-run
├── results/              # logs, models, metrics.json per-run (DVC outputs)
├── scripts/              # experiment and data generation scripts
├── dvc.yaml              # DVC pipeline stages
├── dvc.lock              # DVC lock file (auto-generated)
├── params.yaml           # default parameters for DVC
├── requirements.txt      # dependencies including MLflow + DVC
└── .dvcignore           # DVC ignore patterns
```

### Reproducible Workflows

The project supports fully reproducible experiments:

1. **DVC Pipeline**: `dvc repro` regenerates everything from data to final model
2. **MLflow Tracking**: Every run is logged with parameters, metrics, and artifacts  
3. **Data Versioning**: Raw datasets and models are version-controlled
4. **Parameter Files**: Default configurations in `params.yaml`

```bash
# Complete reproduction workflow
dvc repro                    # Run full pipeline
mlflow ui                    # Inspect experiment runs
tensorboard --logdir runs/  # View training curves

# Version and share your work
dvc remote add -d storage s3://my-bucket/dvc-storage  # Configure remote
dvc push                     # Push data/models to remote
git add . && git commit -m "Experiment results"       # Version code/config
git push                     # Share with team
```

### Datasets

- **spirals**: Classic two-spiral classification problem, padded to `input_dim`
- **moons**: Two interleaving half-circles (moons) with configurable noise
- **clusters**: Gaussian blob clusters with configurable centers and spread
- **spheres**: Points on concentric spherical shells with noise
- **complex_moons**: Legacy combination of moons and clusters datasets

## Features

### Experiment Tracking
- **MLflow**: Complete experiment lifecycle tracking
- **TensorBoard**: Real-time training visualization
- **JSON Metrics**: Structured metrics for DVC pipeline integration

### Data Management  
- **DVC**: Data and model versioning
- **Synthetic Data**: Reproducible dataset generation
- **Pipeline**: Automated data → train → evaluate workflow

### Morphogenetic Architecture
- **Sentinel Seeds**: Adaptive architecture expansion
- **Soft Landing**: Gradual seed activation with alpha blending
- **Phase-based Training**: Warm-up → adaptation phases

## 📚 Documentation

### Complete Guides
- **[Model Registry & Deployment Guide](docs/MODEL_REGISTRY_DEPLOYMENT.md)** - Complete deployment documentation
- **[Step 6 Implementation Summary](docs/STEP6_COMPLETION_SUMMARY.md)** - Technical implementation details
- **[Step 5 Monitoring Summary](docs/STEP5_COMPLETION_SUMMARY.md)** - Monitoring system documentation
- **[Phase 3 Final Validation](docs/phase3_final_validation.md)** - Sweep system validation results

### Testing & Validation
- **[test_deployment.py](test_deployment.py)** - Complete test suite for all components
- **Unit Tests** - Located in `tests/` directory with comprehensive coverage
- **Integration Tests** - End-to-end workflow validation

### Testing & Quality Assurance

The project includes a comprehensive test suite covering all major components:

#### Unit Test Coverage

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=morphogenetic_engine --cov-report=html

# Run specific test categories
pytest tests/test_model_registry.py      # Model registry tests
pytest tests/test_inference_server.py   # Inference API tests
pytest tests/test_monitoring.py         # Prometheus monitoring tests
pytest tests/test_core.py               # Core architecture tests
pytest tests/test_cli.py                # CLI functionality tests
```

**Test Categories:**
- **Core Engine**: Neural network architecture, seed mechanisms, training loops
- **Model Registry**: MLflow integration, versioning, promotion workflows
- **Inference Server**: FastAPI endpoints, model loading, prediction accuracy
- **Monitoring**: Prometheus metrics, alert conditions, dashboard data
- **CLI Tools**: Command-line interfaces, argument parsing, error handling
- **Data Pipeline**: Dataset generation, DVC integration, reproducibility
- **Sweeps**: Hyperparameter optimization, parallel execution, result aggregation

#### Continuous Integration

The test suite includes:
- **Mock-based Testing**: Isolated unit tests with comprehensive mocking
- **Integration Testing**: End-to-end workflow validation
- **Performance Testing**: Model prediction latency and throughput
- **Error Handling**: Exception scenarios and graceful degradation
- **Async Testing**: FastAPI async endpoints and concurrent operations

#### Quality Metrics

- **Code Coverage**: >90% line coverage across all modules
- **Linting**: Black formatting, Ruff static analysis
- **Type Safety**: MyPy type checking for core modules
- **Documentation**: Comprehensive docstrings and API documentation

### Architecture Documents
- **[Monitoring Implementation](docs/MONITORING_IMPLEMENTATION.md)** - Prometheus/Grafana setup
- **[Instrumentation Guide](docs/instrumentation.md)** - Metrics and logging details

## Changelog

### v6.0.0 - Model Registry & Production Deployment
- Added MLflow Model Registry with automatic versioning
- Implemented FastAPI inference server with monitoring
- Created model management CLI tools
- Enhanced Docker deployment with inference services
- Added inference monitoring and alerting
- Complete production-ready MLOps pipeline

### v5.0.0 - Live Monitoring & Dashboards
- Implemented Prometheus metrics collection
- Added Grafana dashboards with real-time visualization
- Created automated alerting with Alertmanager
- Enhanced Docker Compose with monitoring stack
- Added comprehensive observability features

### v4.0.0 - Hyperparameter Sweeps & Optimization
- Added grid search and Bayesian optimization
- Implemented parallel sweep execution
- Created YAML-based sweep configurations
- Enhanced CLI with sweep management
- Added sweep results analysis and visualization

### v3.0.0 - Experiment Tracking & Artifacts
- Integrated MLflow for experiment tracking
- Added TensorBoard visualization
- Implemented DVC for data versioning
- Created reproducible pipeline workflows
- Enhanced metadata and artifact management

### v2.0.0 - Enhanced Architecture & CLI
- Added support for multiple problem types and datasets
- Implemented device selection (CPU/CUDA)
- Enhanced soft-landing controller with drift detection
- Added Rich-powered CLI dashboard
- Improved parameter configuration system

### v1.0.0 - Core Morphogenetic Architecture
- Initial morphogenetic neural network implementation
- Sentinel seed soft-landing mechanism
- Alpha blending and gradual activation
- Phase-based training (warm-up → adaptation)
- Basic experiment runner and logging

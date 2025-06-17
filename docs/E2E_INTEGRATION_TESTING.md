# End-to-End Integration Testing

This document describes the comprehensive end-to-end integration test suite for the Kaslite morphogenetic engine.

## Overview

The E2E integration test validates the complete pipeline from experiment configuration through model training, registration, deployment, and inference serving. It ensures that all components work together correctly in a production-like environment.

## Test Coverage

### 1. Configuration Loading (`test_1_configuration_loading`)
- ✅ YAML configuration parsing
- ✅ Parameter combination generation
- ✅ Sweep configuration validation

### 2. Experiment Execution (`test_2_experiment_execution`)
- ✅ Complete experiment workflow
- ✅ Data generation and processing
- ✅ Model training and evaluation
- ✅ Results logging and storage

### 3. Model Registry Integration (`test_3_model_registry_integration`)
- ✅ MLflow model registration
- ✅ Model versioning and metadata
- ✅ Model promotion workflows

### 4. CLI Functionality (`test_4_sweep_cli_functionality`)
- ✅ Sweep CLI instantiation
- ✅ Configuration loading through CLI
- ✅ Parameter combination generation

### 5. Reports Generation (`test_5_reports_generation`)
- ✅ Reports CLI functionality
- ✅ Results analysis capabilities
- ✅ Output file validation

### 6. Inference Server Deployment (`test_6_inference_server_deployment`)
- ✅ FastAPI server startup
- ✅ Health endpoint (`/health`)
- ✅ Models endpoint (`/models`)
- ✅ Prediction endpoint (`/predict`)

### 7. Monitoring & Metrics (`test_7_monitoring_metrics`)
- ✅ Prometheus metrics endpoint (`/metrics`)
- ✅ Key metrics validation
- ✅ Monitoring data collection

## Running the Tests

### Option 1: Standalone Script
```bash
# Run the complete E2E test suite
python run_e2e_test.py
```

### Option 2: Pytest Integration
```bash
# Run with pytest for detailed output
pytest tests/test_e2e_integration.py -v -s

# Run only the E2E test
pytest tests/test_e2e_integration.py::test_e2e_integration -v -s
```

### Option 3: Manual Test Components
```python
from tests.test_e2e_integration import E2EIntegrationTest

# Run individual test components
test_suite = E2EIntegrationTest()
test_suite.setup()
test_suite.test_1_configuration_loading()
test_suite.test_2_experiment_execution()
# ... etc
test_suite.teardown()
```

## Test Configuration

The E2E test uses a minimal configuration optimized for speed:

```yaml
sweep_type: grid
experiment:
  problem_type: spirals
  device: cpu
  n_samples: 200          # Small dataset
  batch_size: 16
  epochs: 15              # Reduced for speed
  hidden_dim: 32          # Smaller architecture
  num_layers: 2
  seeds_per_layer: 1
parameters:
  learning_rate: [0.01]   # Single value
execution:
  max_time_minutes: 5
  max_parallel: 1
optimization:
  objective: val_acc
  direction: maximize
```

## Test Environment

- **Isolation**: Tests run in a temporary directory to avoid conflicts
- **Port Management**: Uses port 8901 for inference server to avoid conflicts
- **Resource Limits**: Configured for fast execution with minimal resources
- **Cleanup**: Automatic cleanup of test artifacts and processes

## Expected Outcomes

### Success Criteria
- ✅ All 7 test components pass
- ✅ Experiment completes successfully
- ✅ Model training achieves reasonable performance
- ✅ Inference server starts and responds
- ✅ All API endpoints are functional
- ✅ Monitoring metrics are collected

### Performance Expectations
- **Total Test Time**: ~5-10 minutes
- **Experiment Training**: ~2-3 minutes
- **Server Startup**: ~3 seconds
- **API Response Time**: <1 second per request

## Troubleshooting

### Common Issues

**1. Port Conflicts**
```bash
# Check if port 8901 is in use
lsof -i :8901

# Kill any conflicting processes
kill -9 <PID>
```

**2. MLflow Registry Issues**
```bash
# Clear MLflow tracking directory if needed
rm -rf mlruns/
```

**3. Memory Issues**
```bash
# Monitor memory usage during test
htop

# Reduce test parameters if needed
```

**4. Timeout Issues**
```bash
# Increase timeout values in test configuration
# Check system resources and performance
```

### Debug Mode

Run with debug information:
```bash
# Enable verbose output
PYTHONPATH=. python run_e2e_test.py

# Run with pytest debug flags
pytest tests/test_e2e_integration.py -v -s --tb=long
```

## CI/CD Integration

The E2E test can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run E2E Integration Test
  run: |
    python run_e2e_test.py
  timeout-minutes: 15
```

## Test Data

- **Datasets**: Synthetic spirals dataset (200 samples)
- **Models**: Small neural networks (32 hidden units)
- **Experiments**: Single parameter configuration
- **Results**: Temporary files cleaned up automatically

## Validation Metrics

The test validates several key metrics:
- **Training Convergence**: Loss decreases over epochs
- **Model Registration**: Successful MLflow integration
- **API Functionality**: All endpoints respond correctly
- **Monitoring Data**: Prometheus metrics are generated
- **Resource Cleanup**: No lingering processes or files

---

This E2E integration test provides comprehensive validation of the entire Kaslite morphogenetic engine pipeline, ensuring production readiness and system reliability.

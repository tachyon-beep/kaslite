# Model Registry & Deployment Guide

This guide covers the newly implemented Model Registry and Deployment features for the morphogenetic engine.

## Overview

The Model Registry & Deployment system provides:

1. **MLflow Model Registry** - Automatic model registration and versioning
2. **Inference API Server** - FastAPI-based REST API for model serving
3. **Model Management CLI** - Command-line tools for model lifecycle management
4. **Containerized Deployment** - Docker-based deployment with monitoring
5. **Inference Monitoring** - Real-time metrics and alerting for inference performance

## Quick Start

### 1. Train and Register a Model

Run an experiment with automatic model registration:

```bash
# Train a model (automatically registers if val_acc >= 70%)
python scripts/run_morphogenetic_experiment.py --problem spirals --device cpu

# The best models are automatically registered to MLflow Model Registry
# Models with >90% accuracy are auto-promoted to "Staging"
```

### 2. Deploy the Inference Server

```bash
# Start the full deployment stack (inference + monitoring)
docker compose -f docker-compose.deploy.yml up -d

# Check inference server health
curl http://localhost:8080/health

# View available models
curl http://localhost:8080/models
```

### 3. Make Predictions

```bash
# Make a prediction via REST API
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.5, 0.3, 0.1]]}'

# Response includes predictions, probabilities, and model version used
```

## Model Registry Operations

### Using the CLI Tool

```bash
# List all model versions
python -m morphogenetic_engine.cli.model_registry_cli list

# Find the best model by validation accuracy
python -m morphogenetic_engine.cli.model_registry_cli best --metric val_acc

# Promote a specific version to production
python -m morphogenetic_engine.cli.model_registry_cli promote Production --version 3

# Register a model manually
python -m morphogenetic_engine.cli.model_registry_cli register <run_id> --val-acc 0.95

# Get current production model
python -m morphogenetic_engine.cli.model_registry_cli production
```

### Programmatic Usage

```python
from morphogenetic_engine.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Get production model URI
model_uri = registry.get_production_model_uri()

# Load and use the model
import mlflow.pytorch
model = mlflow.pytorch.load_model(model_uri)
```

## Inference API Reference

### Endpoints

- **GET /health** - Health check and model status
- **GET /metrics** - Prometheus metrics
- **GET /models** - List available model versions
- **POST /predict** - Make predictions
- **POST /reload-model** - Reload production model

### Prediction Request Format

```json
{
  "data": [[0.5, 0.3, 0.1], [0.2, 0.8, 0.4]],
  "model_version": "3"  // Optional: specific version to use
}
```

### Prediction Response Format

```json
{
  "predictions": [1, 0],
  "probabilities": [[0.2, 0.8], [0.9, 0.1]],
  "model_version": "3",
  "inference_time_ms": 15.2
}
```

## Model Lifecycle

### Automatic Registration

Models are automatically registered during training when:
- Validation accuracy >= 70% (configurable threshold)
- Training completes successfully
- MLflow run is active

### Promotion Workflow

1. **None → Staging**: Models with >90% accuracy are auto-promoted
2. **Staging → Production**: Manual promotion via CLI or API
3. **Production → Archived**: Automatic when new model is promoted

### Model Metadata

Each registered model includes:
- Performance metrics (accuracy, loss, etc.)
- Training configuration tags
- Problem type and device information
- Seeds activation status
- Creation timestamp and run ID

## Monitoring & Alerting

### Inference Metrics

The system monitors:
- Request rate and latency
- Error rates and status codes
- Model prediction time
- Model load time
- Active model version

### Alerts

Critical alerts:
- Inference server down
- High error rate (>10%)
- No model loaded

Warning alerts:
- High latency (>2s at 95th percentile)
- Slow predictions (>1s model time)

### Dashboards

Access monitoring dashboards:
- **Grafana**: http://localhost:3000 (admin/kaslite)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │  Model Registry │    │ Inference API   │
│   Pipeline      │───▶│   (MLflow)      │───▶│   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Experiment    │    │  Model Storage  │    │   Monitoring    │
│   Tracking      │    │   & Metadata    │    │   & Alerts      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration

### Model Registration Thresholds

Edit `scripts/run_morphogenetic_experiment.py`:

```python
# Register models with >70% accuracy
if final_stats.get('val_acc', 0) >= 0.7:
    # ...register model...

# Auto-promote to Staging if >90% accuracy  
if final_stats.get('val_acc', 0) >= 0.9:
    # ...promote to staging...
```

### Inference Server Settings

Environment variables:
- `MLFLOW_TRACKING_URI` - MLflow server URL
- `LOG_LEVEL` - Logging level (info, debug, warning)
- `PROMETHEUS_MULTIPROC_DIR` - Metrics directory

## Troubleshooting

### Common Issues

1. **No models found**: Ensure MLflow experiments have run and models are registered
2. **Inference server won't start**: Check MLflow connectivity and model availability
3. **High latency**: Consider model optimization or scaling inference instances

### Logs

```bash
# View inference server logs
docker compose -f docker-compose.deploy.yml logs inference

# View all service logs
docker compose -f docker-compose.deploy.yml logs
```

### Health Checks

```bash
# Check inference server health
curl http://localhost:8080/health

# Check if model is loaded
curl http://localhost:8080/models

# Test prediction endpoint
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.0, 0.0]]}'
```

## Production Considerations

### Security
- The current setup is for development/testing
- For production, add authentication, HTTPS, and proper CORS configuration
- Use environment-specific model registries
- Implement proper secrets management

### Scaling
- Use container orchestration (Kubernetes) for production scaling
- Consider model caching strategies for high-traffic scenarios
- Implement request queuing for load management

### Monitoring
- Set up log aggregation (ELK stack, etc.)
- Configure alert notifications (email, Slack, PagerDuty)
- Monitor infrastructure metrics alongside application metrics

## Next Steps

Future enhancements could include:
- A/B testing framework for model comparisons
- Batch inference endpoints
- Model performance drift detection
- Automated model retraining pipelines
- Model explainability endpoints

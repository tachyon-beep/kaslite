# Kaslite Monitoring Quick Test

This example demonstrates the complete monitoring stack.

## Step-by-Step Test

### 1. Start the Monitoring Stack

```bash
# Start all monitoring services
docker compose up -d

# Check all services are running
docker compose ps
```

### 2. Run a Quick Experiment

```bash
# Run a short experiment to generate metrics
python scripts/run_morphogenetic_experiment.py \
    --problem_type spirals \
    --n_samples 1000 \
    --warm_up_epochs 5 \
    --adaptation_epochs 10
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/kaslite)
- **Prometheus**: http://localhost:9090  
- **Raw Metrics**: http://localhost:8000/metrics
- **Alertmanager**: http://localhost:9093

### 4. View Real-time Metrics

In Grafana, you should see:
- ✅ Validation accuracy curves
- ✅ Training/validation loss
- ✅ Seed state transitions
- ✅ Alpha blending progression
- ✅ Interface drift monitoring
- ✅ Germination events

### 5. Test Alerting

Force an alert by running an experiment that will have low accuracy:

```bash
# This should trigger low accuracy alerts
python scripts/run_morphogenetic_experiment.py \
    --problem_type complex_moons \
    --n_samples 500 \
    --warm_up_epochs 2 \
    --adaptation_epochs 3 \
    --acc_threshold 0.99
```

### 6. Cleanup

```bash
# Stop all services
docker compose down

# Remove all data (optional)
docker compose down -v
```

## Expected Metrics

When everything is working, you should see these key metrics in Prometheus:

- `kaslite_epochs_total`
- `kaslite_validation_accuracy`
- `kaslite_seed_alpha`
- `kaslite_seed_drift`
- `kaslite_germinations_total`

## Troubleshooting

- **Port conflicts**: Change ports in docker-compose.yml if needed
- **No metrics**: Check that monitoring is initialized in the experiment script
- **Grafana login**: Default is admin/kaslite
- **Missing data**: Ensure experiment is running and generating activity

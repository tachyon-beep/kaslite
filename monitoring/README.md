# Kaslite Live Monitoring & Dashboards

This directory contains the complete monitoring infrastructure for the Kaslite morphogenetic engine.

## Quick Start

1. **Start the monitoring stack:**
   ```bash
   docker compose up -d
   ```

2. **Run an experiment:**
   ```bash
   python scripts/run_morphogenetic_experiment.py --problem_type spirals --warm_up_epochs 10 --adaptation_epochs 20
   ```

3. **Access dashboards:**
   - Grafana: http://localhost:3000 (admin/kaslite)
   - Prometheus: http://localhost:9090
   - Application metrics: http://localhost:8000/metrics

## Configuration

### Alertmanager (Slack Integration)

To enable Slack notifications:

1. Create a Slack webhook URL
2. Update `monitoring/alertmanager.yml`:
   ```yaml
   global:
     slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
   ```

### Custom Alerts

Add custom alerting rules in `monitoring/rules.yml`:

```yaml
- alert: CustomAlert
  expr: your_prometheus_expression
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Your alert description"
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kaslite App   │───▶│   Prometheus    │───▶│    Grafana      │
│ (Port 8000)     │    │ (Port 9090)     │    │ (Port 3000)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Alertmanager   │
                       │ (Port 9093)     │
                       └─────────────────┘
                                │
                                ▼
                         Slack/Email Alerts
```

## Available Metrics

- **Training**: Epochs, loss, accuracy by phase
- **Seeds**: State, alpha blending, drift, health signals  
- **Germination**: Controller activity and seed activations
- **Performance**: Timing, resource usage

See the Grafana dashboard for a complete visualization of all metrics.

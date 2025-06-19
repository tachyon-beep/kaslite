# Kaslite Scripts Directory

This directory contains automation scripts for managing the Kaslite morphogenetic engine and its monitoring infrastructure.

## 🖥️ Monitoring Scripts

### Main Control Script
- **`monitoring.sh`** - Interactive menu for complete monitoring stack management
  - User-friendly interface with options for all operations
  - Includes status checks, service management, and troubleshooting
  - **Recommended**: Start here for all monitoring operations

### Individual Scripts
- **`start_monitoring.sh`** - Start all monitoring services
  - Prometheus, Grafana, Alertmanager (Docker Compose)
  - TensorBoard (local process)
  - Port conflict detection and service health checks
  
- **`stop_monitoring.sh`** - Stop all monitoring services
  - Graceful shutdown of Docker services and TensorBoard
  - Port cleanup verification
  - Data preservation (volumes maintained)

- **`status_monitoring.sh`** - Check status of all services
  - Service accessibility checks
  - Port usage analysis
  - Data directory verification
  - Process status overview

## 🚀 Quick Start

```bash
# Interactive control center (recommended)
./scripts/monitoring.sh

# Direct commands
./scripts/start_monitoring.sh      # Start everything
./scripts/stop_monitoring.sh       # Stop everything
./scripts/status_monitoring.sh     # Check status
```

## 📊 Services Managed

**Docker Compose Services:**
- **Prometheus** (http://localhost:9090) - Metrics collection
- **Grafana** (http://localhost:3000) - Dashboards (admin/kaslite)
- **Alertmanager** (http://localhost:9093) - Alert management
- **Kaslite App** (http://localhost:8000) - Main application

**Local Processes:**
- **TensorBoard** (http://localhost:6006) - ML visualization

## 🔧 Requirements

- Docker and Docker Compose installed and running
- Python environment with TensorBoard (`pip install tensorboard`)
- Ports 3000, 6006, 8000, 9090, 9093 available
- Bash shell (Linux/macOS/WSL)

## 📁 Data Persistence

- **Prometheus**: Metrics stored in Docker volume (7-day retention)
- **Grafana**: Dashboards and settings in Docker volume
- **TensorBoard**: Logs in `./runs/` directory
- **Experiment Results**: Data in `./results/` and `./mlruns/`

## 🚨 Troubleshooting

### Port Conflicts
```bash
# Check what's using a port
netstat -tuln | grep :9090
lsof -ti:9090

# Kill process on port
kill $(lsof -ti:9090)
```

### Docker Issues
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs prometheus
docker-compose logs grafana

# Restart specific service
docker-compose restart prometheus
```

### Script Permissions
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

## 📚 Additional Documentation

- **Main README**: `../README.md` - Full project documentation
- **Monitoring Guide**: `../monitoring/README.md` - Detailed monitoring setup
- **User Guide**: `../docs/phase3_user_guide.md` - Complete usage guide

## 🔗 Service URLs

When running, access these services:

- **Grafana Dashboard**: http://localhost:3000 (admin/kaslite)
- **Prometheus UI**: http://localhost:9090
- **Alertmanager**: http://localhost:9093  
- **TensorBoard**: http://localhost:6006
- **App Health**: http://localhost:8000/health
- **App Metrics**: http://localhost:8000/metrics

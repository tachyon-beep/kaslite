#!/bin/bash

# Kaslite Monitoring Stack - Status Check Script
# This script checks the status of all monitoring services

echo "📊 Kaslite Monitoring Stack Status"
echo "=================================="

# Get the script directory to ensure we're in the right location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Use 'docker compose' if available, otherwise fall back to 'docker-compose'
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo ""
echo "🐳 Docker Compose Services:"
echo "----------------------------"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found in current directory"
else
    # Show container status
    $DOCKER_COMPOSE ps 2>/dev/null || echo "No Docker Compose services running"
fi

echo ""
echo "🌐 Service Accessibility:"
echo "-------------------------"

# Function to check service availability
check_service() {
    local url=$1
    local name=$2
    local expected_status=${3:-200}
    
    if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" | grep -q "$expected_status"; then
        echo "✅ $name: $url"
        return 0
    else
        echo "❌ $name: $url (not accessible)"
        return 1
    fi
}

# Check web services
check_service "http://localhost:3000/api/health" "Grafana"
check_service "http://localhost:9090/-/ready" "Prometheus"
check_service "http://localhost:9093/-/ready" "Alertmanager"
check_service "http://localhost:8000/health" "Kaslite App"
check_service "http://localhost:6006" "TensorBoard"

echo ""
echo "🔧 Process Status:"
echo "------------------"

# Check for TensorBoard processes
TENSORBOARD_PIDS=$(pgrep -f "tensorboard.*--logdir=runs" 2>/dev/null || true)
if [ -n "$TENSORBOARD_PIDS" ]; then
    echo "✅ TensorBoard: Running (PID: $TENSORBOARD_PIDS)"
else
    echo "❌ TensorBoard: Not running"
fi

# Check for other monitoring processes (non-Docker)
if pgrep -f "prometheus" > /dev/null 2>&1; then
    echo "ℹ️  Prometheus processes detected (may be Docker containers)"
fi

if pgrep -f "grafana" > /dev/null 2>&1; then
    echo "ℹ️  Grafana processes detected (may be Docker containers)"
fi

echo ""
echo "📁 Data Directories:"
echo "--------------------"

# Check for important directories
if [ -d "runs" ]; then
    RUNS_COUNT=$(find runs -name "events.out.tfevents.*" 2>/dev/null | wc -l)
    echo "✅ TensorBoard logs: runs/ ($RUNS_COUNT event files)"
else
    echo "❌ TensorBoard logs: runs/ directory missing"
fi

if [ -d "results" ]; then
    RESULTS_COUNT=$(find results -name "*.json" -o -name "*.log" 2>/dev/null | wc -l)
    echo "✅ Experiment results: results/ ($RESULTS_COUNT files)"
else
    echo "❌ Experiment results: results/ directory missing"
fi

if [ -d "mlruns" ]; then
    MLRUNS_COUNT=$(find mlruns -type d -name "[0-9]*" 2>/dev/null | wc -l)
    echo "✅ MLflow runs: mlruns/ ($MLRUNS_COUNT experiments)"
else
    echo "❌ MLflow runs: mlruns/ directory missing"
fi

echo ""
echo "🔌 Port Usage:"
echo "--------------"

for port in 3000 6006 8000 9090 9093; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        PROCESS=$(lsof -ti:$port 2>/dev/null | head -1)
        if [ -n "$PROCESS" ]; then
            PROCESS_NAME=$(ps -p $PROCESS -o comm= 2>/dev/null || echo "unknown")
            echo "🔴 Port $port: In use by $PROCESS_NAME (PID: $PROCESS)"
        else
            echo "🔴 Port $port: In use"
        fi
    else
        echo "🟢 Port $port: Available"
    fi
done

echo ""
echo "📋 Quick Actions:"
echo "-----------------"
echo "🚀 Start all services:  ./scripts/start_monitoring.sh"
echo "🛑 Stop all services:   ./scripts/stop_monitoring.sh"
echo "🔄 Restart services:    ./scripts/stop_monitoring.sh && ./scripts/start_monitoring.sh"
echo ""
echo "🌐 Direct Links (if running):"
echo "   • Grafana:      http://localhost:3000   (admin/kaslite)"
echo "   • Prometheus:   http://localhost:9090"
echo "   • Alertmanager: http://localhost:9093"
echo "   • TensorBoard:  http://localhost:6006"
echo "   • App Metrics:  http://localhost:8000/metrics"

#!/bin/bash

# Kaslite Monitoring Stack - Start Script
# This script starts all monitoring services: Docker Compose stack + TensorBoard

set -e  # Exit on any error

echo "🚀 Starting Kaslite Monitoring Stack..."

# Get the script directory to ensure we're in the right location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Function to check if a port is already in use
check_port() {
    local port=$1
    local service=$2
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "⚠️  Warning: Port $port is already in use (needed for $service)"
        echo "   You may want to stop existing services first using: ./scripts/stop_monitoring.sh"
        return 1
    fi
    return 0
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    echo "   Waiting for $service_name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "   ✅ $service_name is ready!"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done
    echo "   ❌ $service_name failed to start after $((max_attempts * 2)) seconds"
    return 1
}

echo "📋 Checking prerequisites..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose > /dev/null 2>&1 && ! docker compose version > /dev/null 2>&1; then
    echo "❌ Docker Compose is not available. Please install docker-compose."
    exit 1
fi

# Use 'docker compose' if available, otherwise fall back to 'docker-compose'
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check for port conflicts
echo "🔍 Checking for port conflicts..."
PORTS_OK=true
check_port 3000 "Grafana" || PORTS_OK=false
check_port 9090 "Prometheus" || PORTS_OK=false
check_port 9093 "Alertmanager" || PORTS_OK=false
check_port 8000 "Kaslite App" || PORTS_OK=false
check_port 6006 "TensorBoard" || PORTS_OK=false

if [ "$PORTS_OK" = false ]; then
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Use './scripts/stop_monitoring.sh' to stop existing services."
        exit 1
    fi
fi

echo ""
echo "🐳 Starting Docker Compose services..."
echo "   This will start: Prometheus, Grafana, Alertmanager, and Kaslite App"

# Start the Docker Compose stack
$DOCKER_COMPOSE up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Docker Compose services"
    exit 1
fi

echo ""
echo "⏳ Waiting for services to be ready..."

# Wait for services to be ready
wait_for_service "http://localhost:9090/-/ready" "Prometheus" &
wait_for_service "http://localhost:3000/api/health" "Grafana" &
wait_for_service "http://localhost:9093/-/ready" "Alertmanager" &
wait_for_service "http://localhost:8000/health" "Kaslite App" &

# Wait for all background jobs to complete
wait

echo ""
echo "📊 Starting TensorBoard..."

# Check if runs directory exists
if [ ! -d "runs" ]; then
    echo "   Creating runs directory..."
    mkdir -p runs
fi

# Start TensorBoard in the background
nohup tensorboard --logdir=runs --port=6006 --bind_all > /dev/null 2>&1 &
TENSORBOARD_PID=$!

# Save TensorBoard PID for stopping later
echo "$TENSORBOARD_PID" > .tensorboard_pid

# Wait a moment for TensorBoard to start
sleep 3

# Check if TensorBoard started successfully
if ps -p $TENSORBOARD_PID > /dev/null; then
    echo "   ✅ TensorBoard started successfully (PID: $TENSORBOARD_PID)"
else
    echo "   ⚠️  TensorBoard may have failed to start"
fi

echo ""
echo "🎉 Kaslite Monitoring Stack is now running!"
echo ""
echo "📱 Access your dashboards:"
echo "   • Grafana:      http://localhost:3000   (admin/kaslite)"
echo "   • Prometheus:   http://localhost:9090"
echo "   • Alertmanager: http://localhost:9093"
echo "   • TensorBoard:  http://localhost:6006"
echo "   • App Metrics:  http://localhost:8000/metrics"
echo "   • App Health:   http://localhost:8000/health"
echo ""
echo "🧪 Run an experiment:"
echo "   python scripts/run_morphogenetic_experiment.py --problem_type spirals --warm_up_epochs 10"
echo ""
echo "🛑 To stop all services:"
echo "   ./scripts/stop_monitoring.sh"
echo ""

# Show container status
echo "📦 Container status:"
$DOCKER_COMPOSE ps

#!/bin/bash

# Kaslite Monitoring Stack - Stop Script
# This script stops all monitoring services: Docker Compose stack + TensorBoard

set -e  # Exit on any error

echo "🛑 Stopping Kaslite Monitoring Stack..."

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

echo "🐳 Stopping Docker Compose services..."

# Stop and remove containers
$DOCKER_COMPOSE down

if [ $? -eq 0 ]; then
    echo "   ✅ Docker Compose services stopped"
else
    echo "   ⚠️  Some Docker services may have failed to stop properly"
fi

echo ""
echo "📊 Stopping TensorBoard..."

# Stop TensorBoard if PID file exists
if [ -f ".tensorboard_pid" ]; then
    TENSORBOARD_PID=$(cat .tensorboard_pid)
    if ps -p $TENSORBOARD_PID > /dev/null 2>&1; then
        kill $TENSORBOARD_PID
        if [ $? -eq 0 ]; then
            echo "   ✅ TensorBoard stopped (PID: $TENSORBOARD_PID)"
        else
            echo "   ⚠️  Failed to stop TensorBoard process $TENSORBOARD_PID"
        fi
    else
        echo "   ℹ️  TensorBoard process $TENSORBOARD_PID was not running"
    fi
    rm -f .tensorboard_pid
else
    # Fallback: try to find and kill any tensorboard processes
    TENSORBOARD_PIDS=$(pgrep -f "tensorboard.*--logdir=runs" 2>/dev/null || true)
    if [ -n "$TENSORBOARD_PIDS" ]; then
        echo "   🔍 Found TensorBoard processes: $TENSORBOARD_PIDS"
        echo "$TENSORBOARD_PIDS" | xargs kill 2>/dev/null || true
        echo "   ✅ TensorBoard processes stopped"
    else
        echo "   ℹ️  No TensorBoard processes found"
    fi
fi

echo ""
echo "🧹 Cleaning up..."

# Optional: Remove volumes (uncomment if you want to clean up data)
# echo "🗑️  Removing Docker volumes..."
# $DOCKER_COMPOSE down -v

# Check if any monitoring processes are still running
echo "🔍 Checking for remaining processes..."

REMAINING_PROCESSES=""
if pgrep -f "prometheus" > /dev/null 2>&1; then
    REMAINING_PROCESSES="$REMAINING_PROCESSES prometheus"
fi
if pgrep -f "grafana" > /dev/null 2>&1; then
    REMAINING_PROCESSES="$REMAINING_PROCESSES grafana"
fi
if pgrep -f "alertmanager" > /dev/null 2>&1; then
    REMAINING_PROCESSES="$REMAINING_PROCESSES alertmanager"
fi
if pgrep -f "tensorboard" > /dev/null 2>&1; then
    REMAINING_PROCESSES="$REMAINING_PROCESSES tensorboard"
fi

if [ -n "$REMAINING_PROCESSES" ]; then
    echo "   ⚠️  Some monitoring processes may still be running:$REMAINING_PROCESSES"
    echo "   You may need to kill them manually if they're not Docker containers"
else
    echo "   ✅ No monitoring processes detected"
fi

# Check if ports are now free
echo ""
echo "🔍 Checking port availability..."
PORTS_FREE=true

for port in 3000 9090 9093 8000 6006; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "   ⚠️  Port $port is still in use"
        PORTS_FREE=false
    else
        echo "   ✅ Port $port is free"
    fi
done

echo ""
if [ "$PORTS_FREE" = true ]; then
    echo "🎉 Kaslite Monitoring Stack stopped successfully!"
else
    echo "⚠️  Monitoring stack stopped, but some ports are still in use."
    echo "   You may need to manually stop processes or wait a moment for them to fully shut down."
fi

echo ""
echo "🚀 To start the monitoring stack again:"
echo "   ./scripts/start_monitoring.sh"
echo ""
echo "📊 To view any existing data when you restart:"
echo "   • Grafana dashboards and settings will be preserved"
echo "   • Prometheus data will be preserved (7-day retention)"
echo "   • TensorBoard logs in ./runs/ will remain available"
echo ""
echo "🗑️  To completely clean up all data (optional):"
echo "   docker-compose down -v  # Remove all volumes and data"

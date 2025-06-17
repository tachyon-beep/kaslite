#!/bin/bash

# Kaslite Monitoring Stack - Main Control Script
# Interactive menu for managing the complete monitoring infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    Kaslite Monitoring Control Center                 ║"
echo "║                                                                      ║"
echo "║  Complete monitoring stack: Prometheus, Grafana, Alertmanager,      ║"
echo "║  TensorBoard, and application metrics                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to run other scripts
run_script() {
    local script_name=$1
    local script_path="$SCRIPT_DIR/$script_name"
    
    if [ -f "$script_path" ] && [ -x "$script_path" ]; then
        echo -e "${YELLOW}Running $script_name...${NC}"
        echo ""
        "$script_path"
    else
        echo -e "${RED}Error: $script_name not found or not executable${NC}"
        echo "Expected location: $script_path"
    fi
}

# Function to show menu
show_menu() {
    echo ""
    echo -e "${GREEN}What would you like to do?${NC}"
    echo ""
    echo "  1) 🚀 Start monitoring stack (Prometheus, Grafana, Alertmanager, TensorBoard)"
    echo "  2) 🛑 Stop monitoring stack"
    echo "  3) 📊 Check status of all services"
    echo "  4) 🔄 Restart monitoring stack"
    echo "  5) 🌐 Show service URLs"
    echo "  6) 🧪 Run a quick test experiment"
    echo "  7) 📋 Show logs from Docker services"
    echo "  8) 🗑️  Clean up all monitoring data"
    echo "  9) ❓ Help and documentation"
    echo "  q) Quit"
    echo ""
    echo -n "Enter your choice [1-9, q]: "
}

# Function to show URLs
show_urls() {
    echo -e "${GREEN}Service URLs:${NC}"
    echo ""
    echo "📊 Dashboards and Interfaces:"
    echo "   • Grafana Dashboard:  http://localhost:3000   (admin/kaslite)"
    echo "   • Prometheus Targets: http://localhost:9090"
    echo "   • Alertmanager:       http://localhost:9093"
    echo "   • TensorBoard:        http://localhost:6006"
    echo ""
    echo "🔧 API Endpoints:"
    echo "   • App Health Check:   http://localhost:8000/health"
    echo "   • App Metrics:        http://localhost:8000/metrics"
    echo "   • Prometheus Metrics: http://localhost:9090/metrics"
    echo ""
    echo "💡 Tip: Use Ctrl+Click (or Cmd+Click on Mac) to open links in your browser"
}

# Function to run test experiment
run_test_experiment() {
    echo -e "${GREEN}Running a quick test experiment...${NC}"
    echo ""
    
    cd "$SCRIPT_DIR/.."
    
    if [ -f "scripts/run_morphogenetic_experiment.py" ]; then
        echo "Starting experiment: spirals with minimal epochs for testing"
        python scripts/run_morphogenetic_experiment.py \
            --problem_type spirals \
            --n_samples 500 \
            --warm_up_epochs 5 \
            --adaptation_epochs 10 \
            --batch_size 32
        echo ""
        echo -e "${GREEN}✅ Test experiment completed!${NC}"
        echo "📊 Check TensorBoard (http://localhost:6006) to see the results"
    else
        echo -e "${RED}❌ Experiment script not found${NC}"
    fi
}

# Function to show logs
show_logs() {
    echo -e "${GREEN}Recent logs from Docker services:${NC}"
    echo ""
    
    cd "$SCRIPT_DIR/.."
    
    # Use 'docker compose' if available, otherwise fall back to 'docker-compose'
    if docker compose version > /dev/null 2>&1; then
        DOCKER_COMPOSE="docker compose"
    else
        DOCKER_COMPOSE="docker-compose"
    fi
    
    echo "=== Kaslite App Logs ==="
    $DOCKER_COMPOSE logs --tail=20 kaslite-app 2>/dev/null || echo "Kaslite app not running"
    
    echo ""
    echo "=== Prometheus Logs ==="
    $DOCKER_COMPOSE logs --tail=10 prometheus 2>/dev/null || echo "Prometheus not running"
    
    echo ""
    echo "=== Grafana Logs ==="
    $DOCKER_COMPOSE logs --tail=10 grafana 2>/dev/null || echo "Grafana not running"
}

# Function to clean up data
cleanup_data() {
    echo -e "${YELLOW}This will remove all monitoring data including:${NC}"
    echo "  • Docker volumes (Prometheus data, Grafana settings)"
    echo "  • TensorBoard logs"
    echo "  • Experiment results (optional)"
    echo ""
    echo -e "${RED}⚠️  This action cannot be undone!${NC}"
    echo ""
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " confirm
    
    if [ "$confirm" = "yes" ]; then
        echo ""
        echo -e "${YELLOW}Cleaning up...${NC}"
        
        cd "$SCRIPT_DIR/.."
        
        # Use 'docker compose' if available, otherwise fall back to 'docker-compose'
        if docker compose version > /dev/null 2>&1; then
            DOCKER_COMPOSE="docker compose"
        else
            DOCKER_COMPOSE="docker-compose"
        fi
        
        # Stop and remove everything
        $DOCKER_COMPOSE down -v
        
        # Remove TensorBoard logs
        if [ -d "runs" ]; then
            echo "Removing TensorBoard logs..."
            rm -rf runs/*
        fi
        
        # Optionally remove results
        echo ""
        read -p "Also remove experiment results in results/ directory? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ -d "results" ]; then
                echo "Removing experiment results..."
                rm -rf results/*
            fi
        fi
        
        echo -e "${GREEN}✅ Cleanup completed!${NC}"
    else
        echo "Cleanup cancelled."
    fi
}

# Function to show help
show_help() {
    echo -e "${GREEN}Kaslite Monitoring Stack Help${NC}"
    echo ""
    echo "📚 Documentation:"
    echo "   • Main README: README.md"
    echo "   • Monitoring docs: monitoring/README.md"
    echo "   • User guide: docs/phase3_user_guide.md"
    echo ""
    echo "🔧 Manual Commands:"
    echo "   • Start: ./scripts/start_monitoring.sh"
    echo "   • Stop:  ./scripts/stop_monitoring.sh"
    echo "   • Status: ./scripts/status_monitoring.sh"
    echo ""
    echo "🐳 Docker Commands:"
    echo "   • View containers: docker-compose ps"
    echo "   • View logs: docker-compose logs [service-name]"
    echo "   • Restart service: docker-compose restart [service-name]"
    echo ""
    echo "📊 What each service does:"
    echo "   • Prometheus: Collects and stores metrics from your experiments"
    echo "   • Grafana: Creates dashboards and visualizations"
    echo "   • Alertmanager: Sends notifications when things go wrong"
    echo "   • TensorBoard: Shows ML training curves and model graphs"
    echo "   • Kaslite App: Your main application with metrics endpoint"
    echo ""
    echo "🚨 Troubleshooting:"
    echo "   • Port conflicts: Use 'netstat -tuln | grep :PORT' to check"
    echo "   • Docker issues: Check 'docker ps' and 'docker logs CONTAINER'"
    echo "   • Permissions: Ensure scripts are executable (chmod +x scripts/*.sh)"
}

# Main menu loop
while true; do
    show_menu
    read -r choice
    
    case $choice in
        1)
            echo ""
            run_script "start_monitoring.sh"
            ;;
        2)
            echo ""
            run_script "stop_monitoring.sh"
            ;;
        3)
            echo ""
            run_script "status_monitoring.sh"
            ;;
        4)
            echo ""
            echo -e "${YELLOW}Restarting monitoring stack...${NC}"
            run_script "stop_monitoring.sh"
            echo ""
            echo -e "${YELLOW}Starting monitoring stack...${NC}"
            run_script "start_monitoring.sh"
            ;;
        5)
            echo ""
            show_urls
            ;;
        6)
            echo ""
            run_test_experiment
            ;;
        7)
            echo ""
            show_logs
            ;;
        8)
            echo ""
            cleanup_data
            ;;
        9)
            echo ""
            show_help
            ;;
        q|Q)
            echo ""
            echo -e "${GREEN}👋 Goodbye!${NC}"
            echo ""
            exit 0
            ;;
        *)
            echo ""
            echo -e "${RED}Invalid choice. Please enter 1-9 or 'q'.${NC}"
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}Press Enter to continue...${NC}"
    read -r
done

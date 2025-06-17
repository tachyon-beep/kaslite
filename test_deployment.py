#!/usr/bin/env python3
"""
Test script for Model Registry & Deployment functionality.

This script validates the model registry, inference server, and deployment features.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import requests


def test_model_registry():
    """Test model registry CLI functionality."""
    print("🧪 Testing Model Registry...")
    
    try:
        # Test listing models
        result = subprocess.run([
            sys.executable, "-m", "morphogenetic_engine.cli.model_registry_cli", "list"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Model registry CLI is working")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print("⚠️  Model registry CLI failed or no models found")
            print(f"   Error: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"❌ Model registry test failed: {e}")


def test_inference_server():
    """Test inference server endpoints."""
    print("\n🧪 Testing Inference Server...")
    
    base_url = "http://localhost:8080"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Model loaded: {health_data.get('model_loaded')}")
            print(f"   Model version: {health_data.get('model_version')}")
        else:
            print(f"⚠️  Health endpoint returned {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint failed: {e}")
        return False
    
    # Test models endpoint
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print("✅ Models endpoint working")
            print(f"   Current version: {models_data.get('current_version')}")
            print(f"   Available versions: {models_data.get('available_versions')}")
        else:
            print(f"⚠️  Models endpoint returned {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Models endpoint failed: {e}")
    
    # Test metrics endpoint
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            print("✅ Metrics endpoint working")
            print(f"   Metrics data length: {len(response.text)} characters")
        else:
            print(f"⚠️  Metrics endpoint returned {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Metrics endpoint failed: {e}")
    
    # Test prediction endpoint (if model is loaded)
    try:
        # Simple test data
        test_data = {
            "data": [[0.5, 0.3, 0.1]]
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            pred_data = response.json()
            print("✅ Prediction endpoint working")
            print(f"   Predictions: {pred_data.get('predictions')}")
            print(f"   Model version: {pred_data.get('model_version')}")
            print(f"   Inference time: {pred_data.get('inference_time_ms')}ms")
        elif response.status_code == 503:
            print("⚠️  No model available for prediction")
        else:
            print(f"⚠️  Prediction endpoint returned {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction endpoint failed: {e}")
    
    return True


def test_docker_deployment():
    """Test Docker deployment status."""
    print("\n🧪 Testing Docker Deployment...")
    
    try:
        # Check if deployment containers are running
        result = subprocess.run([
            "docker", "compose", "-f", "docker-compose.deploy.yml", "ps", "--services", "--filter", "status=running"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            running_services = result.stdout.strip().split('\n')
            running_services = [s for s in running_services if s]  # Remove empty strings
            
            print(f"✅ Docker deployment status:")
            print(f"   Running services: {len(running_services)}")
            for service in running_services:
                print(f"   - {service}")
                
            # Check key services
            expected_services = ['inference-server', 'prometheus', 'grafana']
            for service in expected_services:
                if service in running_services:
                    print(f"   ✅ {service} is running")
                else:
                    print(f"   ⚠️  {service} is not running")
                    
        else:
            print("⚠️  Could not check Docker deployment status")
            print(f"   Error: {result.stderr.strip()}")
            
    except Exception as e:
        print(f"❌ Docker deployment test failed: {e}")


def test_monitoring_integration():
    """Test monitoring system integration."""
    print("\n🧪 Testing Monitoring Integration...")
    
    # Test Prometheus
    try:
        response = requests.get("http://localhost:9090/api/v1/targets", timeout=10)
        if response.status_code == 200:
            targets_data = response.json()
            print("✅ Prometheus is accessible")
            
            # Check if inference server is being scraped
            targets = targets_data.get('data', {}).get('activeTargets', [])
            inference_targets = [t for t in targets if 'inference' in t.get('job', '')]
            
            if inference_targets:
                print(f"   ✅ Inference server is being monitored ({len(inference_targets)} targets)")
            else:
                print("   ⚠️  Inference server not found in Prometheus targets")
                
        else:
            print(f"⚠️  Prometheus returned {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prometheus test failed: {e}")
    
    # Test Grafana
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ Grafana is accessible")
        else:
            print(f"⚠️  Grafana returned {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Grafana test failed: {e}")


def run_quick_experiment():
    """Run a quick experiment to generate model registry data."""
    print("\n🧪 Running Quick Experiment...")
    
    try:
        project_root = Path(__file__).parent
        
        result = subprocess.run([
            sys.executable, "scripts/run_morphogenetic_experiment.py",
            "--problem", "spirals",
            "--device", "cpu",
            "--max-epochs", "5",  # Quick run
            "--patience", "3"
        ], cwd=project_root, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Quick experiment completed successfully")
            
            # Check if model was registered
            if "Model registered" in result.stdout:
                print("   ✅ Model was automatically registered")
            else:
                print("   ⚠️  Model registration not detected (may be due to low accuracy)")
                
        else:
            print("⚠️  Quick experiment failed")
            print(f"   Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠️  Quick experiment timed out")
    except Exception as e:
        print(f"❌ Quick experiment failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Model Registry & Deployment Test Suite")
    print("=" * 50)
    
    # Run tests
    test_model_registry()
    test_inference_server()
    test_docker_deployment()
    test_monitoring_integration()
    
    # Ask user if they want to run a quick experiment
    try:
        response = input("\n🤔 Run a quick experiment to test model registration? (y/N): ")
        if response.lower() in ['y', 'yes']:
            run_quick_experiment()
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
    
    print("\n✨ Test suite completed!")
    print("\n📚 For more information, see docs/MODEL_REGISTRY_DEPLOYMENT.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Demo script to showcase Phase 2: Experiment Tracking & Artifacts integration.

This script demonstrates the full workflow with MLflow and DVC integration.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"⚠️  Command failed with return code {result.returncode}")
        return False
    else:
        print("✅ Command completed successfully")
        return True

def main():
    """Demonstrate the Phase 2 integration."""
    print("🚀 Phase 2: Experiment Tracking & Artifacts Demo")
    print("This demo showcases MLflow and DVC integration for reproducible ML experiments")
    
    # Check if we're in the right directory
    if not Path("dvc.yaml").exists():
        print("❌ Error: dvc.yaml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # 1. Generate data using DVC
    if not run_command("dvc repro generate_data", "Generate synthetic dataset using DVC"):
        return
    
    # 2. Run a quick experiment with MLflow tracking
    cmd = """python scripts/run_morphogenetic_experiment.py \
        --problem_type spirals \
        --n_samples 300 \
        --warm_up_epochs 3 \
        --adaptation_epochs 3 \
        --hidden_dim 32 \
        --device cpu"""
    
    if not run_command(cmd, "Run experiment with MLflow tracking"):
        return
    
    # 3. Show MLflow experiments
    if not run_command("ls -la mlruns/", "Check MLflow experiment storage"):
        return
    
    # 4. Show results and metrics
    if not run_command("ls -la results/", "Check experiment results"):
        return
    
    # 5. Show TensorBoard logs
    if not run_command("ls -la runs/", "Check TensorBoard logs"):
        return
    
    print(f"\n{'='*60}")
    print("🎉 Phase 2 Demo Completed Successfully!")
    print(f"{'='*60}")
    
    print("\n📊 View Your Results:")
    print("1. MLflow UI:      mlflow ui")
    print("   Access at:      http://localhost:5000")
    print("\n2. TensorBoard:    tensorboard --logdir runs/")
    print("   Access at:      http://localhost:6006")
    
    print("\n🔄 Reproducible Workflow:")
    print("• Data versioning: DVC tracks synthetic datasets")
    print("• Experiment tracking: MLflow logs parameters, metrics, artifacts")
    print("• Full pipeline: dvc repro reproduces entire workflow")
    print("• Model versioning: Models saved with MLflow")
    
    print("\n📁 Directory Structure:")
    print("├── data/          # DVC-tracked datasets")
    print("├── mlruns/        # MLflow experiment storage") 
    print("├── runs/          # TensorBoard logs")
    print("├── results/       # Experiment logs & metrics")
    print("└── dvc.yaml       # DVC pipeline definition")

if __name__ == "__main__":
    main()

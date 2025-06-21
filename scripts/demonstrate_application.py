#!/usr/bin/env python3
"""
Demonstrate full morphogenetic engine application: runs a real experiment with parameters chosen
to reliably trigger multiple seeds and adaptation, using the actual user-facing workflow.

This script is intended for demos and one-touch functionality demonstrations.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Difficult Demonstration: CIFAR-10 or Complex Spheres"""
    project_root = Path(__file__).parent.parent.resolve()
    script_path = project_root / "scripts" / "run_morphogenetic_experiment.py"
    if not script_path.exists():
        print(f"Could not find main experiment script at {script_path}", file=sys.stderr)
        sys.exit(1)

    # Uncomment one of the following blocks to run the desired demo

    # --- CIFAR-10 (very hard) ---
    # cmd = [
    #     sys.executable, str(script_path),
    #     "--problem_type", "cifar10",
    #     "--hidden_dim", "4",
    #     "--num_layers", "2",
    #     "--seeds_per_layer", "8",
    #     "--warm_up_epochs", "30",
    #     "--adaptation_epochs", "500",
    #     "--batch_size", "256",
    #     "--blend_steps", "5",
    #     "--progress_thresh", "0.01",
    #     "--acc_threshold", "0.95",
    #     "--lr", "0.0001",
    #     "--device", "cpu",
    #     "--seed", "12345",
    # ]

    # --- Complex Spheres (synthetic, very hard) ---
    cmd = [
        sys.executable, str(script_path),
        "--problem_type", "spheres",
        "--sphere_count", "15",
        "--sphere_size", "1000",
        "--sphere_radii", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "--sphere_noise", "0.2",
        "--input_dim", "6",
        "--num_classes", "5",
        "--hidden_dim", "8",
        "--num_layers", "5",
        "--seeds_per_layer", "5",
        "--warm_up_epochs", "300",
        "--adaptation_epochs", "1000",
        "--batch_size", "256",
        "--blend_steps", "5",
        "--progress_thresh", "0.01",
        "--acc_threshold", "0.92",
        "--lr", "0.0005",
        "--device", "cpu",
        "--seed", "54321",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Standalone data generation script for DVC pipeline.

This script generates synthetic datasets and saves them to files for DVC tracking.
"""

import argparse
import numpy as np
from pathlib import Path

from morphogenetic_engine import datasets


def main():
    """Generate dataset based on CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic datasets")
    parser.add_argument("--problem_type", required=True, 
                       choices=["spirals", "moons", "clusters", "spheres", "complex_moons"],
                       help="Type of problem to generate data for")
    parser.add_argument("--n_samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--input_dim", type=int, default=3, help="Input dimension")
    
    # Dataset-specific parameters
    parser.add_argument("--noise", type=float, default=0.25, help="Spirals noise")
    parser.add_argument("--rotations", type=int, default=4, help="Spirals rotations")
    parser.add_argument("--moon_noise", type=float, default=0.1, help="Moons noise")
    parser.add_argument("--moon_sep", type=float, default=0.5, help="Moons separation")
    parser.add_argument("--cluster_count", type=int, default=2, help="Number of clusters")
    parser.add_argument("--cluster_std", type=float, default=0.5, help="Cluster std dev")
    parser.add_argument("--cluster_sep", type=float, default=3.0, help="Cluster separation")
    parser.add_argument("--sphere_count", type=int, default=2, help="Number of spheres")
    parser.add_argument("--sphere_radii", type=float, nargs='+', default=[1.0, 2.0], help="Sphere radii")
    parser.add_argument("--sphere_noise", type=float, default=0.1, help="Sphere noise")
    
    args = parser.parse_args()
    
    # Generate data based on problem type
    if args.problem_type == "spirals":
        X, y = datasets.create_spirals(
            n_samples=args.n_samples,
            noise=args.noise,
            rotations=args.rotations,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "moons":
        X, y = datasets.create_moons(
            n_samples=args.n_samples,
            moon_noise=args.moon_noise,
            moon_sep=args.moon_sep,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "complex_moons":
        X, y = datasets.create_complex_moons(
            n_samples=args.n_samples,
            noise=args.moon_noise,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "clusters":
        X, y = datasets.create_clusters(
            cluster_count=args.cluster_count,
            cluster_size=args.n_samples // args.cluster_count,
            cluster_std=args.cluster_std,
            cluster_sep=args.cluster_sep,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "spheres":
        X, y = datasets.create_spheres(
            sphere_count=args.sphere_count,
            sphere_size=args.n_samples // args.sphere_count,
            sphere_radii=args.sphere_radii,
            sphere_noise=args.sphere_noise,
            input_dim=args.input_dim,
        )
    else:
        raise ValueError(f"Unknown problem type: {args.problem_type}")
    
    # Save the data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .npz file to include both X and y
    np.savez(output_path.with_suffix('.npz'), X=X, y=y)
    
    print(f"Generated {args.problem_type} dataset with {len(X)} samples")
    print(f"Saved to: {output_path}")
    print(f"Data shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()

"""
Dataset generation utilities for morphogenetic architecture experiments.

This module provides functions to generate various synthetic datasets
for testing and evaluating morphogenetic neural networks.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from sklearn.datasets import make_blobs, make_moons


def create_spirals(
    n_samples: int = 2000, noise: float = 0.25, rotations: int = 4, input_dim: int = 2
):
    """Generate the classic two-spirals toy dataset, optionally padded to input_dim."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    n = np.sqrt(rng.random(n_samples // 2)) * rotations * 2 * np.pi
    d1x = np.cos(n) * n + rng.random(n_samples // 2) * noise
    d1y = np.sin(n) * n + rng.random(n_samples // 2) * noise

    X = np.vstack((np.hstack((d1x, -d1x)), np.hstack((d1y, -d1y)))).T
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # Pad with independent N(0,1) features if input_dim > 2
    if input_dim > 2:
        padding = rng.standard_normal((n_samples, input_dim - 2))
        X = np.hstack((X, padding))

    return X.astype(np.float32), y.astype(np.int64)


def create_complex_moons(n_samples: int = 2000, noise: float = 0.1, input_dim: int = 2):
    """Generate complex moons dataset: two half-moons + two Gaussian clusters."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Generate two interleaved half-moons
    n_moons = n_samples // 2
    X_moons, y_moons = make_moons(n_samples=n_moons, noise=noise, random_state=42)

    # Generate two Gaussian clusters
    n_clusters = n_samples - n_moons
    n_cluster1 = n_clusters // 2
    n_cluster2 = n_clusters - n_cluster1

    # Cluster 1: centered at (2, 2)
    cluster1 = rng.multivariate_normal([2.0, 2.0], [[0.5, 0.1], [0.1, 0.5]], n_cluster1)
    y_cluster1 = np.zeros(n_cluster1)

    # Cluster 2: centered at (-2, -2)
    cluster2 = rng.multivariate_normal([-2.0, -2.0], [[0.5, -0.1], [-0.1, 0.5]], n_cluster2)
    y_cluster2 = np.ones(n_cluster2)

    # Concatenate all data
    X = np.vstack((X_moons, cluster1, cluster2))
    y = np.hstack((y_moons, y_cluster1, y_cluster2))

    # Shuffle the data
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    # Pad with independent N(0,1) features if input_dim > 2
    if input_dim > 2:
        padding = rng.standard_normal((len(X), input_dim - 2))
        X = np.hstack((X, padding))

    return X.astype(np.float32), y.astype(np.int64)


def create_moons(
    n_samples: int = 2000,
    moon_noise: float = 0.1,
    moon_sep: float = 0.5,
    input_dim: int = 2,
):
    """Generate two interleaved half-moons dataset."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Generate moons using sklearn
    X, y = make_moons(n_samples=n_samples, noise=moon_noise, random_state=42)

    # Adjust separation by scaling x-coordinates
    X[:, 0] *= moon_sep + 1.0

    # Pad with independent N(0,1) features if input_dim > 2
    if input_dim > 2:
        padding = rng.standard_normal((n_samples, input_dim - 2))
        X = np.hstack((X, padding))

    return X.astype(np.float32), y.astype(np.int64)


def create_clusters(
    cluster_count: int = 2,
    cluster_size: int = 500,
    cluster_std: float = 0.5,
    cluster_sep: float = 3.0,
    input_dim: int = 3,
):
    """Generate Gaussian clusters in n-dimensional space."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Use cluster_size as samples per cluster, but cap total based on realistic limits
    n_samples = cluster_count * cluster_size

    # Generate cluster centers
    centers = []
    for i in range(cluster_count):
        # Place centers in a circle/sphere pattern
        angle = 2 * np.pi * i / cluster_count
        if input_dim == 2:
            center = cluster_sep * np.array([np.cos(angle), np.sin(angle)])
        elif input_dim == 3:
            # Use spherical coordinates for 3D
            phi = np.pi * i / cluster_count  # elevation angle
            center = cluster_sep * np.array(
                [np.cos(angle) * np.sin(phi), np.sin(angle) * np.sin(phi), np.cos(phi)]
            )
        else:
            # For higher dimensions, randomize the remaining coordinates
            center = cluster_sep * np.concatenate(
                [[np.cos(angle), np.sin(angle)], rng.standard_normal(input_dim - 2)]
            )
        centers.append(center)

    # Generate data using make_blobs
    X, y = cast(  # pylint: disable=unbalanced-tuple-unpacking
        tuple[np.ndarray, np.ndarray],
        make_blobs(
            n_samples=n_samples,
            centers=np.array(centers),
            cluster_std=cluster_std,
            n_features=input_dim,
            random_state=42,
            return_centers=False,
        ),
    )

    # Convert to binary classification by grouping clusters
    # Odd clusters become class 0, even clusters become class 1
    y_binary = y % 2

    return X.astype(np.float32), y_binary.astype(np.int64)


def create_spheres(
    sphere_count: int = 2,
    sphere_size: int = 500,
    sphere_radii: str = "1,2",
    sphere_noise: float = 0.05,
    input_dim: int = 3,
):
    """Generate concentric spherical shells in n-dimensional space."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Parse radii string
    radii = [float(r.strip()) for r in sphere_radii.split(",")]
    if len(radii) != sphere_count:
        raise ValueError(f"Number of radii ({len(radii)}) must match sphere_count ({sphere_count})")

    X_list = []
    y_list = []

    for i, radius in enumerate(radii):
        # Generate points on unit sphere surface
        if input_dim == 2:
            # For 2D, generate points on circle
            angles = rng.uniform(0, 2 * np.pi, sphere_size)
            x = np.cos(angles)
            y_coord = np.sin(angles)
            points = np.column_stack([x, y_coord])
        elif input_dim == 3:
            # For 3D, use spherical coordinates
            u = rng.uniform(0, 1, sphere_size)
            v = rng.uniform(0, 1, sphere_size)
            theta = 2 * np.pi * u  # azimuthal angle
            phi = np.arccos(2 * v - 1)  # polar angle

            x = np.sin(phi) * np.cos(theta)
            y_coord = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points = np.column_stack([x, y_coord, z])
        else:
            # For higher dimensions, use Gaussian method and normalize
            points = rng.standard_normal((sphere_size, input_dim))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        # Scale to desired radius and add noise
        points = points * radius
        if sphere_noise > 0:
            noise = rng.normal(0, sphere_noise, points.shape)
            points += noise

        X_list.append(points)
        # Convert to binary classification by grouping spheres
        # Odd spheres become class 0, even spheres become class 1
        y_list.append(np.full(sphere_size, i % 2))

    # Concatenate all spheres
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle the data
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    return X.astype(np.float32), y.astype(np.int64)

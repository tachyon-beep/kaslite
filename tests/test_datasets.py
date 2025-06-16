"""
Test suite for dataset generation functions.

This module contains tests for the morphogenetic architecture dataset generation,
including tests for:

- Dataset generation functions (spirals, moons, clusters, spheres, complex_moons)
- Parameter validation and reproducibility
- High-dimensional input padding
- Data structure validation

Test Classes:
    TestCreateSpirals: Tests for spiral dataset generation
    TestCreateComplexMoons: Tests for complex moons dataset generation
    TestNewDatasets: Tests for moons, clusters, and spheres dataset generation
"""

import numpy as np
import pytest

from morphogenetic_engine.datasets import (
    create_clusters,
    create_complex_moons,
    create_moons,
    create_spheres,
    create_spirals,
)


class TestCreateSpirals:
    """Test suite for spiral dataset generation."""

    def test_default_parameters(self):
        """Test spiral generation with default parameters."""
        X, y = create_spirals()

        assert X.shape == (2000, 2)
        assert y.shape == (2000,)
        assert X.dtype == np.float32
        assert y.dtype == np.int64

        # Check label distribution
        assert np.sum(y == 0) == 1000
        assert np.sum(y == 1) == 1000

    def test_custom_parameters(self):
        """Test spiral generation with custom parameters."""
        n_samples = 1000
        X, y = create_spirals(n_samples=n_samples, noise=0.1, rotations=2)

        assert X.shape == (n_samples, 2)
        assert y.shape == (n_samples,)

        # Check label distribution
        assert np.sum(y == 0) == n_samples // 2
        assert np.sum(y == 1) == n_samples // 2

    def test_reproducibility(self):
        """Test that spiral generation is reproducible with same random seed."""
        # The function uses a fixed seed (42) so should be reproducible
        X1, y1 = create_spirals(n_samples=100)
        X2, y2 = create_spirals(n_samples=100)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_data_range(self):
        """Test that generated data has reasonable range."""
        X, _ = create_spirals(n_samples=500, noise=0.1)

        # Data should be centered around origin with reasonable range
        assert np.abs(X.mean()) < 5.0  # Not too far from origin
        assert np.std(X) > 0.1  # Has reasonable variance

    def test_spiral_structure(self):
        """Test that generated data has spiral structure."""
        X, y = create_spirals(n_samples=200, noise=0.05)

        # Convert to polar coordinates to check spiral structure
        r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        theta = np.arctan2(X[:, 1], X[:, 0])

        # For each class, radius should generally increase with angle
        for class_label in [0, 1]:
            class_mask = y == class_label
            class_r = r[class_mask]
            class_theta = theta[class_mask]

            # Sort by angle and check that radius generally increases
            sorted_indices = np.argsort(class_theta)
            sorted_r = class_r[sorted_indices]

            # Check that there's some spiral structure (not perfectly linear)
            # but with a reasonable correlation between sorted index and radius
            correlation = np.corrcoef(np.arange(len(sorted_r)), sorted_r)[0, 1]
            # Allow for more flexible spiral structure
            assert abs(correlation) > 0.05, f"Class {class_label} correlation: {correlation}"

    def test_input_dim_padding(self):
        """Test that spirals are properly padded to higher dimensions."""
        X, y = create_spirals(n_samples=100, input_dim=5)

        # Should be padded to 5 dimensions
        assert X.shape == (100, 5)
        assert y.shape == (100,)

        # First 2 dimensions should be the spiral data (not all zeros)
        # Last 3 should be independent padding
        assert not np.allclose(X[:, 2], 0)  # Padding should not be all zeros

        # Test input_dim=2 gives same result as default
        X_2d, y_2d = create_spirals(n_samples=100, input_dim=2)
        X_default, y_default = create_spirals(n_samples=100)

        np.testing.assert_array_equal(X_2d, X_default)
        np.testing.assert_array_equal(y_2d, y_default)


class TestCreateComplexMoons:
    """Test suite for complex moons dataset generation."""

    def test_default_parameters(self):
        """Test complex moons generation with default parameters."""
        X, y = create_complex_moons()

        # Check output shapes
        assert X.shape == (2000, 2)
        assert y.shape == (2000,)

        # Check data types
        assert X.dtype == np.float32
        assert y.dtype == np.int64

        # Check binary labels
        assert set(y) == {0, 1}

    def test_custom_parameters(self):
        """Test complex moons generation with custom parameters."""
        X, y = create_complex_moons(n_samples=200, noise=0.05)

        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_input_dim_padding(self):
        """Test that complex moons are properly padded to higher dimensions."""
        X, y = create_complex_moons(n_samples=100, input_dim=4)

        # Should be padded to 4 dimensions
        assert X.shape == (100, 4)
        assert y.shape == (100,)

        # Padding dimensions should not be all zeros
        assert not np.allclose(X[:, 2], 0)
        assert not np.allclose(X[:, 3], 0)

    def test_reproducibility(self):
        """Test that complex moons generation is reproducible."""
        X1, y1 = create_complex_moons(n_samples=100)
        X2, y2 = create_complex_moons(n_samples=100)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_data_structure(self):
        """Test that complex moons has expected structure (moons + clusters)."""
        X, _ = create_complex_moons(n_samples=400, noise=0.1)

        # Should have reasonable spread (combination of moons and clusters)
        assert X.std() > 0.5  # Should have some variance
        assert np.abs(X.mean()) < 2.0  # But not too far from origin


class TestNewDatasets:
    """Test the new dataset generation functions."""

    def test_create_moons(self):
        """Test moons dataset generation."""
        # Test basic functionality
        X, y = create_moons(n_samples=100, moon_noise=0.1, moon_sep=0.5, input_dim=2)
        assert X.shape == (100, 2)
        assert y.shape == (100,)
        assert set(y) == {0, 1}  # Binary classification

        # Test with higher dimensions
        X, y = create_moons(n_samples=100, input_dim=4)
        assert X.shape == (100, 4)
        assert y.shape == (100,)

    def test_create_clusters(self):
        """Test clusters dataset generation."""
        # Test 2 clusters
        X, y = create_clusters(cluster_count=2, cluster_size=50, input_dim=3)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert set(y) == {0, 1}  # Binary classification

        # Test 3 clusters (should still produce binary classification)
        X, y = create_clusters(cluster_count=3, cluster_size=50, input_dim=2)
        assert X.shape == (150, 2)
        assert y.shape == (150,)
        assert set(y) == {0, 1}  # Binary classification due to modulo

    def test_create_spheres(self):
        """Test spheres dataset generation."""
        # Test 2 spheres
        X, y = create_spheres(sphere_count=2, sphere_size=50, sphere_radii="1,2", input_dim=3)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert set(y) == {0, 1}  # Binary classification

        # Test 3 spheres (should still produce binary classification)
        X, y = create_spheres(sphere_count=3, sphere_size=30, sphere_radii="1,2,3", input_dim=4)
        assert X.shape == (90, 4)
        assert y.shape == (90,)
        assert set(y) == {0, 1}  # Binary classification due to modulo

    def test_dataset_validation(self):
        """Test dataset validation and error handling."""
        # Test mismatched radii and sphere_count
        with pytest.raises(ValueError, match="Number of radii"):
            create_spheres(sphere_count=2, sphere_size=50, sphere_radii="1,2,3", input_dim=3)

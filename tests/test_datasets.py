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

# pylint: disable=redefined-outer-name
# Pytest fixtures are expected to be redefined in test method parameters

import time
from typing import Callable

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from morphogenetic_engine.datasets import create_clusters, create_complex_moons, create_moons, create_spheres, create_spirals

# Constants to replace magic numbers
DEFAULT_SAMPLES = 2000
SPIRAL_CLASSES = 2
MAX_DATA_RANGE = 5.0
MIN_VARIANCE_THRESHOLD = 0.1
MIN_CORRELATION_THRESHOLD = 0.05
PERFORMANCE_TIMEOUT_SECONDS = 5.0
LARGE_DATASET_SIZE = 50000


@pytest.fixture
def dataset_shape_validator() -> Callable:
    """Helper function to validate dataset shapes and types."""

    def _validate(X: np.ndarray, y: np.ndarray, expected_samples: int, expected_dims: int) -> None:
        assert X.shape == (
            expected_samples,
            expected_dims,
        ), f"Expected X shape ({expected_samples}, {expected_dims}), got {X.shape}"
        assert y.shape == (expected_samples,), f"Expected y shape ({expected_samples},), got {y.shape}"
        assert X.dtype == np.float32, f"Expected X dtype float32, got {X.dtype}"
        assert y.dtype == np.int64, f"Expected y dtype int64, got {y.dtype}"

    return _validate


@pytest.fixture
def binary_classification_validator() -> Callable:
    """Helper function to validate binary classification labels."""

    def _validate(y: np.ndarray) -> None:
        unique_labels = set(y)
        assert unique_labels == {0, 1}, f"Expected binary labels {{0, 1}}, got {unique_labels}"

    return _validate


@pytest.fixture
def classification_validator() -> Callable:
    """Helper function to validate classification labels (binary or single-class)."""

    def _validate(y: np.ndarray, allow_single_class: bool = False) -> None:
        unique_labels = set(y)
        if allow_single_class:
            assert unique_labels.issubset({0, 1}), f"Expected labels from {{0, 1}}, got {unique_labels}"
            assert len(unique_labels) >= 1, f"Expected at least one class, got {unique_labels}"
        else:
            assert unique_labels == {0, 1}, f"Expected binary labels {{0, 1}}, got {unique_labels}"

    return _validate


class TestCreateSpirals:
    """Test suite for spiral dataset generation."""

    def test_default_parameters(self, dataset_shape_validator, binary_classification_validator):
        """Test spiral generation with default parameters."""
        X, y = create_spirals()

        dataset_shape_validator(X, y, DEFAULT_SAMPLES, 2)
        binary_classification_validator(y)

        # Check label distribution
        assert np.sum(y == 0) == DEFAULT_SAMPLES // 2, "Expected equal distribution of class 0"
        assert np.sum(y == 1) == DEFAULT_SAMPLES // 2, "Expected equal distribution of class 1"

    def test_custom_parameters(self, dataset_shape_validator, binary_classification_validator):
        """Test spiral generation with custom parameters."""
        n_samples = 1000
        X, y = create_spirals(n_samples=n_samples, noise=0.1, rotations=2)

        dataset_shape_validator(X, y, n_samples, 2)
        binary_classification_validator(y)

        # Check label distribution
        assert np.sum(y == 0) == n_samples // 2, "Expected equal distribution of class 0"
        assert np.sum(y == 1) == n_samples // 2, "Expected equal distribution of class 1"

    def test_boundary_conditions(self, dataset_shape_validator, binary_classification_validator):
        """Test spiral generation with boundary condition parameters."""
        # Test minimum samples for binary classification
        X, y = create_spirals(n_samples=2)
        dataset_shape_validator(X, y, 2, 2)
        binary_classification_validator(y)

        # Test very small sample sizes
        X, y = create_spirals(n_samples=4)
        dataset_shape_validator(X, y, 4, 2)
        assert np.sum(y == 0) == 2, "Expected 2 samples of class 0"
        assert np.sum(y == 1) == 2, "Expected 2 samples of class 1"

        # Test minimum input dimension
        X, y = create_spirals(n_samples=100, input_dim=2)
        dataset_shape_validator(X, y, 100, 2)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid n_samples
        with pytest.raises((ValueError, TypeError)):
            create_spirals(n_samples=0)

        with pytest.raises((ValueError, TypeError)):
            create_spirals(n_samples=-1)

        # Test odd n_samples (spiral function creates n//2 per class, so 3 becomes 2 total)
        X, _ = create_spirals(n_samples=3)
        assert X.shape[0] == 2, "Spiral function with n_samples=3 creates 2 samples (1 per class)"

        # Test invalid input_dim
        with pytest.raises((ValueError, TypeError)):
            create_spirals(input_dim=0)

        with pytest.raises((ValueError, TypeError)):
            create_spirals(input_dim=1)

    def test_reproducibility(self):
        """Test that spiral generation is reproducible with same random seed."""
        # The function uses a fixed seed (42) so should be reproducible
        X1, y1 = create_spirals(n_samples=100)
        X2, y2 = create_spirals(n_samples=100)

        np.testing.assert_array_equal(X1, X2, "Generated data should be reproducible")
        np.testing.assert_array_equal(y1, y2, "Generated labels should be reproducible")

    def test_data_range(self):
        """Test that generated data has reasonable range."""
        X, _ = create_spirals(n_samples=500, noise=0.1)

        # Data should be centered around origin with reasonable range
        assert np.abs(X.mean()) < MAX_DATA_RANGE, f"Data mean {X.mean():.3f} should be within ±{MAX_DATA_RANGE}"
        assert np.std(X) > MIN_VARIANCE_THRESHOLD, f"Data std {np.std(X):.3f} should be > {MIN_VARIANCE_THRESHOLD}"

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
            assert (
                abs(correlation) > MIN_CORRELATION_THRESHOLD
            ), f"Class {class_label} correlation {correlation:.3f} should be > {MIN_CORRELATION_THRESHOLD}"

    def test_input_dim_padding(self, dataset_shape_validator, binary_classification_validator):
        """Test that spirals are properly padded to higher dimensions."""
        X, y = create_spirals(n_samples=100, input_dim=5)

        # Should be padded to 5 dimensions
        dataset_shape_validator(X, y, 100, 5)
        binary_classification_validator(y)

        # First 2 dimensions should be the spiral data (not all zeros)
        # Last 3 should be independent padding
        assert not np.allclose(X[:, 2], 0), "Padding dimensions should not be all zeros"

        # Test input_dim=2 gives same result as default
        X_2d, y_2d = create_spirals(n_samples=100, input_dim=2)
        X_default, y_default = create_spirals(n_samples=100)

        np.testing.assert_array_equal(X_2d, X_default, "2D result should match default")
        np.testing.assert_array_equal(y_2d, y_default, "2D labels should match default")


class TestCreateComplexMoons:
    """Test suite for complex moons dataset generation."""

    def test_default_parameters(self, dataset_shape_validator, binary_classification_validator):
        """Test complex moons generation with default parameters."""
        X, y = create_complex_moons()

        dataset_shape_validator(X, y, DEFAULT_SAMPLES, 2)
        binary_classification_validator(y)

    def test_custom_parameters(self, dataset_shape_validator):
        """Test complex moons generation with custom parameters."""
        X, y = create_complex_moons(n_samples=200, noise=0.05)

        dataset_shape_validator(X, y, 200, 2)

    def test_boundary_conditions(self, dataset_shape_validator, binary_classification_validator):
        """Test complex moons generation with boundary condition parameters."""
        # Test minimum samples
        X, y = create_complex_moons(n_samples=4)
        dataset_shape_validator(X, y, 4, 2)
        binary_classification_validator(y)

        # Test minimum input dimension
        X, y = create_complex_moons(n_samples=100, input_dim=2)
        dataset_shape_validator(X, y, 100, 2)

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid n_samples
        with pytest.raises((ValueError, TypeError)):
            create_complex_moons(n_samples=0)

        with pytest.raises((ValueError, TypeError)):
            create_complex_moons(n_samples=-1)

        # Test invalid input_dim
        with pytest.raises((ValueError, TypeError)):
            create_complex_moons(input_dim=0)

        with pytest.raises((ValueError, TypeError)):
            create_complex_moons(input_dim=1)

    def test_input_dim_padding(self, dataset_shape_validator):
        """Test that complex moons are properly padded to higher dimensions."""
        X, y = create_complex_moons(n_samples=100, input_dim=4)

        # Should be padded to 4 dimensions
        dataset_shape_validator(X, y, 100, 4)

        # Padding dimensions should not be all zeros
        assert not np.allclose(X[:, 2], 0), "Padding dimension 2 should not be all zeros"
        assert not np.allclose(X[:, 3], 0), "Padding dimension 3 should not be all zeros"

    def test_reproducibility(self):
        """Test that complex moons generation is reproducible."""
        X1, y1 = create_complex_moons(n_samples=100)
        X2, y2 = create_complex_moons(n_samples=100)

        np.testing.assert_array_equal(X1, X2, "Generated data should be reproducible")
        np.testing.assert_array_equal(y1, y2, "Generated labels should be reproducible")

    def test_data_structure(self):
        """Test that complex moons has expected structure (moons + clusters)."""
        X, _ = create_complex_moons(n_samples=400, noise=0.1)

        # Should have reasonable spread (combination of moons and clusters)
        assert X.std() > 0.5, f"Data std {X.std():.3f} should be > 0.5 for reasonable variance"
        assert np.abs(X.mean()) < 2.0, f"Data mean {np.abs(X.mean()):.3f} should be < 2.0, not too far from origin"


class TestNewDatasets:
    """Test the new dataset generation functions."""

    def test_create_moons(self, dataset_shape_validator, binary_classification_validator):
        """Test moons dataset generation."""
        # Test basic functionality
        X, y = create_moons(n_samples=100, moon_noise=0.1, moon_sep=0.5, input_dim=2)
        dataset_shape_validator(X, y, 100, 2)
        binary_classification_validator(y)

        # Test with higher dimensions
        X, y = create_moons(n_samples=100, input_dim=4)
        dataset_shape_validator(X, y, 100, 4)

    def test_create_moons_boundary_conditions(self, dataset_shape_validator, binary_classification_validator):
        """Test moons dataset generation with boundary conditions."""
        # Test minimum samples
        X, y = create_moons(n_samples=2, input_dim=2)
        dataset_shape_validator(X, y, 2, 2)
        binary_classification_validator(y)

    def test_create_moons_parameter_validation(self):
        """Test moons parameter validation."""
        with pytest.raises((ValueError, TypeError)):
            create_moons(n_samples=0)

        with pytest.raises((ValueError, TypeError)):
            create_moons(n_samples=-1)

        with pytest.raises((ValueError, TypeError)):
            create_moons(input_dim=1)

    def test_create_clusters(self, dataset_shape_validator, binary_classification_validator):
        """Test clusters dataset generation."""
        # Test 2 clusters
        X, y = create_clusters(cluster_count=2, cluster_size=50, input_dim=3)
        dataset_shape_validator(X, y, 100, 3)
        binary_classification_validator(y)

        # Test 3 clusters (should still produce binary classification)
        X, y = create_clusters(cluster_count=3, cluster_size=50, input_dim=2)
        dataset_shape_validator(X, y, 150, 2)
        binary_classification_validator(y)  # Binary classification due to modulo

    def test_create_clusters_boundary_conditions(self, dataset_shape_validator):
        """Test clusters dataset generation with boundary conditions."""
        # Test minimum cluster configuration
        X, y = create_clusters(cluster_count=1, cluster_size=2, input_dim=2)
        dataset_shape_validator(X, y, 2, 2)

        # Test very small clusters
        X, y = create_clusters(cluster_count=2, cluster_size=1, input_dim=2)
        dataset_shape_validator(X, y, 2, 2)

    def test_create_clusters_parameter_validation(self):
        """Test clusters parameter validation."""
        with pytest.raises((ValueError, TypeError)):
            create_clusters(cluster_count=0)

        with pytest.raises((ValueError, TypeError)):
            create_clusters(cluster_size=0)

        with pytest.raises((ValueError, TypeError)):
            create_clusters(input_dim=1)

    def test_create_spheres(self, dataset_shape_validator, binary_classification_validator):
        """Test spheres dataset generation."""
        # Test 2 spheres
        X, y = create_spheres(sphere_count=2, sphere_size=50, sphere_radii="1,2", input_dim=3)
        dataset_shape_validator(X, y, 100, 3)
        binary_classification_validator(y)

        # Test 3 spheres (should still produce binary classification)
        X, y = create_spheres(sphere_count=3, sphere_size=30, sphere_radii="1,2,3", input_dim=4)
        dataset_shape_validator(X, y, 90, 4)
        binary_classification_validator(y)  # Binary classification due to modulo

    def test_create_spheres_boundary_conditions(self, dataset_shape_validator):
        """Test spheres dataset generation with boundary conditions."""
        # Test minimum sphere configuration
        X, y = create_spheres(sphere_count=1, sphere_size=2, sphere_radii="1", input_dim=3)
        dataset_shape_validator(X, y, 2, 3)

        # Test minimum dimensions
        X, y = create_spheres(sphere_count=2, sphere_size=5, sphere_radii="1,2", input_dim=2)
        dataset_shape_validator(X, y, 10, 2)

    def test_create_spheres_parameter_validation(self):
        """Test spheres parameter validation."""
        with pytest.raises((ValueError, TypeError)):
            create_spheres(sphere_count=0)

        with pytest.raises((ValueError, TypeError)):
            create_spheres(sphere_size=0)

        with pytest.raises((ValueError, TypeError)):
            create_spheres(input_dim=1)

    def test_dataset_validation(self):
        """Test dataset validation and error handling."""
        # Test mismatched radii and sphere_count
        with pytest.raises(ValueError, match="Number of radii"):
            create_spheres(sphere_count=2, sphere_size=50, sphere_radii="1,2,3", input_dim=3)

        # Test invalid radii format
        with pytest.raises(ValueError):
            create_spheres(sphere_count=2, sphere_size=50, sphere_radii="invalid,format", input_dim=3)

        # Test empty radii
        with pytest.raises(ValueError):
            create_spheres(sphere_count=1, sphere_size=50, sphere_radii="", input_dim=3)


class TestDatasetPerformance:
    """Test performance and scalability of dataset generation functions."""

    def test_large_spiral_generation(self):
        """Test that large spiral datasets can be generated efficiently."""
        start_time = time.time()
        X, _ = create_spirals(n_samples=LARGE_DATASET_SIZE)
        duration = time.time() - start_time

        assert (
            duration < PERFORMANCE_TIMEOUT_SECONDS
        ), f"Large dataset generation took {duration:.2f}s, should be < {PERFORMANCE_TIMEOUT_SECONDS}s"
        assert X.shape == (LARGE_DATASET_SIZE, 2), "Large dataset should have correct shape"

    def test_large_complex_moons_generation(self):
        """Test that large complex moons datasets can be generated efficiently."""
        start_time = time.time()
        X, _ = create_complex_moons(n_samples=LARGE_DATASET_SIZE)
        duration = time.time() - start_time

        assert (
            duration < PERFORMANCE_TIMEOUT_SECONDS
        ), f"Large dataset generation took {duration:.2f}s, should be < {PERFORMANCE_TIMEOUT_SECONDS}s"
        assert X.shape == (LARGE_DATASET_SIZE, 2), "Large dataset should have correct shape"


class TestDataQuality:
    """Test data quality metrics for all dataset generation functions."""

    def test_spiral_data_quality(self):
        """Test statistical properties of spiral datasets."""
        X, y = create_spirals(n_samples=1000, noise=0.1)

        # Test class balance
        class_counts = np.bincount(y)
        assert np.all(class_counts >= 450), f"Class imbalance too severe: {class_counts}"

        # Test data distribution
        assert not np.any(np.isnan(X)), "Dataset should not contain NaN values"
        assert not np.any(np.isinf(X)), "Dataset should not contain infinite values"
        assert np.all(np.isfinite(X)), "All values should be finite"

        # Test variance
        assert np.var(X) > 0.01, f"Dataset variance {np.var(X):.6f} too low"

    def test_complex_moons_data_quality(self):
        """Test statistical properties of complex moons datasets."""
        X, y = create_complex_moons(n_samples=1000, noise=0.1)

        # Test class balance (may not be perfectly balanced due to structure)
        class_counts = np.bincount(y)
        assert len(class_counts) == 2, "Should have exactly 2 classes"
        assert np.min(class_counts) >= 200, f"Minimum class count {np.min(class_counts)} too low"

        # Test data distribution
        assert not np.any(np.isnan(X)), "Dataset should not contain NaN values"
        assert not np.any(np.isinf(X)), "Dataset should not contain infinite values"
        assert np.all(np.isfinite(X)), "All values should be finite"

    def test_clusters_data_quality(self):
        """Test statistical properties of clusters datasets."""
        X, y = create_clusters(cluster_count=3, cluster_size=100, input_dim=3)

        # Test class balance (binary due to modulo)
        class_counts = np.bincount(y)
        assert len(class_counts) == 2, "Should have exactly 2 classes after modulo"

        # Test data distribution
        assert not np.any(np.isnan(X)), "Dataset should not contain NaN values"
        assert not np.any(np.isinf(X)), "Dataset should not contain infinite values"
        assert np.all(np.isfinite(X)), "All values should be finite"

    def test_spheres_data_quality(self):
        """Test statistical properties of spheres datasets."""
        X, y = create_spheres(sphere_count=2, sphere_size=100, sphere_radii="1,2", input_dim=3)

        # Test class balance
        class_counts = np.bincount(y)
        assert len(class_counts) == 2, "Should have exactly 2 classes"
        assert np.all(class_counts == 100), f"Each class should have 100 samples: {class_counts}"

        # Test data distribution
        assert not np.any(np.isnan(X)), "Dataset should not contain NaN values"
        assert not np.any(np.isinf(X)), "Dataset should not contain infinite values"
        assert np.all(np.isfinite(X)), "All values should be finite"

        # Test spherical structure (points should be roughly on sphere surfaces)
        for sphere_label in [0, 1]:
            sphere_mask = y == sphere_label
            sphere_points = X[sphere_mask]
            distances = np.linalg.norm(sphere_points, axis=1)
            expected_radius = sphere_label + 1  # radii are "1,2"

            # Allow some variance due to noise
            assert (
                np.abs(np.mean(distances) - expected_radius) < 0.5
            ), f"Sphere {sphere_label} mean distance {np.mean(distances):.3f} should be near {expected_radius}"


class TestDatasetCompatibility:
    """Test that datasets are compatible with typical ML workflows."""

    @pytest.mark.parametrize(
        "dataset_func,params",
        [
            (create_spirals, {"n_samples": 100, "input_dim": 2}),
            (create_complex_moons, {"n_samples": 100, "input_dim": 2}),
            (create_moons, {"n_samples": 100, "input_dim": 2}),
            (create_clusters, {"cluster_count": 2, "cluster_size": 50, "input_dim": 2}),
            (
                create_spheres,
                {"sphere_count": 2, "sphere_size": 50, "sphere_radii": "1,2", "input_dim": 3},
            ),
            # Add single-class edge cases
            (create_clusters, {"cluster_count": 1, "cluster_size": 50, "input_dim": 2}),
            (
                create_spheres,
                {"sphere_count": 1, "sphere_size": 50, "sphere_radii": "1.5", "input_dim": 3},
            ),
        ],
    )
    def test_dataset_sklearn_compatibility(self, dataset_func, params):
        """Test that datasets work with sklearn estimators."""
        X, y = dataset_func(**params)

        # Check if we have single-class data
        unique_classes = set(y)
        if len(unique_classes) == 1:
            # Single-class scenario - just verify structure
            assert len(y) > 0, "Should have at least one sample"
            assert unique_classes.issubset({0, 1}), "Single class should be 0 or 1"
            # Skip ML model testing for single-class data
            return

        # Multi-class scenario - test with ML model
        # Should be able to split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Should be able to fit a simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Should be able to make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test), "Predictions should match test set size"
        assert set(predictions).issubset({0, 1}), "Predictions should be binary"


class TestParameterizedDatasets:
    """Parameterized tests to reduce code duplication across dataset functions."""

    @pytest.mark.parametrize(
        "dataset_func,default_params,expected_shape",
        [
            (create_spirals, {"n_samples": 100}, (100, 2)),
            (create_complex_moons, {"n_samples": 100}, (100, 2)),
            (create_moons, {"n_samples": 100}, (100, 2)),
            (create_clusters, {"cluster_count": 2, "cluster_size": 50}, (100, 3)),
            (
                create_spheres,
                {"sphere_count": 2, "sphere_size": 50, "sphere_radii": "1,2"},
                (100, 3),
            ),
        ],
    )
    def test_basic_functionality(
        self,
        dataset_func,
        default_params,
        expected_shape,
        dataset_shape_validator,
        binary_classification_validator,
    ):
        """Test basic functionality across all dataset functions."""
        X, y = dataset_func(**default_params)
        dataset_shape_validator(X, y, expected_shape[0], expected_shape[1])
        binary_classification_validator(y)

    @pytest.mark.parametrize(
        "dataset_func,params",
        [
            (create_spirals, {"n_samples": 50, "input_dim": 4}),
            (create_complex_moons, {"n_samples": 50, "input_dim": 5}),
            (create_moons, {"n_samples": 50, "input_dim": 3}),
            (create_clusters, {"cluster_count": 2, "cluster_size": 25, "input_dim": 4}),
            (
                create_spheres,
                {"sphere_count": 2, "sphere_size": 25, "sphere_radii": "1,2", "input_dim": 4},
            ),
        ],
    )
    def test_high_dimensional_padding(self, dataset_func, params, dataset_shape_validator):
        """Test high-dimensional padding across all dataset functions."""
        X, y = dataset_func(**params)
        expected_samples = params.get("n_samples", params.get("cluster_count", 2) * params.get("cluster_size", 25))
        dataset_shape_validator(X, y, expected_samples, params["input_dim"])

    @pytest.mark.parametrize(
        "dataset_func,invalid_params,error_match",
        [
            (create_spirals, {"n_samples": -1}, "n_samples must be a positive integer"),
            (create_spirals, {"input_dim": 1}, "input_dim must be an integer >= 2"),
            (create_spirals, {"noise": -0.1}, "noise must be a non-negative number"),
            (create_complex_moons, {"n_samples": 0}, "n_samples must be a positive integer"),
            (create_complex_moons, {"input_dim": 0}, "input_dim must be an integer >= 2"),
            (create_moons, {"n_samples": 1}, "n_samples must be at least 2"),
            (create_clusters, {"cluster_count": 0}, "cluster_count must be a positive integer"),
            (create_clusters, {"cluster_size": -1}, "cluster_size must be a positive integer"),
            (create_spheres, {"sphere_count": 0}, "sphere_count must be a positive integer"),
            (create_spheres, {"sphere_size": -1}, "sphere_size must be a positive integer"),
        ],
    )
    def test_parameter_validation_across_functions(self, dataset_func, invalid_params, error_match):
        """Test parameter validation across all dataset functions."""
        with pytest.raises(ValueError, match=error_match):
            dataset_func(**invalid_params)

    @pytest.mark.parametrize(
        "dataset_func,params",
        [
            (create_spirals, {"n_samples": 100}),
            (create_complex_moons, {"n_samples": 100}),
            (create_moons, {"n_samples": 100}),
            (create_clusters, {"cluster_count": 2, "cluster_size": 50}),
            (create_spheres, {"sphere_count": 2, "sphere_size": 50, "sphere_radii": "1,2"}),
        ],
    )
    def test_reproducibility_across_functions(self, dataset_func, params):
        """Test reproducibility across all dataset functions."""
        X1, y1 = dataset_func(**params)
        X2, y2 = dataset_func(**params)

        np.testing.assert_array_equal(X1, X2, f"{dataset_func.__name__} should be reproducible")
        np.testing.assert_array_equal(y1, y2, f"{dataset_func.__name__} labels should be reproducible")


class TestAdvancedErrorHandling:
    """Test advanced error handling and edge cases."""

    def test_extreme_parameter_ranges(self):
        """Test datasets with extreme but valid parameter ranges."""
        # Very small datasets
        X, _ = create_spirals(n_samples=2)
        assert X.shape == (2, 2), "Should handle minimum sample size"

        # Very high dimensions
        X, _ = create_spirals(n_samples=10, input_dim=100)
        assert X.shape == (10, 100), "Should handle high dimensions"

        # Very low noise
        X, _ = create_spirals(n_samples=10, noise=0.0)
        assert not np.any(np.isnan(X)), "Zero noise should not create NaN"

        # Very high noise (but still reasonable)
        X, _ = create_spirals(n_samples=10, noise=10.0)
        assert np.all(np.isfinite(X)), "High noise should still produce finite values"

    def test_sphere_radii_edge_cases(self):
        """Test edge cases for sphere radii parsing."""
        # Single sphere
        X, _ = create_spheres(sphere_count=1, sphere_size=10, sphere_radii="5", input_dim=3)
        assert X.shape == (10, 3), "Single sphere should work"

        # Very small radii
        X, _ = create_spheres(sphere_count=2, sphere_size=5, sphere_radii="0.01,0.02", input_dim=3)
        assert X.shape == (10, 3), "Very small radii should work"

        # Radii with whitespace
        X, _ = create_spheres(sphere_count=2, sphere_size=5, sphere_radii=" 1 , 2 ", input_dim=3)
        assert X.shape == (10, 3), "Radii with whitespace should be handled"

    def test_cluster_geometry_edge_cases(self):
        """Test edge cases for cluster geometry."""
        # Single cluster
        X, _ = create_clusters(cluster_count=1, cluster_size=10, input_dim=2)
        assert X.shape == (10, 2), "Single cluster should work"

        # Very small separation
        X, _ = create_clusters(cluster_count=2, cluster_size=5, cluster_sep=0.1, input_dim=2)
        assert X.shape == (10, 2), "Very small cluster separation should work"

        # Very large separation
        X, _ = create_clusters(cluster_count=2, cluster_size=5, cluster_sep=100.0, input_dim=2)
        assert X.shape == (10, 2), "Very large cluster separation should work"

    def test_comprehensive_error_messages(self):
        """Test that error messages are comprehensive and helpful."""
        # Test type errors - use valid types but invalid values to avoid type checker issues
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            create_spirals(n_samples=-999)  # Use invalid int instead of string

        with pytest.raises(ValueError, match="input_dim must be an integer >= 2"):
            create_spirals(input_dim=1)  # Use invalid int instead of float

        # Test sphere radii parsing errors
        with pytest.raises(ValueError, match="sphere_radii must be comma-separated numbers"):
            create_spheres(sphere_count=2, sphere_size=10, sphere_radii="a,b", input_dim=3)

        with pytest.raises(ValueError, match="Number of radii .* must match sphere_count"):
            create_spheres(sphere_count=3, sphere_size=10, sphere_radii="1,2", input_dim=3)


class TestPropertyBasedDatasets:
    """Property-based testing for dataset generation functions."""

    @given(
        n_samples=st.integers(min_value=2, max_value=1000),
        noise=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        input_dim=st.integers(min_value=2, max_value=10),
    )
    def test_spiral_properties(self, n_samples, noise, input_dim):
        """Test that spiral datasets satisfy fundamental properties."""
        X, y = create_spirals(n_samples=n_samples, noise=noise, input_dim=input_dim)

        # Shape properties
        expected_samples = 2 * (n_samples // 2)  # Balanced classes
        assert X.shape == (expected_samples, input_dim)
        assert y.shape == (expected_samples,)

        # Data type properties
        assert X.dtype == np.float32
        assert y.dtype == np.int64

        # Value properties
        assert np.all(np.isfinite(X)), "All values should be finite"
        assert set(y) == {0, 1}, "Should have binary labels"

        # Class balance property
        class_counts = np.bincount(y)
        assert len(class_counts) == 2, "Should have exactly 2 classes"
        assert abs(class_counts[0] - class_counts[1]) <= 1, "Classes should be balanced"

    @given(
        cluster_count=st.integers(min_value=2, max_value=5),  # Start with 2 for binary labels
        cluster_size=st.integers(min_value=1, max_value=100),
        input_dim=st.integers(min_value=2, max_value=8),
    )
    def test_cluster_properties(self, cluster_count, cluster_size, input_dim):
        """Test that cluster datasets satisfy fundamental properties."""
        # Ensure we have at least 2 total samples
        assume(cluster_count * cluster_size >= 2)

        X, y = create_clusters(cluster_count=cluster_count, cluster_size=cluster_size, input_dim=input_dim)

        # Shape properties
        expected_samples = cluster_count * cluster_size
        assert X.shape == (expected_samples, input_dim)
        assert y.shape == (expected_samples,)

        # Data properties
        assert X.dtype == np.float32
        assert y.dtype == np.int64
        assert np.all(np.isfinite(X)), "All values should be finite"
        assert set(y) == {0, 1}, "Should have binary labels (due to modulo)"

    @given(
        sphere_count=st.integers(min_value=2, max_value=4),  # Start with 2 for binary labels
        sphere_size=st.integers(min_value=1, max_value=50),
        input_dim=st.integers(min_value=2, max_value=6),
    )
    def test_sphere_properties(self, sphere_count, sphere_size, input_dim):
        """Test that sphere datasets satisfy fundamental properties."""
        # Ensure we have at least 2 total samples
        assume(sphere_count * sphere_size >= 2)

        # Generate valid radii string
        radii = [str(i + 1) for i in range(sphere_count)]
        sphere_radii = ",".join(radii)

        X, y = create_spheres(
            sphere_count=sphere_count,
            sphere_size=sphere_size,
            sphere_radii=sphere_radii,
            input_dim=input_dim,
        )

        # Shape properties
        expected_samples = sphere_count * sphere_size
        assert X.shape == (expected_samples, input_dim)
        assert y.shape == (expected_samples,)

        # Data properties
        assert X.dtype == np.float32
        assert y.dtype == np.int64
        assert np.all(np.isfinite(X)), "All values should be finite"
        assert set(y) == {0, 1}, "Should have binary labels (due to modulo)"

    @given(
        n_samples=st.integers(min_value=2, max_value=500),
        input_dim=st.integers(min_value=2, max_value=8),
    )
    def test_reproducibility_property(self, n_samples, input_dim):
        """Test that all datasets are reproducible with same parameters."""
        # Test spirals
        X1, y1 = create_spirals(n_samples=n_samples, input_dim=input_dim)
        X2, y2 = create_spirals(n_samples=n_samples, input_dim=input_dim)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

        # Test complex moons
        X1, y1 = create_complex_moons(n_samples=n_samples, input_dim=input_dim)
        X2, y2 = create_complex_moons(n_samples=n_samples, input_dim=input_dim)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestDocumentationExamples:
    """Test examples that could be included in documentation."""

    def test_single_class_behavior_examples(self):
        """Test and document the single-class behavior for clusters and spheres."""
        # Example 1: Single cluster produces single-class output
        X_single, y_single = create_clusters(cluster_count=1, cluster_size=100, input_dim=2)
        assert set(y_single) == {0}, "Single cluster always produces class 0"
        assert X_single.shape == (100, 2), "Shape should match expected"

        # Example 2: Multiple clusters produce binary classification
        X_multi, y_multi = create_clusters(cluster_count=3, cluster_size=50, input_dim=2)
        assert set(y_multi) == {0, 1}, "Multiple clusters produce binary via modulo"
        assert X_multi.shape == (150, 2), "Shape should match total samples"

        # Example 3: Single sphere produces single-class output
        X_sphere_single, y_sphere_single = create_spheres(sphere_count=1, sphere_size=80, sphere_radii="2.0", input_dim=3)
        assert set(y_sphere_single) == {0}, "Single sphere always produces class 0"
        assert X_sphere_single.shape == (80, 3), "Shape should match expected"

        # Example 4: Multiple spheres produce binary classification
        X_sphere_multi, y_sphere_multi = create_spheres(sphere_count=4, sphere_size=25, sphere_radii="1,2,3,4", input_dim=3)
        assert set(y_sphere_multi) == {0, 1}, "Multiple spheres produce binary via modulo"
        assert X_sphere_multi.shape == (100, 3), "Shape should match total samples"

    def test_robust_functions_always_binary(self):
        """Test that spirals, moons, and complex_moons always produce binary classification."""
        functions_to_test: list[tuple[Callable, dict[str, int]]] = [
            (create_spirals, {"n_samples": 50, "input_dim": 2}),
            (create_moons, {"n_samples": 50, "input_dim": 2}),
            (create_complex_moons, {"n_samples": 50, "input_dim": 2}),
        ]

        for func, params in functions_to_test:
            X, y = func(**params)
            assert set(y) == {0, 1}, f"{func.__name__} should always produce binary classification"
            assert not np.any(np.isnan(X)), f"{func.__name__} should not produce NaN"
            assert not np.any(np.isinf(X)), f"{func.__name__} should not produce Inf"

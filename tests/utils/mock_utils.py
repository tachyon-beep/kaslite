"""Mock utilities for testing.

This module provides mock objects, data classes, and seed manager utilities
for comprehensive testing of the morphogenetic engine.
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import torch

from morphogenetic_engine.core import SeedManager


@dataclass
class MockExperimentArgs:
    """Mock experiment arguments for testing.

    This class provides a comprehensive set of default arguments for testing
    morphogenetic experiments. All arguments have sensible defaults that can
    be overridden as needed for specific test cases.

    Attributes:
        problem_type: Type of problem to solve ('spirals', 'moons', etc.)
        n_samples: Number of samples in the dataset
        input_dim: Dimensionality of input features
        train_frac: Fraction of data used for training
        batch_size: Mini-batch size for training
        device: Device to run computations on ('cpu' or 'cuda')
        seed: Random seed for reproducibility
        warm_up_epochs: Number of warm-up epochs before adaptation
        adaptation_epochs: Number of epochs with adaptive behavior
        lr: Base learning rate
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers in the network
        seeds_per_layer: Number of adaptive seeds per layer
        blend_steps: Steps for blending adaptive behavior
        shadow_lr: Learning rate for shadow network
        progress_thresh: Threshold for progress detection
        drift_warn: Warning threshold for drift detection
        acc_threshold: Target accuracy threshold

        # Dataset-specific parameters
        noise: Noise level for synthetic datasets
        rotations: Number of rotations for spiral datasets
        moon_noise: Noise level for moon datasets
        moon_sep: Separation distance for moon datasets
        cluster_count: Number of clusters
        cluster_size: Size of each cluster
        cluster_std: Standard deviation within clusters
        cluster_sep: Separation between clusters
        sphere_count: Number of spheres in sphere datasets
        sphere_size: Size of each sphere
        sphere_radii: List of sphere radii
        sphere_noise: Noise level for sphere datasets
    """

    problem_type: str = "spirals"
    n_samples: int = 1000
    input_dim: int = 2
    train_frac: float = 0.8
    batch_size: int = 32
    device: str = "cpu"
    seed: int = 42
    warm_up_epochs: int = 10
    adaptation_epochs: int = 50
    epochs: int = 100  # Total epochs (alias for max_epochs)
    max_epochs: int = 100  # Maximum number of epochs
    lr: float = 0.001
    hidden_dim: int = 64
    num_layers: int = 3
    seeds_per_layer: int = 4
    blend_steps: int = 30
    shadow_lr: float = 0.001
    progress_thresh: float = 0.6
    drift_warn: float = 0.12
    acc_threshold: float = 0.95
    # Dataset-specific parameters
    noise: float = 0.1
    rotations: float = 1.0
    moon_noise: float = 0.1
    moon_sep: float = 1.0
    cluster_count: int = 3
    cluster_size: int = 100
    cluster_std: float = 0.5
    cluster_sep: float = 2.0
    sphere_count: int = 2
    sphere_size: int = 500
    sphere_radii: list[float] = field(default_factory=lambda: [1.0, 2.0])
    sphere_noise: float = 0.1

    def __post_init__(self) -> None:
        """Validate arguments after initialization.

        Raises:
            ValueError: If any argument values are invalid.
        """
        self._validate_arguments()

    def _validate_arguments(self) -> None:
        """Validate that all arguments have sensible values."""
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")

        if not 0.0 < self.train_frac <= 1.0:
            raise ValueError(f"train_frac must be in (0, 1], got {self.train_frac}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {self.device}")

        valid_problems = {"spirals", "moons", "complex_moons", "clusters", "spheres"}
        if self.problem_type not in valid_problems:
            raise ValueError(
                f"problem_type must be one of {valid_problems}, got {self.problem_type}"
            )

        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")

        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        if not 0.0 <= self.acc_threshold <= 1.0:
            raise ValueError(f"acc_threshold must be in [0, 1], got {self.acc_threshold}")

    @property
    def learning_rate(self) -> float:
        """Alias for lr to provide more intuitive access."""
        return self.lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Setter for learning_rate that updates lr."""
        self.lr = value


class MockDataLoader:
    """Mock DataLoader for testing.

    This class mimics the behavior of PyTorch DataLoader for testing purposes.
    """

    def __init__(
        self, batch_size: int = 32, num_batches: int = 10, input_dim: int = 2, num_classes: int = 2
    ):
        """Initialize mock DataLoader.

        Args:
            batch_size: Size of each batch
            num_batches: Number of batches to yield
            input_dim: Dimensionality of input features
            num_classes: Number of output classes

        Raises:
            ValueError: If any parameter is non-positive
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_batches <= 0:
            raise ValueError(f"num_batches must be positive, got {num_batches}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.input_dim = input_dim
        self.num_classes = num_classes
        self._current_batch = 0

    def __iter__(self):
        """Initialize iterator."""
        self._current_batch = 0
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch.

        Returns:
            Tuple of (inputs, targets)

        Raises:
            StopIteration: When all batches have been yielded
        """
        if self._current_batch >= self.num_batches:
            raise StopIteration

        # Import here to avoid circular dependency
        from .tensor_utils import create_test_input, create_test_target

        inputs = create_test_input(self.batch_size, self.input_dim)
        targets = create_test_target(self.batch_size, self.num_classes)

        self._current_batch += 1
        return inputs, targets

    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches


# =============================================================================
# Mock Creation Functions
# =============================================================================


def create_mock_args(**overrides: Any) -> MockExperimentArgs:
    """Create mock experiment arguments with optional overrides.

    Args:
        **overrides: Any arguments to override from defaults

    Returns:
        MockExperimentArgs instance with specified overrides

    Raises:
        ValueError: If an unknown argument is provided
    """
    # Common aliases for argument names
    aliases = {
        "learning_rate": "lr",
        "num_epochs": "epochs",
        "max_epochs": "epochs",
        "hidden_size": "hidden_dim",
        "train_fraction": "train_frac",
    }

    args = MockExperimentArgs()
    for key, value in overrides.items():
        # Check if key is an alias and map it to the actual attribute name
        actual_key = aliases.get(key, key)

        if hasattr(args, actual_key):
            setattr(args, actual_key, value)
        else:
            raise ValueError(f"Unknown argument: {key} (mapped to {actual_key})")
    return args


def create_test_seed_manager() -> SeedManager:
    """Create a seed manager for testing.

    Returns:
        Fresh SeedManager instance for testing
    """
    return SeedManager()


def create_configured_seed_manager(num_seeds: int = 3) -> SeedManager:
    """Create a seed manager with pre-configured seeds.

    Args:
        num_seeds: Number of mock seeds to create

    Returns:
        SeedManager with mock seeds configured

    Raises:
        ValueError: If num_seeds is non-positive
    """
    if num_seeds <= 0:
        raise ValueError(f"num_seeds must be positive, got {num_seeds}")

    manager = SeedManager()
    manager.seeds.clear()  # Clear any default seeds
    for i in range(num_seeds):
        # Create mock seed modules
        mock_seed = MagicMock()
        mock_seed.state = "dormant"
        mock_seed.alpha = 0.0
        manager.seeds[f"seed_{i}"] = {"module": mock_seed, "layer": i}
    return manager


def create_mock_device(device_type: str = "cpu") -> MagicMock:
    """Create a mock device for testing.

    Args:
        device_type: Type of device to mock ('cpu' or 'cuda')

    Returns:
        Mock device object

    Raises:
        ValueError: If device_type is not supported
    """
    if device_type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device type: {device_type}")

    mock_device = MagicMock()
    mock_device.__str__ = MagicMock(return_value=device_type)
    mock_device.type = device_type
    return mock_device


def create_mock_progress_callback() -> MagicMock:
    """Create a mock progress callback for testing.

    Returns:
        Mock progress callback with standard methods
    """
    callback = MagicMock()
    callback.on_epoch_start = MagicMock()
    callback.on_epoch_end = MagicMock()
    callback.on_batch_start = MagicMock()
    callback.on_batch_end = MagicMock()
    callback.on_training_start = MagicMock()
    callback.on_training_end = MagicMock()
    return callback


def create_mock_logger() -> MagicMock:
    """Create a mock logger for testing.

    Returns:
        Mock logger with standard logging methods
    """
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    logger.setLevel = MagicMock()
    return logger


def create_mock_optimizer(lr: float = 0.001) -> MagicMock:
    """Create a mock optimizer for testing.

    Args:
        lr: Learning rate for the mock optimizer

    Returns:
        Mock optimizer with standard methods

    Raises:
        ValueError: If lr is non-positive
    """
    if lr <= 0.0:
        raise ValueError(f"Learning rate must be positive, got {lr}")

    optimizer = MagicMock()
    optimizer.step = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.state_dict = MagicMock(return_value={"lr": lr})
    optimizer.load_state_dict = MagicMock()

    # Mock parameter groups
    optimizer.param_groups = [{"lr": lr, "params": []}]

    return optimizer


def create_mock_scheduler() -> MagicMock:
    """Create a mock learning rate scheduler for testing.

    Returns:
        Mock scheduler with standard methods
    """
    scheduler = MagicMock()
    scheduler.step = MagicMock()
    scheduler.get_last_lr = MagicMock(return_value=[0.001])
    scheduler.state_dict = MagicMock(return_value={})
    scheduler.load_state_dict = MagicMock()
    return scheduler

"""Comprehensive test utilities for morphogenetic engine testing.

This module provides robust testing utilities following modern Python 3.12+ patterns
and the project's coding guidelines. It includes utilities for tensor operations,
mock data generation, file system testing, and experiment configuration.

All utilities follow the project's core principles:
- Clarity is paramount: Code is explicit and easy to understand
- Fail fast, fail loudly: Clear error messages for misuse
- Modern Python 3.12+: Uses latest language features and type hints
"""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator
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
            raise ValueError(f"problem_type must be one of {valid_problems}, got {self.problem_type}")
        
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


# =============================================================================
# Tensor Testing Utilities
# =============================================================================

def create_test_tensor(
    shape: tuple[int, ...], 
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    requires_grad: bool = False,
    seed: int | None = None
) -> torch.Tensor:
    """Create a test tensor with specified properties.
    
    Args:
        shape: Shape of the tensor to create
        dtype: Data type of the tensor
        device: Device to place the tensor on
        requires_grad: Whether tensor should require gradients
        seed: Optional seed for reproducible random values
        
    Returns:
        Test tensor with specified properties
        
    Raises:
        ValueError: If shape contains non-positive dimensions
        RuntimeError: If device is not available
    """
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"All dimensions must be positive, got {shape}")
    
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    return torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)


def create_test_input(
    batch_size: int = 4, 
    input_dim: int = 2,
    seed: int | None = None
) -> torch.Tensor:
    """Create test input tensor.
    
    Args:
        batch_size: Number of samples in the batch
        input_dim: Dimensionality of each input sample
        seed: Optional seed for reproducible values
        
    Returns:
        Input tensor of shape (batch_size, input_dim)
        
    Raises:
        ValueError: If batch_size or input_dim are non-positive
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    
    return create_test_tensor((batch_size, input_dim), seed=seed)


def create_test_target(
    batch_size: int = 4, 
    num_classes: int = 2,
    seed: int | None = None
) -> torch.Tensor:
    """Create test target tensor for classification.
    
    Args:
        batch_size: Number of samples in the batch
        num_classes: Number of classes in the classification problem
        seed: Optional seed for reproducible values
        
    Returns:
        Target tensor of shape (batch_size,) with integer class labels
        
    Raises:
        ValueError: If batch_size or num_classes are non-positive
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    return torch.randint(0, num_classes, (batch_size,))


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    """Assert tensor has expected shape.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        
    Raises:
        AssertionError: If tensor shape doesn't match expected shape
    """
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: torch.dtype | None = None,
    expected_device: str | None = None,
    requires_grad: bool | None = None
) -> None:
    """Assert multiple tensor properties at once.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected shape (optional)
        expected_dtype: Expected data type (optional)
        expected_device: Expected device (optional)
        requires_grad: Expected gradient requirement (optional)
        
    Raises:
        AssertionError: If any property doesn't match expectations
    """
    if expected_shape is not None:
        assert_tensor_shape(tensor, expected_shape)
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, (
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
        )
    
    if expected_device is not None:
        device_str = str(tensor.device)
        assert device_str == expected_device, (
            f"Expected device {expected_device}, got {device_str}"
        )
    
    if requires_grad is not None:
        assert tensor.requires_grad == requires_grad, (
            f"Expected requires_grad {requires_grad}, got {tensor.requires_grad}"
        )


def assert_tensors_close(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor, 
    rtol: float = 1e-5, 
    atol: float = 1e-8,
    equal_nan: bool = False
) -> None:
    """Assert two tensors are close within tolerance.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance parameter
        atol: Absolute tolerance parameter
        equal_nan: Whether to treat NaN values as equal
        
    Raises:
        AssertionError: If tensors are not close within tolerance
        ValueError: If tensors have different shapes
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors must have same shape: {tensor1.shape} vs {tensor2.shape}"
        )
    
    close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not close:
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        assert False, f"Tensors not close within tolerance: max diff = {max_diff:.2e}"


# =============================================================================
# Model Testing Utilities
# =============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Total number of parameters in the model
        
    Raises:
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
    
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Number of trainable parameters in the model
        
    Raises:
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_model_output_shape(
    model: torch.nn.Module, 
    input_tensor: torch.Tensor, 
    expected_output_shape: tuple[int, ...]
) -> None:
    """Assert model produces output of expected shape.
    
    Args:
        model: Model to test
        input_tensor: Input to pass through the model
        expected_output_shape: Expected shape of model output
        
    Raises:
        AssertionError: If output shape doesn't match expected shape
        TypeError: If model is not a PyTorch module
        RuntimeError: If model forward pass fails
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
    
    model.eval()
    try:
        with torch.no_grad():
            output = model(input_tensor)
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}") from e
    
    assert_tensor_shape(output, expected_output_shape)


def create_mock_model(input_dim: int, output_dim: int, hidden_dim: int = 64) -> torch.nn.Module:
    """Create a simple mock model for testing.
    
    Args:
        input_dim: Input dimension size
        output_dim: Output dimension size
        hidden_dim: Hidden layer dimension size
        
    Returns:
        Simple sequential model for testing
        
    Raises:
        ValueError: If any dimension is non-positive
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
    
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim)
    )


def assert_model_in_eval_mode(model: torch.nn.Module) -> None:
    """Assert that model is in evaluation mode.
    
    Args:
        model: Model to check
        
    Raises:
        AssertionError: If model is not in eval mode
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
    
    assert not model.training, "Model should be in eval mode"


def assert_model_in_train_mode(model: torch.nn.Module) -> None:
    """Assert that model is in training mode.
    
    Args:
        model: Model to check
        
    Raises:
        AssertionError: If model is not in training mode
        TypeError: If model is not a PyTorch module
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected torch.nn.Module, got {type(model)}")
    
    assert model.training, "Model should be in training mode"


# =============================================================================
# Seed Manager Testing Utilities
# =============================================================================

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


def assert_seed_state(seed_manager: SeedManager, seed_id: str, expected_state: str) -> None:
    """Assert that a seed is in the expected state.
    
    Args:
        seed_manager: SeedManager to check
        seed_id: ID of the seed to check
        expected_state: Expected state string
        
    Raises:
        AssertionError: If seed state doesn't match
        KeyError: If seed_id doesn't exist
    """
    if seed_id not in seed_manager.seeds:
        raise KeyError(f"Seed {seed_id} not found in manager")
    
    actual_state = seed_manager.seeds[seed_id]["module"].state
    assert actual_state == expected_state, (
        f"Expected seed {seed_id} state {expected_state}, got {actual_state}"
    )


# =============================================================================
# File System Testing Utilities
# =============================================================================

@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for testing.
    
    Yields:
        Path to the temporary directory
        
    Note:
        Directory is automatically cleaned up when context exits
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@contextmanager
def temporary_file(suffix: str = ".txt") -> Generator[Path, None, None]:
    """Create a temporary file for testing.
    
    Args:
        suffix: File suffix/extension
        
    Yields:
        Path to the temporary file
        
    Note:
        File is automatically cleaned up when context exits
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        try:
            yield Path(temp_file.name)
        finally:
            Path(temp_file.name).unlink(missing_ok=True)


def create_test_log_file(content: str, suffix: str = ".log") -> Path:
    """Create a test log file with specified content.
    
    Args:
        content: Content to write to the file
        suffix: File suffix/extension
        
    Returns:
        Path to the created file
        
    Note:
        Caller is responsible for cleaning up the file
    """
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix=suffix, delete=False, encoding='utf-8'
    )
    try:
        temp_file.write(content)
    finally:
        temp_file.close()
    
    return Path(temp_file.name)


def create_test_json_file(data: dict[str, Any], suffix: str = ".json") -> Path:
    """Create a test JSON file with specified data.
    
    Args:
        data: Data to serialize as JSON
        suffix: File suffix/extension
        
    Returns:
        Path to the created file
        
    Raises:
        TypeError: If data is not JSON serializable
        
    Note:
        Caller is responsible for cleaning up the file
    """
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix=suffix, delete=False, encoding='utf-8'
    )
    try:
        json.dump(data, temp_file, indent=2)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Data not JSON serializable: {e}") from e
    finally:
        temp_file.close()
    
    return Path(temp_file.name)


def assert_file_exists(file_path: Path | str) -> None:
    """Assert that a file exists.
    
    Args:
        file_path: Path to the file to check
        
    Raises:
        AssertionError: If file doesn't exist
        TypeError: If file_path is not a valid path type
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    if not isinstance(path, Path):
        raise TypeError(f"Expected Path or str, got {type(file_path)}")
    
    assert path.exists(), f"File does not exist: {path}"
    assert path.is_file(), f"Path exists but is not a file: {path}"


def assert_directory_exists(dir_path: Path | str) -> None:
    """Assert that a directory exists.
    
    Args:
        dir_path: Path to the directory to check
        
    Raises:
        AssertionError: If directory doesn't exist
        TypeError: If dir_path is not a valid path type
    """
    path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    if not isinstance(path, Path):
        raise TypeError(f"Expected Path or str, got {type(dir_path)}")
    
    assert path.exists(), f"Directory does not exist: {path}"
    assert path.is_dir(), f"Path exists but is not a directory: {path}"


# =============================================================================
# Experiment Configuration Testing Utilities
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
        'learning_rate': 'lr',
        'num_epochs': 'epochs',
        'max_epochs': 'epochs',
        'hidden_size': 'hidden_dim',
        'train_fraction': 'train_frac',
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


def create_test_experiment_config(**overrides: Any) -> dict[str, Any]:
    """Create a test experiment configuration.
    
    Args:
        **overrides: Configuration values to override
        
    Returns:
        Dictionary containing experiment configuration
    """
    default_config = {
        "problem_type": "spirals",
        "n_samples": 1000,
        "input_dim": 2,
        "train_frac": 0.8,
        "batch_size": 32,
        "device": "cpu",
        "seed": 42,
        "warm_up_epochs": 10,
        "adaptation_epochs": 50,
        "lr": 0.001,
        "hidden_dim": 64,
        "num_layers": 3,
        "seeds_per_layer": 4,
        "blend_steps": 30,
        "shadow_lr": 0.001,
        "progress_thresh": 0.6,
        "drift_warn": 0.12,
        "acc_threshold": 0.95,
    }
    default_config.update(overrides)
    return default_config


def create_test_final_stats(**overrides: Any) -> dict[str, Any]:
    """Create test final statistics for experiments.
    
    Args:
        **overrides: Statistics values to override
        
    Returns:
        Dictionary containing final experiment statistics
    """
    default_stats = {
        "best_acc": 0.95,
        "accuracy_dip": 0.05,
        "recovery_time": 10,
        "seeds_activated": True,
        "final_acc": 0.94,
        "training_time": 120.5,
        "convergence_epoch": 45,
    }
    default_stats.update(overrides)
    return default_stats


def create_test_sweep_config(**overrides: Any) -> dict[str, Any]:
    """Create test sweep configuration for hyperparameter optimization.
    
    Args:
        **overrides: Sweep configuration values to override
        
    Returns:
        Dictionary containing sweep configuration
    """
    default_config = {
        "method": "grid",
        "parameters": {
            "lr": {"values": [0.001, 0.01, 0.1]},
            "batch_size": {"values": [16, 32, 64]},
            "hidden_dim": {"values": [32, 64, 128]},
        },
        "metric": {
            "name": "best_acc",
            "goal": "maximize"
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10
        }
    }
    default_config.update(overrides)
    return default_config


# =============================================================================
# Assertion Utilities for Domain-Specific Logic
# =============================================================================

def assert_valid_experiment_slug(slug: str) -> None:
    """Assert that an experiment slug follows the expected format.
    
    Args:
        slug: Experiment slug to validate
        
    Raises:
        AssertionError: If slug format is invalid
        TypeError: If slug is not a string
    """
    if not isinstance(slug, str):
        raise TypeError(f"Expected string, got {type(slug)}")
    
    # Handle compound problem types by finding the known problem types
    problem_types = ["complex_moons", "spirals", "moons", "clusters", "spheres"]
    
    # Find which problem type this slug starts with
    problem_type = None
    for pt in problem_types:
        if slug.startswith(pt + "_"):
            problem_type = pt
            break
    
    if problem_type is None:
        assert False, f"Slug doesn't start with a known problem type: {slug}"
    
    # Split the remaining part after the problem type
    remaining = slug[len(problem_type) + 1:]  # +1 to skip the underscore
    parts = remaining.split('_')
    
    # Now we should have: dim{N}, {device}, h{hidden}, bs{batch}, lr{lr}, pt{thresh}, dw{drift}
    assert len(parts) >= 7, (
        f"Slug should have at least 7 parts after problem type, got {len(parts)}: {slug}"
    )
    
    assert parts[0].startswith("dim"), f"Expected dim prefix, got: {parts[0]}"
    
    match parts[1]:
        case "cpu" | "cuda":
            pass  # Valid device
        case _:
            assert False, f"Invalid device in slug: {parts[1]}"
    
    assert parts[2].startswith("h"), f"Expected h prefix for hidden dim, got: {parts[2]}"
    assert parts[3].startswith("bs"), f"Expected bs prefix for batch size, got: {parts[3]}"
    assert parts[4].startswith("lr"), f"Expected lr prefix for learning rate, got: {parts[4]}"
    assert parts[5].startswith("pt"), f"Expected pt prefix for progress thresh, got: {parts[5]}"
    assert parts[6].startswith("dw"), f"Expected dw prefix for drift warn, got: {parts[6]}"


def assert_valid_metrics_json(metrics: dict[str, Any]) -> None:
    """Assert that metrics JSON contains required fields.
    
    Args:
        metrics: Metrics dictionary to validate
        
    Raises:
        AssertionError: If required fields are missing or invalid
        TypeError: If metrics is not a dictionary
    """
    if not isinstance(metrics, dict):
        raise TypeError(f"Expected dict, got {type(metrics)}")
    
    required_fields = ["best_acc"]
    for field in required_fields:
        assert field in metrics, f"Missing required field: {field}"
    
    best_acc = metrics["best_acc"]
    assert isinstance(best_acc, (int, float)), (
        f"best_acc should be numeric, got {type(best_acc)}"
    )
    assert 0.0 <= best_acc <= 1.0, (
        f"best_acc should be between 0 and 1, got: {best_acc}"
    )
    
    # Validate optional fields if present
    if "final_acc" in metrics:
        final_acc = metrics["final_acc"]
        assert isinstance(final_acc, (int, float)), (
            f"final_acc should be numeric, got {type(final_acc)}"
        )
        assert 0.0 <= final_acc <= 1.0, (
            f"final_acc should be between 0 and 1, got: {final_acc}"
        )
    
    if "training_time" in metrics:
        training_time = metrics["training_time"]
        assert isinstance(training_time, (int, float)), (
            f"training_time should be numeric, got {type(training_time)}"
        )
        assert training_time >= 0.0, (
            f"training_time should be non-negative, got: {training_time}"
        )


def assert_log_file_format(log_content: str) -> None:
    """Assert that log file follows expected format.
    
    Args:
        log_content: Content of the log file to validate
        
    Raises:
        AssertionError: If log format is invalid
        TypeError: If log_content is not a string
    """
    if not isinstance(log_content, str):
        raise TypeError(f"Expected string, got {type(log_content)}")
    
    lines = log_content.strip().split('\n')
    assert len(lines) > 0, "Log file should not be empty"
    
    # Check for header
    assert lines[0].startswith("# Morphogenetic Architecture Experiment Log"), (
        "Log should start with experiment header"
    )
    
    # Check for CSV header
    csv_header_found = False
    for line in lines:
        if line == "epoch,seed,state,alpha":
            csv_header_found = True
            break
    assert csv_header_found, "Log should contain CSV header"


def assert_accuracy_range(accuracy: float, min_acc: float = 0.0, max_acc: float = 1.0) -> None:
    """Assert that accuracy is within expected range.
    
    Args:
        accuracy: Accuracy value to check
        min_acc: Minimum allowed accuracy
        max_acc: Maximum allowed accuracy
        
    Raises:
        AssertionError: If accuracy is out of range
        TypeError: If accuracy is not numeric
    """
    if not isinstance(accuracy, (int, float)):
        raise TypeError(f"Expected numeric accuracy, got {type(accuracy)}")
    
    assert min_acc <= accuracy <= max_acc, (
        f"Accuracy {accuracy} not in range [{min_acc}, {max_acc}]"
    )


def assert_convergence_behavior(
    accuracies: list[float], 
    min_final_acc: float = 0.8,
    tolerance: float = 0.05
) -> None:
    """Assert that training shows proper convergence behavior.
    
    Args:
        accuracies: List of accuracies over training epochs
        min_final_acc: Minimum expected final accuracy
        tolerance: Tolerance for convergence detection
        
    Raises:
        AssertionError: If convergence behavior is invalid
        ValueError: If accuracies list is invalid
    """
    if not accuracies:
        raise ValueError("Accuracies list cannot be empty")
    
    if not all(isinstance(acc, (int, float)) for acc in accuracies):
        raise ValueError("All accuracies must be numeric")
    
    final_acc = accuracies[-1]
    assert final_acc >= min_final_acc, (
        f"Final accuracy {final_acc} below minimum {min_final_acc}"
    )
    
    # Check for general upward trend
    if len(accuracies) >= 10:
        early_avg = sum(accuracies[:5]) / 5
        late_avg = sum(accuracies[-5:]) / 5
        assert late_avg >= early_avg - tolerance, (
            f"No improvement detected: early={early_avg:.3f}, late={late_avg:.3f}"
        )


# =============================================================================
# Mock Utilities
# =============================================================================

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
    optimizer.state_dict = MagicMock(return_value={'lr': lr})
    optimizer.load_state_dict = MagicMock()
    
    # Mock parameter groups
    optimizer.param_groups = [{'lr': lr, 'params': []}]
    
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


class MockDataLoader:
    """Mock DataLoader for testing.
    
    This class mimics the behavior of PyTorch DataLoader for testing purposes.
    """
    
    def __init__(
        self, 
        batch_size: int = 32, 
        num_batches: int = 10, 
        input_dim: int = 2, 
        num_classes: int = 2
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
        
        inputs = create_test_input(self.batch_size, self.input_dim)
        targets = create_test_target(self.batch_size, self.num_classes)
        
        self._current_batch += 1
        return inputs, targets
    
    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_batches


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_test_environment() -> None:
    """Validate that the test environment is properly configured.
    
    Raises:
        RuntimeError: If test environment is not properly configured
        ImportError: If required dependencies are missing
    """
    # Check PyTorch
    try:
        import torch
        if not hasattr(torch, '__version__'):
            raise ImportError("PyTorch version information not available")
    except ImportError as e:
        raise ImportError(f"PyTorch not available: {e}") from e
    
    # Check that basic tensor operations work
    try:
        test_tensor = torch.randn(2, 2)
        _ = test_tensor + test_tensor
    except Exception as e:
        raise RuntimeError(f"Basic tensor operations failed: {e}") from e
    
    # Check random seed functionality
    try:
        torch.manual_seed(42)
        tensor1 = torch.randn(5)
        torch.manual_seed(42)
        tensor2 = torch.randn(5)
        if not torch.allclose(tensor1, tensor2):
            raise RuntimeError("Random seed functionality not working")
    except Exception as e:
        raise RuntimeError(f"Random seed validation failed: {e}") from e


def setup_test_logging() -> MagicMock:
    """Set up logging for tests.
    
    Returns:
        Mock logger configured for testing
    """
    logger = create_mock_logger()
    # Configure any default behaviors
    logger.level = "DEBUG"
    return logger

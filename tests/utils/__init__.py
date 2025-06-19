"""Test utilities package for morphogenetic engine testing.

This package provides robust testing utilities following modern Python 3.12+ patterns
and the project's coding guidelines. It includes utilities for tensor operations,
mock data generation, file system testing, and experiment configuration.

All utilities follow the project's core principles:
- Clarity is paramount: Code is explicit and easy to understand
- Fail fast, fail loudly: Clear error messages for misuse
- Modern Python 3.12+: Uses latest language features and type hints
"""

# Mock utilities and data classes
from .mock_utils import (
    MockDataLoader,
    MockExperimentArgs,
    create_configured_seed_manager,
    create_mock_args,
    create_mock_device,
    create_mock_logger,
    create_mock_optimizer,
    create_mock_progress_callback,
    create_mock_scheduler,
    create_test_seed_manager,
)

# Tensor testing utilities
from .tensor_utils import (
    assert_tensor_properties,
    assert_tensor_shape,
    assert_tensors_close,
    create_test_input,
    create_test_target,
    create_test_tensor,
)

# Model testing utilities
from .model_utils import (
    assert_model_in_eval_mode,
    assert_model_in_train_mode,
    assert_model_output_shape,
    count_parameters,
    count_trainable_parameters,
    create_mock_model,
)

# File system utilities
from .file_utils import (
    assert_directory_exists,
    assert_file_exists,
    create_test_json_file,
    create_test_log_file,
    temporary_directory,
    temporary_file,
)

# Experiment configuration utilities
from .config_utils import (
    create_test_experiment_config,
    create_test_final_stats,
    create_test_sweep_config,
)

# Assertion utilities for domain-specific logic
from .assertion_utils import (
    assert_accuracy_range,
    assert_convergence_behavior,
    assert_log_file_format,
    assert_seed_state,
    assert_valid_experiment_slug,
    assert_valid_metrics_json,
)

# Environment validation utilities
from .validation_utils import (
    setup_test_logging,
    validate_test_environment,
)

# Sweep utilities (existing)
from .sweep_test_utils import SweepConfigBuilder

__all__ = [
    # Mock utilities
    "MockDataLoader",
    "MockExperimentArgs",
    "create_configured_seed_manager",
    "create_mock_args",
    "create_mock_device",
    "create_mock_logger",
    "create_mock_optimizer",
    "create_mock_progress_callback",
    "create_mock_scheduler",
    "create_test_seed_manager",
    # Tensor utilities
    "assert_tensor_properties",
    "assert_tensor_shape",
    "assert_tensors_close",
    "create_test_input",
    "create_test_target",
    "create_test_tensor",
    # Model utilities
    "assert_model_in_eval_mode",
    "assert_model_in_train_mode",
    "assert_model_output_shape",
    "count_parameters",
    "count_trainable_parameters",
    "create_mock_model",
    # File utilities
    "assert_directory_exists",
    "assert_file_exists",
    "create_test_json_file",
    "create_test_log_file",
    "temporary_directory",
    "temporary_file",
    # Config utilities
    "create_test_experiment_config",
    "create_test_final_stats",
    "create_test_sweep_config",
    # Assertion utilities
    "assert_accuracy_range",
    "assert_convergence_behavior",
    "assert_log_file_format",
    "assert_seed_state",
    "assert_valid_experiment_slug",
    "assert_valid_metrics_json",
    # Validation utilities
    "setup_test_logging",
    "validate_test_environment",
    # Sweep utilities
    "SweepConfigBuilder",
]

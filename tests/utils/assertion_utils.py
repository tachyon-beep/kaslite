"""Assertion utilities for domain-specific testing.

This module provides specialized assertion functions for validating
experiment results, log formats, metrics, and convergence behavior.
"""

from typing import Any

from morphogenetic_engine.core import SeedManager


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
    assert (
        actual_state == expected_state
    ), f"Expected seed {seed_id} state {expected_state}, got {actual_state}"


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
    remaining = slug[len(problem_type) + 1 :]  # +1 to skip the underscore
    parts = remaining.split("_")

    # Now we should have: dim{N}, {device}, h{hidden}, bs{batch}, lr{lr}, pt{thresh}, dw{drift}
    assert (
        len(parts) >= 7
    ), f"Slug should have at least 7 parts after problem type, got {len(parts)}: {slug}"

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
    assert isinstance(best_acc, (int, float)), f"best_acc should be numeric, got {type(best_acc)}"
    assert 0.0 <= best_acc <= 1.0, f"best_acc should be between 0 and 1, got: {best_acc}"

    # Validate optional fields if present
    if "final_acc" in metrics:
        final_acc = metrics["final_acc"]
        assert isinstance(
            final_acc, (int, float)
        ), f"final_acc should be numeric, got {type(final_acc)}"
        assert 0.0 <= final_acc <= 1.0, f"final_acc should be between 0 and 1, got: {final_acc}"

    if "training_time" in metrics:
        training_time = metrics["training_time"]
        assert isinstance(
            training_time, (int, float)
        ), f"training_time should be numeric, got {type(training_time)}"
        assert training_time >= 0.0, f"training_time should be non-negative, got: {training_time}"


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

    lines = log_content.strip().split("\n")
    assert len(lines) > 0, "Log file should not be empty"

    # Check for header
    assert lines[0].startswith(
        "# Morphogenetic Architecture Experiment Log"
    ), "Log should start with experiment header"

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

    assert (
        min_acc <= accuracy <= max_acc
    ), f"Accuracy {accuracy} not in range [{min_acc}, {max_acc}]"


def assert_convergence_behavior(
    accuracies: list[float], min_final_acc: float = 0.8, tolerance: float = 0.05
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
    assert final_acc >= min_final_acc, f"Final accuracy {final_acc} below minimum {min_final_acc}"

    # Check for general upward trend
    if len(accuracies) >= 10:
        early_avg = sum(accuracies[:5]) / 5
        late_avg = sum(accuracies[-5:]) / 5
        assert (
            late_avg >= early_avg - tolerance
        ), f"No improvement detected: early={early_avg:.3f}, late={late_avg:.3f}"

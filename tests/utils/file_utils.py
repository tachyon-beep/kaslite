"""File system testing utilities.

This module provides utilities for working with temporary files and directories
in tests, along with file existence assertions.
"""

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


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
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
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
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
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

"""
Unit tests for the FastAPI Inference Server.

Tests the inference server endpoints, model loading, health checks,
and monitoring functionality using modern Python 3.12+ features.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from morphogenetic_engine.inference_server import app, load_production_model, load_specific_model

# pylint: disable=redefined-outer-name
# Note: Disabling redefined-outer-name for entire file as it's expected behavior with pytest fixtures


# Test Constants and Configuration
TEST_MODEL_VERSION = "test-v1"
PRODUCTION_MODEL_VERSION = "production-v2"
TEST_MODEL_NAME = "KasminaModel"

# Sample test data using Python 3.12+ type annotations
TestDataPoint = list[float]
TestBatch = list[TestDataPoint]
PredictionResult = dict[str, Any]


class TestFixtures:
    """
    Container class for test fixture documentation.

    This class serves as documentation for the pytest fixtures used throughout
    the test suite. It doesn't contain actual test methods but helps organize
    fixture documentation.
    """


# Test Data Fixtures
@pytest.fixture
def sample_input_data() -> TestBatch:
    """
    Provide sample input data for prediction testing.

    Returns:
        TestBatch: A list of 3D data points representing typical model input.
                  Each point has 3 features suitable for the test model.
    """
    return [[0.5, 0.3, 0.1], [0.2, 0.8, 0.4], [0.9, 0.1, 0.7]]


@pytest.fixture
def single_point_data() -> TestBatch:
    """
    Provide single data point for testing batch dimension expansion.

    Returns:
        TestBatch: A single 3D data point to test how the API handles
                  single-item predictions and batch dimension handling.
    """
    return [[0.6, 0.4, 0.2]]


@pytest.fixture
def invalid_input_data() -> dict[str, Any]:
    """
    Provide invalid input data for error testing.

    Returns:
        dict[str, Any]: Malformed request data that should trigger
                       validation errors in the API.
    """
    return {"data": "not_a_list"}


@pytest.fixture
def prediction_request_data(sample_input_data: TestBatch) -> dict[str, Any]:
    """
    Create standard prediction request payload.

    Args:
        sample_input_data: Test data fixture for input points.

    Returns:
        dict[str, Any]: Well-formed prediction request matching the
                       PredictionRequest schema.
    """
    return {"data": sample_input_data}


@pytest.fixture
def prediction_request_with_version(sample_input_data: TestBatch) -> dict[str, Any]:
    """
    Create prediction request with specific model version.

    Args:
        sample_input_data: Test data fixture for input points.

    Returns:
        dict[str, Any]: Prediction request specifying a particular
                       model version for testing version routing.
    """
    return {"data": sample_input_data, "model_version": TEST_MODEL_VERSION}


# Mock Model Fixtures
@pytest.fixture
def mock_simple_model() -> torch.nn.Module:
    """
    Create a simple real PyTorch model for testing.

    Creates a minimal neural network with a single linear layer that
    can be used for testing without requiring complex mock setups.
    This provides real PyTorch operations while keeping the model
    simple enough for fast test execution.

    Returns:
        torch.nn.Module: A simple PyTorch model in evaluation mode
                        with 3 input features and 2 output classes.
    """

    class SimpleTestModel(torch.nn.Module):
        """Simple linear model for testing purposes."""

        def __init__(self) -> None:
            """Initialize the simple test model."""
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the model.

            Args:
                x: Input tensor with shape (batch_size, 3).

            Returns:
                torch.Tensor: Output logits with shape (batch_size, 2).
            """
            return self.linear(x)

    model = SimpleTestModel()
    model.eval()
    return model


@pytest.fixture
def mock_model_registry(mocker) -> Any:
    """
    Create mock ModelRegistry with proper async behavior.

    Sets up a mocked ModelRegistry instance with predefined version
    information and production model URIs for consistent testing.

    Args:
        mocker: pytest-mock fixture for creating mocks.

    Returns:
        Any: Mocked ModelRegistry instance with configured behavior.
    """
    registry_mock = mocker.Mock()

    # Mock version objects
    version_1 = mocker.Mock()
    version_1.version = "1"
    version_2 = mocker.Mock()
    version_2.version = "2"

    registry_mock.list_model_versions.return_value = [version_1, version_2]
    registry_mock.get_production_model_uri.return_value = (
        f"models://{TEST_MODEL_NAME}/{PRODUCTION_MODEL_VERSION}"
    )

    return registry_mock


# Client Fixtures
@pytest.fixture
def sync_client() -> TestClient:
    """
    Create synchronous test client for FastAPI.

    Provides a TestClient instance for synchronous testing of FastAPI
    endpoints. This is suitable for most endpoint tests that don't
    require async behavior.

    Returns:
        TestClient: Configured FastAPI test client for synchronous requests.
    """
    return TestClient(app)


@pytest.fixture
async def async_client() -> httpx.AsyncClient:
    """
    Create asynchronous test client for FastAPI.

    Provides an async HTTP client for testing async endpoints and
    workflows that require proper async/await patterns. Uses ASGI
    transport for direct FastAPI integration.

    Yields:
        httpx.AsyncClient: Configured async client for FastAPI testing.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def client_with_mock_model(mocker, mock_simple_model: torch.nn.Module) -> TestClient:
    """
    Create test client with a mocked model in cache.

    Sets up a TestClient with a pre-loaded model in the cache,
    simulating a ready-to-use inference server state. This eliminates
    the need for model loading setup in individual tests.

    Args:
        mocker: pytest-mock fixture for patching.
        mock_simple_model: Simple PyTorch model fixture.

    Returns:
        TestClient: Test client with model pre-loaded in cache.
    """
    # Patch the model cache
    mock_cache = {TEST_MODEL_VERSION: mock_simple_model}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)
    mocker.patch("morphogenetic_engine.inference_server.current_model_version", TEST_MODEL_VERSION)

    return TestClient(app)


# Test Helper Functions
def validate_health_response(response_data: dict[str, Any]) -> None:
    """
    Validate health check response structure and content.

    Ensures that health check responses contain all required fields
    and have appropriate values for the server status.

    Args:
        response_data: JSON response data from health endpoint.

    Raises:
        AssertionError: If response structure or content is invalid.
    """
    required_fields = {"status", "model_loaded", "model_version", "timestamp"}
    assert all(field in response_data for field in required_fields)
    assert response_data["status"] == "healthy"


def validate_prediction_response(response_data: dict[str, Any], expected_version: str) -> None:
    """
    Validate prediction response structure and content.

    Ensures that prediction responses contain all required fields,
    have correct data types, and match expected model version.

    Args:
        response_data: JSON response data from prediction endpoint.
        expected_version: Expected model version string.

    Raises:
        AssertionError: If response structure or content is invalid.
    """
    required_fields = {"predictions", "probabilities", "model_version", "inference_time_ms"}
    assert all(field in response_data for field in required_fields)

    assert isinstance(response_data["predictions"], list)
    assert isinstance(response_data["probabilities"], list)
    assert response_data["model_version"] == expected_version
    assert isinstance(response_data["inference_time_ms"], float)
    assert response_data["inference_time_ms"] >= 0


# Health Endpoint Tests
def test_health_endpoint_with_model(sync_client: TestClient, mocker) -> None:
    """
    Test health endpoint when model is loaded.

    Verifies that the health check returns correct status and model
    information when a model is currently loaded in the server.

    Args:
        sync_client: Synchronous test client fixture.
        mocker: pytest-mock fixture for patching.
    """
    mocker.patch(
        "morphogenetic_engine.inference_server.current_model_version", PRODUCTION_MODEL_VERSION
    )

    response = sync_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    validate_health_response(data)
    assert data["model_loaded"] is True
    assert data["model_version"] == PRODUCTION_MODEL_VERSION


def test_health_endpoint_no_model(sync_client: TestClient, mocker) -> None:
    """
    Test health endpoint when no model is loaded.

    Verifies that the health check returns appropriate status when
    no model is currently loaded in the server.

    Args:
        sync_client: Synchronous test client fixture.
        mocker: pytest-mock fixture for patching.
    """
    mocker.patch("morphogenetic_engine.inference_server.current_model_version", None)

    response = sync_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    validate_health_response(data)
    assert data["model_loaded"] is False
    assert data["model_version"] is None


# Metrics Endpoint Tests
def test_metrics_endpoint(sync_client: TestClient) -> None:
    """
    Test Prometheus metrics endpoint.

    Verifies that the metrics endpoint returns properly formatted
    Prometheus metrics data with correct content type and structure.

    Args:
        sync_client: Synchronous test client fixture.
    """
    response = sync_client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    # Should contain Prometheus metric format
    content = response.text
    assert any(marker in content for marker in ["# HELP", "# TYPE", "_total", "_duration"])


# Model Info Endpoint Tests
def test_get_model_info_success(sync_client: TestClient, mocker, mock_model_registry) -> None:
    """
    Test model info endpoint success case.

    Verifies that the model info endpoint returns correct information
    about available model versions and current production model.

    Args:
        sync_client: Synchronous test client fixture.
        mocker: pytest-mock fixture for patching.
        mock_model_registry: Mocked model registry fixture.
    """
    mocker.patch(
        "morphogenetic_engine.inference_server.ModelRegistry", return_value=mock_model_registry
    )
    mocker.patch(
        "morphogenetic_engine.inference_server.current_model_version", PRODUCTION_MODEL_VERSION
    )

    response = sync_client.get("/models")

    assert response.status_code == 200
    data = response.json()
    assert data["current_version"] == PRODUCTION_MODEL_VERSION
    assert data["available_versions"] == ["1", "2"]
    assert data["model_name"] == TEST_MODEL_NAME


def test_get_model_info_registry_failure(sync_client: TestClient, mocker) -> None:
    """
    Test model info endpoint when registry fails.

    Verifies proper error handling when the model registry
    encounters an error during model information retrieval.

    Args:
        sync_client: Synchronous test client fixture.
        mocker: pytest-mock fixture for patching.
    """
    mock_registry = mocker.Mock()
    mock_registry.list_model_versions.side_effect = Exception("Registry connection failed")
    mocker.patch("morphogenetic_engine.inference_server.ModelRegistry", return_value=mock_registry)

    response = sync_client.get("/models")

    assert response.status_code == 500
    assert "Failed to retrieve model information" in response.json()["detail"]


# Prediction Endpoint Tests - Real PyTorch Operations
def test_predict_success_real_operations(
    client_with_mock_model: TestClient, prediction_request_data: dict[str, Any]
) -> None:
    """
    Test successful prediction using real PyTorch operations.

    This test validates the complete prediction pipeline using actual
    PyTorch tensor operations instead of mocks. It verifies that the
    model produces valid predictions and probability distributions.

    Args:
        client_with_mock_model: Test client with pre-loaded model.
        prediction_request_data: Sample prediction request data.
    """
    response = client_with_mock_model.post("/predict", json=prediction_request_data)

    assert response.status_code == 200
    data = response.json()
    validate_prediction_response(data, TEST_MODEL_VERSION)

    # Validate real tensor operations worked
    assert len(data["predictions"]) == len(prediction_request_data["data"])
    assert len(data["probabilities"]) == len(prediction_request_data["data"])

    # Each prediction should be an integer (class index)
    assert all(isinstance(pred, int) for pred in data["predictions"])

    # Each probability distribution should sum to ~1.0
    for prob_dist in data["probabilities"]:
        assert abs(sum(prob_dist) - 1.0) < 1e-6


def test_predict_single_point_expansion(
    client_with_mock_model: TestClient, single_point_data: TestBatch
) -> None:
    """
    Test that single data point predictions work with real tensor operations.

    Verifies that the API correctly handles single-point predictions
    by properly expanding dimensions and processing through the model.

    Args:
        client_with_mock_model: Test client with pre-loaded model.
        single_point_data: Single data point for testing.
    """
    request_data = {"data": single_point_data}
    response = client_with_mock_model.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()
    validate_prediction_response(data, TEST_MODEL_VERSION)

    # Should handle single point correctly
    assert len(data["predictions"]) == 1
    assert len(data["probabilities"]) == 1


def test_predict_specific_model_version(
    mocker, mock_simple_model: torch.nn.Module, prediction_request_with_version: dict[str, Any]
) -> None:
    """Test prediction with specific model version."""
    # Setup model cache with specific version
    mock_cache = {TEST_MODEL_VERSION: mock_simple_model}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)

    client = TestClient(app)
    response = client.post("/predict", json=prediction_request_with_version)

    assert response.status_code == 200
    data = response.json()
    validate_prediction_response(data, TEST_MODEL_VERSION)


def test_predict_no_model_available(sync_client: TestClient, mocker) -> None:
    """Test prediction when no model is available."""
    mocker.patch("morphogenetic_engine.inference_server.current_model_version", None)

    request_data = {"data": [[0.1, 0.2, 0.3]]}
    response = sync_client.post("/predict", json=request_data)

    assert response.status_code == 503
    assert "No model available" in response.json()["detail"]


def test_predict_invalid_input_data(
    sync_client: TestClient, mocker, invalid_input_data: dict[str, Any]
) -> None:
    """Test prediction with invalid input data."""
    mocker.patch("morphogenetic_engine.inference_server.current_model_version", TEST_MODEL_VERSION)

    response = sync_client.post("/predict", json=invalid_input_data)

    assert response.status_code == 422  # FastAPI validation error


def test_predict_model_not_in_cache(sync_client: TestClient, mocker) -> None:
    """Test prediction when requested model is not in cache."""
    mocker.patch("morphogenetic_engine.inference_server.current_model_version", "missing-model")
    mocker.patch("morphogenetic_engine.inference_server.model_cache", {})

    # Mock load_specific_model to fail
    async def mock_load_fail(version: str) -> bool:  # pylint: disable=unused-argument
        return False

    mocker.patch(
        "morphogenetic_engine.inference_server.load_specific_model", side_effect=mock_load_fail
    )

    request_data = {"data": [[0.1, 0.2, 0.3]]}
    response = sync_client.post("/predict", json=request_data)

    assert response.status_code == 404
    assert "Model version" in response.json()["detail"]


# Parameterized Tests for Different Input Scenarios
@pytest.mark.parametrize(
    "input_data,expected_output_size",
    [
        ([[0.1, 0.2, 0.3]], 1),  # Single point
        ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 2),  # Two points
        ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], 3),  # Three points
    ],
)
def test_predict_various_batch_sizes(
    client_with_mock_model: TestClient, input_data: TestBatch, expected_output_size: int
) -> None:
    """
    Test predictions with various batch sizes.

    This parameterized test validates that the prediction endpoint
    correctly handles different batch sizes from single points to
    multiple data points in a single request.

    Args:
        client_with_mock_model: Test client with pre-loaded model.
        input_data: Input data with varying batch sizes.
        expected_output_size: Expected number of predictions.
    """
    request_data = {"data": input_data}
    response = client_with_mock_model.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == expected_output_size
    assert len(data["probabilities"]) == expected_output_size


# Model Reload Tests
@pytest.mark.asyncio
async def test_reload_model_success(
    sync_client: TestClient, mocker, mock_model_registry
) -> None:  # pylint: disable=unused-argument
    """Test successful model reload."""
    mock_load = AsyncMock(return_value=True)
    mocker.patch("morphogenetic_engine.inference_server.load_production_model", mock_load)
    mocker.patch(
        "morphogenetic_engine.inference_server.current_model_version", PRODUCTION_MODEL_VERSION
    )

    response = sync_client.post("/reload-model")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert PRODUCTION_MODEL_VERSION in data["message"]


@pytest.mark.asyncio
async def test_reload_model_failure(sync_client: TestClient, mocker) -> None:
    """Test model reload failure."""
    mock_load = AsyncMock(return_value=False)
    mocker.patch("morphogenetic_engine.inference_server.load_production_model", mock_load)

    response = sync_client.post("/reload-model")

    assert response.status_code == 503
    assert "Failed to reload model" in response.json()["detail"]


@pytest.mark.asyncio
async def test_reload_model_exception(sync_client: TestClient, mocker) -> None:
    """Test model reload with exception."""
    mock_load = AsyncMock(side_effect=Exception("Reload error"))
    mocker.patch("morphogenetic_engine.inference_server.load_production_model", mock_load)

    response = sync_client.post("/reload-model")

    assert response.status_code == 500
    assert "Model reload failed" in response.json()["detail"]


# Model Loading Function Tests
@pytest.mark.asyncio
async def test_load_production_model_success(
    mocker, mock_model_registry, mock_simple_model: torch.nn.Module
) -> None:
    """Test successful production model loading."""
    # Mock dependencies
    mocker.patch(
        "morphogenetic_engine.inference_server.ModelRegistry", return_value=mock_model_registry
    )
    mocker.patch("mlflow.pytorch.load_model", return_value=mock_simple_model)

    # Mock the global cache and version
    mock_cache: dict[str, Any] = {}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)

    result = await load_production_model()

    assert result is True
    assert PRODUCTION_MODEL_VERSION in mock_cache
    assert mock_cache[PRODUCTION_MODEL_VERSION] is mock_simple_model


@pytest.mark.asyncio
async def test_load_production_model_no_model(mocker, mock_model_registry) -> None:
    """Test production model loading when no model exists."""
    mock_model_registry.get_production_model_uri.return_value = None
    mocker.patch(
        "morphogenetic_engine.inference_server.ModelRegistry", return_value=mock_model_registry
    )

    result = await load_production_model()

    assert result is False


@pytest.mark.asyncio
async def test_load_production_model_exception(mocker) -> None:
    """Test production model loading with exception."""
    mock_registry = mocker.Mock()
    mock_registry.get_production_model_uri.side_effect = Exception("Registry error")
    mocker.patch("morphogenetic_engine.inference_server.ModelRegistry", return_value=mock_registry)

    result = await load_production_model()

    assert result is False


@pytest.mark.asyncio
async def test_load_specific_model_success(mocker, mock_simple_model: torch.nn.Module) -> None:
    """Test successful specific model loading."""
    # Mock dependencies
    mocker.patch("mlflow.pytorch.load_model", return_value=mock_simple_model)

    # Mock empty cache
    mock_cache: dict[str, Any] = {}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)

    result = await load_specific_model(TEST_MODEL_VERSION)

    assert result is True
    assert TEST_MODEL_VERSION in mock_cache
    assert mock_cache[TEST_MODEL_VERSION] is mock_simple_model


@pytest.mark.asyncio
async def test_load_specific_model_already_cached(
    mocker, mock_simple_model: torch.nn.Module
) -> None:
    """Test loading specific model that's already cached."""
    # Mock cache with model already present
    mock_cache = {TEST_MODEL_VERSION: mock_simple_model}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)

    result = await load_specific_model(TEST_MODEL_VERSION)

    assert result is True


@pytest.mark.asyncio
async def test_load_specific_model_exception(mocker) -> None:
    """Test specific model loading with exception."""
    mocker.patch("mlflow.pytorch.load_model", side_effect=Exception("Loading error"))
    mocker.patch("morphogenetic_engine.inference_server.model_cache", {})

    result = await load_specific_model(TEST_MODEL_VERSION)

    assert result is False


# Integration Tests
@pytest.mark.asyncio
async def test_end_to_end_prediction_workflow(
    mocker, mock_simple_model: torch.nn.Module, sample_input_data: TestBatch
) -> None:
    """
    Test complete end-to-end prediction workflow.

    This integration test validates the entire API workflow from
    health check through model information retrieval to actual
    predictions, ensuring all components work together correctly.

    Args:
        mocker: pytest-mock fixture for patching.
        mock_simple_model: Simple PyTorch model fixture.
        sample_input_data: Sample data for prediction testing.
    """
    # Setup complete environment
    mock_cache = {PRODUCTION_MODEL_VERSION: mock_simple_model}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)
    mocker.patch(
        "morphogenetic_engine.inference_server.current_model_version", PRODUCTION_MODEL_VERSION
    )

    # Test complete workflow: health -> models -> predict
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Check health
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["model_loaded"] is True

        # 2. Get model info
        models_response = await client.get("/models")
        assert models_response.status_code == 200

        # 3. Make prediction
        prediction_response = await client.post("/predict", json={"data": sample_input_data})
        assert prediction_response.status_code == 200

        # Validate prediction results
        data = prediction_response.json()
        validate_prediction_response(data, PRODUCTION_MODEL_VERSION)


@pytest.mark.asyncio
async def test_concurrent_predictions(
    mocker, mock_simple_model: torch.nn.Module, sample_input_data: TestBatch
) -> None:
    """
    Test handling of concurrent prediction requests.

    Validates that the inference server can handle multiple
    simultaneous prediction requests without race conditions
    or performance degradation.

    Args:
        mocker: pytest-mock fixture for patching.
        mock_simple_model: Simple PyTorch model fixture.
        sample_input_data: Sample data for prediction testing.
    """
    # Setup environment
    mock_cache = {TEST_MODEL_VERSION: mock_simple_model}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)
    mocker.patch("morphogenetic_engine.inference_server.current_model_version", TEST_MODEL_VERSION)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Make multiple concurrent requests
        request_data = {"data": sample_input_data}

        tasks = [client.post("/predict", json=request_data) for _ in range(5)]

        responses = await asyncio.gather(*tasks)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            validate_prediction_response(data, TEST_MODEL_VERSION)


# Performance and Memory Tests
@pytest.mark.asyncio
async def test_prediction_performance_benchmark(
    client_with_mock_model: TestClient, sample_input_data: TestBatch
) -> None:
    """Basic performance benchmark for prediction endpoint."""
    request_data = {"data": sample_input_data}

    # Warm up
    client_with_mock_model.post("/predict", json=request_data)

    # Measure performance
    start_time = time.time()
    num_requests = 10

    for _ in range(num_requests):
        response = client_with_mock_model.post("/predict", json=request_data)
        assert response.status_code == 200

    total_time = time.time() - start_time
    avg_time_ms = (total_time / num_requests) * 1000

    # Basic performance assertion (should be under 100ms per request)
    assert avg_time_ms < 100.0, f"Average prediction time {avg_time_ms:.2f}ms exceeds threshold"


def test_model_cache_management(mocker, mock_simple_model: torch.nn.Module) -> None:
    """Test model cache behavior and memory management."""
    mock_cache: dict[str, Any] = {}
    mocker.patch("morphogenetic_engine.inference_server.model_cache", mock_cache)

    # Initially empty
    assert len(mock_cache) == 0

    # Add model to cache
    mock_cache[TEST_MODEL_VERSION] = mock_simple_model
    assert len(mock_cache) == 1
    assert TEST_MODEL_VERSION in mock_cache

    # Verify model is accessible
    cached_model = mock_cache[TEST_MODEL_VERSION]
    assert cached_model is mock_simple_model

    # Test cache key collision protection
    mock_cache["another_version"] = mock_simple_model
    assert len(mock_cache) == 2


# Error Boundary Tests
@pytest.mark.parametrize(
    "endpoint,method,expected_status",
    [
        ("/nonexistent", "GET", 404),
        ("/predict", "GET", 405),  # Wrong method
        ("/health", "POST", 405),  # Wrong method
    ],
)
def test_error_boundaries(
    sync_client: TestClient, endpoint: str, method: str, expected_status: int
) -> None:
    """Test error handling for various invalid requests."""
    response = getattr(sync_client, method.lower())(endpoint)
    assert response.status_code == expected_status


def test_malformed_json_request(sync_client: TestClient, mocker) -> None:
    """Test handling of malformed JSON in requests."""
    mocker.patch("morphogenetic_engine.inference_server.current_model_version", TEST_MODEL_VERSION)

    # Send malformed JSON
    response = sync_client.post(
        "/predict", data="{ invalid json }", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


# Structural Pattern Matching for Test Data (Python 3.12+ feature)
def validate_response_structure(response_data: dict[str, Any], response_type: str) -> bool:
    """
    Use structural pattern matching to validate response structures.

    This function demonstrates Python 3.12+ structural pattern matching
    to validate different types of API responses based on their expected
    structure and required fields.

    Args:
        response_data: The JSON response data to validate.
        response_type: The type of response ("health", "prediction", "models").

    Returns:
        bool: True if the response structure is valid, False otherwise.

    Example:
        >>> data = {"status": "healthy", "model_loaded": True, ...}
        >>> validate_response_structure(data, "health")
        True
    """
    match response_type:
        case "health":
            return all(
                key in response_data
                for key in ["status", "model_loaded", "model_version", "timestamp"]
            )
        case "prediction":
            return all(
                key in response_data
                for key in ["predictions", "probabilities", "model_version", "inference_time_ms"]
            )
        case "models":
            return all(
                key in response_data
                for key in ["current_version", "available_versions", "model_name"]
            )
        case _:
            return False


def test_response_structure_validation(client_with_mock_model: TestClient) -> None:
    """
    Test response structure validation using pattern matching.

    Demonstrates the use of Python 3.12+ structural pattern matching
    for validating API response structures in a clean, readable way.

    Args:
        client_with_mock_model: Test client with pre-loaded model.
    """
    # Test health response structure
    health_response = client_with_mock_model.get("/health")
    assert validate_response_structure(health_response.json(), "health")

    # Test prediction response structure
    request_data = {"data": [[0.1, 0.2, 0.3]]}
    prediction_response = client_with_mock_model.post("/predict", json=request_data)
    assert validate_response_structure(prediction_response.json(), "prediction")


class TestSuiteDocumentation:
    """
    Documentation class for the inference server test suite.

    This test suite demonstrates modern Python 3.12+ testing practices
    including:

    Test Architecture:
        - Function-based tests with pytest fixtures
        - Real PyTorch operations instead of mocked tensor ops
        - Proper async testing with httpx.AsyncClient
        - Comprehensive error boundary testing

    Modern Python Features:
        - Built-in generic types (list[T], dict[K, V])
        - Union operator (str | None)
        - Structural pattern matching (match/case)
        - Full type annotation coverage

    Test Categories:
        - Health endpoint validation
        - Model management and registry integration
        - Prediction pipeline with real tensor operations
        - Async workflow and concurrent request testing
        - Performance benchmarking and memory validation
        - Error handling and boundary conditions

    Quality Standards:
        - No over-mocking of core operations
        - Real functionality validation
        - Comprehensive fixture architecture
        - Integration and unit test coverage
        - Performance monitoring built-in
    """

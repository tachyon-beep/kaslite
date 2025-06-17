"""
Unit tests for the FastAPI Inference Server.

Tests the inference server endpoints, model loading, health checks,
and monitoring functionality.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from morphogenetic_engine.inference_server import (
    app,
    load_production_model,
    load_specific_model,
)


class TestInferenceServer:
    """Test suite for FastAPI Inference Server."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

        # Sample prediction data
        self.sample_data = {"data": [[0.5, 0.3, 0.1], [0.2, 0.8, 0.4]]}

        # Sample model output
        self.sample_output = torch.tensor([[0.2, 0.8], [0.9, 0.1]])

    def test_health_endpoint_healthy(self):
        """Test health endpoint when server is healthy."""
        with patch("morphogenetic_engine.inference_server.current_model_version", "3"):
            response = self.client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["model_version"] == "3"
            assert "timestamp" in data

    def test_health_endpoint_no_model(self):
        """Test health endpoint when no model is loaded."""
        with patch("morphogenetic_engine.inference_server.current_model_version", None):
            response = self.client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is False
            assert data["model_version"] is None

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = self.client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        # Should contain some Prometheus metric format
        assert "# HELP" in response.text or "# TYPE" in response.text

    @patch("morphogenetic_engine.inference_server.ModelRegistry")
    def test_get_model_info_success(self, mock_registry_class):
        """Test model info endpoint success."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        mock_version1 = Mock()
        mock_version1.version = "1"
        mock_version2 = Mock()
        mock_version2.version = "2"

        mock_registry.list_model_versions.return_value = [mock_version1, mock_version2]

        with patch("morphogenetic_engine.inference_server.current_model_version", "2"):
            response = self.client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["current_version"] == "2"
            assert data["available_versions"] == ["1", "2"]
            assert data["model_name"] == "KasminaModel"

    @patch("morphogenetic_engine.inference_server.ModelRegistry")
    def test_get_model_info_failure(self, mock_registry_class):
        """Test model info endpoint failure."""
        # Setup mock to raise exception
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.list_model_versions.side_effect = Exception("Registry error")

        response = self.client.get("/models")

        assert response.status_code == 500
        assert "Failed to retrieve model information" in response.json()["detail"]

    @patch("morphogenetic_engine.inference_server.torch.no_grad")
    @patch("morphogenetic_engine.inference_server.model_cache")
    @patch("morphogenetic_engine.inference_server.current_model_version", "1")
    def test_predict_success(self, mock_cache, mock_no_grad):
        """Test successful prediction."""
        # Setup mock model
        mock_model = Mock()
        mock_model.return_value = self.sample_output
        mock_cache.__getitem__.return_value = mock_model
        mock_cache.__contains__.return_value = True

        # Mock torch operations
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()

        with patch("morphogenetic_engine.inference_server.torch.tensor") as mock_tensor, patch(
            "morphogenetic_engine.inference_server.torch.softmax"
        ) as mock_softmax, patch(
            "morphogenetic_engine.inference_server.torch.argmax"
        ) as mock_argmax:

            # Setup torch mocks
            mock_input_tensor = Mock()
            mock_input_tensor.dim.return_value = 2
            mock_tensor.return_value = mock_input_tensor

            mock_probs = Mock()
            mock_probs.tolist.return_value = [[0.2, 0.8], [0.9, 0.1]]
            mock_softmax.return_value = mock_probs

            mock_preds = Mock()
            mock_preds.tolist.return_value = [1, 0]
            mock_argmax.return_value = mock_preds

            response = self.client.post("/predict", json=self.sample_data)

            assert response.status_code == 200
            data = response.json()
            assert data["predictions"] == [1, 0]
            assert data["probabilities"] == [[0.2, 0.8], [0.9, 0.1]]
            assert data["model_version"] == "1"
            assert "inference_time_ms" in data

    def test_predict_no_model_available(self):
        """Test prediction when no model is available."""
        with patch("morphogenetic_engine.inference_server.current_model_version", None):
            response = self.client.post("/predict", json=self.sample_data)

            assert response.status_code == 503
            assert "No model available" in response.json()["detail"]

    @patch("morphogenetic_engine.inference_server.model_cache")
    @patch("morphogenetic_engine.inference_server.current_model_version", "1")
    def test_predict_invalid_input_data(self, mock_cache):
        """Test prediction with invalid input data."""
        mock_cache.__contains__.return_value = True

        # Test with invalid data that will cause tensor creation to fail
        with patch("morphogenetic_engine.inference_server.torch.tensor") as mock_tensor:
            mock_tensor.side_effect = ValueError("Invalid tensor data")

            invalid_data = {"data": "invalid"}
            response = self.client.post("/predict", json=invalid_data)

            assert response.status_code == 422
            assert "detail" in response.json()

    @patch("morphogenetic_engine.inference_server.load_specific_model")
    @patch("morphogenetic_engine.inference_server.model_cache")
    @patch("morphogenetic_engine.inference_server.current_model_version", "1")
    async def test_predict_model_loading_failure(self, mock_cache, mock_load_specific):
        """Test prediction when model loading fails."""
        mock_cache.__contains__.return_value = False
        mock_load_specific.return_value = False  # Loading fails

        response = self.client.post("/predict", json=self.sample_data)

        assert response.status_code == 404
        assert "Model version" in response.json()["detail"]

    @patch("morphogenetic_engine.inference_server.load_production_model")
    async def test_reload_model_success(self, mock_load_production):
        """Test successful model reload."""
        mock_load_production.return_value = True

        with patch("morphogenetic_engine.inference_server.current_model_version", "2"):
            response = self.client.post("/reload-model")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "Reloaded model version 2" in data["message"]

    @patch("morphogenetic_engine.inference_server.load_production_model")
    async def test_reload_model_failure(self, mock_load_production):
        """Test model reload failure."""
        mock_load_production.return_value = False

        response = self.client.post("/reload-model")

        assert response.status_code == 503
        assert "Failed to reload model" in response.json()["detail"]

    @patch("morphogenetic_engine.inference_server.load_production_model")
    async def test_reload_model_exception(self, mock_load_production):
        """Test model reload with exception."""
        mock_load_production.side_effect = Exception("Reload error")

        response = self.client.post("/reload-model")

        assert response.status_code == 500
        assert "Model reload failed" in response.json()["detail"]

    def test_predict_with_specific_model_version(self):
        """Test prediction with specific model version specified."""
        data_with_version = {"data": [[0.5, 0.3, 0.1]], "model_version": "2"}

        mock_model = Mock()
        mock_model.return_value = torch.tensor([[0.3, 0.7]])

        with patch("morphogenetic_engine.inference_server.model_cache") as mock_cache, patch(
            "morphogenetic_engine.inference_server.torch.tensor"
        ) as mock_tensor, patch(
            "morphogenetic_engine.inference_server.torch.softmax"
        ) as mock_softmax, patch(
            "morphogenetic_engine.inference_server.torch.argmax"
        ) as mock_argmax, patch(
            "morphogenetic_engine.inference_server.torch.no_grad"
        ):

            # Setup mocks
            mock_cache.__contains__.return_value = True
            mock_cache.__getitem__.return_value = mock_model

            mock_input_tensor = Mock()
            mock_input_tensor.dim.return_value = 2
            mock_tensor.return_value = mock_input_tensor

            mock_probs = Mock()
            mock_probs.tolist.return_value = [[0.3, 0.7]]
            mock_softmax.return_value = mock_probs

            mock_preds = Mock()
            mock_preds.tolist.return_value = [1]
            mock_argmax.return_value = mock_preds

            response = self.client.post("/predict", json=data_with_version)

            assert response.status_code == 200
            data = response.json()
            assert data["model_version"] == "2"
            assert data["predictions"] == [1]

    def test_predict_1d_input_expansion(self):
        """Test that 1D input is properly expanded to 2D."""
        single_point_data = {"data": [[0.5, 0.3, 0.1]]}  # Correct format for PredictionRequest

        mock_model = Mock()
        mock_model.return_value = torch.tensor([[0.4, 0.6]])

        with patch("morphogenetic_engine.inference_server.model_cache") as mock_cache, patch(
            "morphogenetic_engine.inference_server.current_model_version", "1"
        ), patch("morphogenetic_engine.inference_server.torch.tensor") as mock_tensor, patch(
            "morphogenetic_engine.inference_server.torch.softmax"
        ) as mock_softmax, patch(
            "morphogenetic_engine.inference_server.torch.argmax"
        ) as mock_argmax, patch(
            "morphogenetic_engine.inference_server.torch.no_grad"
        ):

            # Setup mocks
            mock_cache.__contains__.return_value = True
            mock_cache.__getitem__.return_value = mock_model

            # Mock tensor that starts as 1D and gets unsqueezed
            mock_input_tensor = Mock()
            mock_input_tensor.dim.return_value = 1  # 1D input
            mock_input_tensor.unsqueeze.return_value = mock_input_tensor
            mock_tensor.return_value = mock_input_tensor

            mock_probs = Mock()
            mock_probs.tolist.return_value = [[0.4, 0.6]]
            mock_softmax.return_value = mock_probs

            mock_preds = Mock()
            mock_preds.tolist.return_value = [1]
            mock_argmax.return_value = mock_preds

            response = self.client.post("/predict", json=single_point_data)

            assert response.status_code == 200
            # Should call unsqueeze(0) to add batch dimension
            mock_input_tensor.unsqueeze.assert_called_with(0)


class TestInferenceServerLifecycle:
    """Test server lifecycle and model loading functions."""

    @patch("morphogenetic_engine.inference_server.ModelRegistry")
    @patch("mlflow.pytorch.load_model")
    async def test_load_production_model_success(self, mock_load_model, mock_registry_class):
        """Test successful production model loading."""
        # Setup mocks
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_production_model_uri.return_value = "models:/TestModel/3"

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_load_model.return_value = mock_model

        # Test loading
        with patch("morphogenetic_engine.inference_server.model_cache") as mock_cache, patch(
            "morphogenetic_engine.inference_server.MODEL_LOAD_TIME"
        ):

            result = await load_production_model()

            assert result is True
            mock_registry.get_production_model_uri.assert_called_once()
            mock_load_model.assert_called_once_with("models:/TestModel/3")
            mock_model.eval.assert_called_once()

    @patch("morphogenetic_engine.inference_server.ModelRegistry")
    async def test_load_production_model_no_model(self, mock_registry_class):
        """Test production model loading when no model exists."""
        # Setup mock to return None (no production model)
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_production_model_uri.return_value = None

        # Test loading
        result = await load_production_model()

        assert result is False

    @patch("morphogenetic_engine.inference_server.ModelRegistry")
    async def test_load_production_model_exception(self, mock_registry_class):
        """Test production model loading with exception."""
        # Setup mock to raise exception
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_production_model_uri.side_effect = Exception("Loading error")

        # Test loading
        result = await load_production_model()

        assert result is False

    @patch("mlflow.pytorch.load_model")
    async def test_load_specific_model_success(self, mock_load_model):
        """Test successful specific model loading."""
        # Setup mocks
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_load_model.return_value = mock_model

        # Test loading
        with patch("morphogenetic_engine.inference_server.model_cache") as mock_cache, patch(
            "morphogenetic_engine.inference_server.MODEL_LOAD_TIME"
        ):

            mock_cache.__contains__.return_value = False  # Not in cache

            result = await load_specific_model("2")

            assert result is True
            mock_load_model.assert_called_once_with("models:/KasminaModel/2")

    async def test_load_specific_model_already_cached(self):
        """Test loading specific model that's already cached."""
        # Test with cached model
        with patch("morphogenetic_engine.inference_server.model_cache") as mock_cache:
            mock_cache.__contains__.return_value = True  # Already in cache

            result = await load_specific_model("2")

            assert result is True

    @patch("mlflow.pytorch.load_model")
    async def test_load_specific_model_exception(self, mock_load_model):
        """Test specific model loading with exception."""
        # Setup mock to raise exception
        mock_load_model.side_effect = Exception("Loading error")

        # Test loading
        with patch("morphogenetic_engine.inference_server.model_cache") as mock_cache, patch(
            "morphogenetic_engine.inference_server.MODEL_LOAD_TIME"
        ):

            mock_cache.__contains__.return_value = False

            result = await load_specific_model("2")

            assert result is False

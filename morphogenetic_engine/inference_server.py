"""
FastAPI inference server for morphogenetic models.

This module provides a REST API for serving trained morphogenetic models
with monitoring, health checks, and model management capabilities.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel

from morphogenetic_engine.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "inference_requests_total", "Total inference requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram("inference_request_duration_seconds", "Inference request duration")
MODEL_PREDICTION_TIME = Histogram("model_prediction_duration_seconds", "Model prediction duration")
MODEL_LOAD_TIME = Histogram("model_load_duration_seconds", "Model loading duration")

# Global model storage
model_cache: Dict[str, Any] = {}
current_model_version: Optional[str] = None


class PredictionRequest(BaseModel):
    """Request schema for model predictions."""

    data: List[List[float]]
    model_version: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response schema for model predictions."""

    predictions: List[int]
    probabilities: List[List[float]]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information response schema."""

    current_version: Optional[str]
    available_versions: List[str]
    model_name: str


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting morphogenetic inference server...")
    await load_production_model()
    yield
    # Shutdown
    logger.info("Shutting down morphogenetic inference server...")


app = FastAPI(
    title="Morphogenetic Model Inference API",
    description="REST API for serving trained morphogenetic neural networks",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Middleware to collect Prometheus metrics."""
    start_time = time.time()

    response = await call_next(request)

    # Record metrics
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    return response


async def load_production_model() -> bool:
    """Load the current production model from the registry."""
    global current_model_version  # pylint: disable=global-statement

    try:
        with MODEL_LOAD_TIME.time():
            registry = ModelRegistry()
            model_uri = registry.get_production_model_uri()

            if not model_uri:
                logger.warning("No production model found in registry")
                return False

            # Extract version from URI
            model_version = model_uri.split("/")[-1]

            # Load model using MLflow
            import mlflow.pytorch as mlflow_pytorch

            model = mlflow_pytorch.load_model(model_uri)
            model.eval()

            # Cache the model
            model_cache[model_version] = model
            current_model_version = model_version

            logger.info("Loaded production model version: %s", model_version)
            return True

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to load production model: %s", e)
        return False


async def load_specific_model(version: str) -> bool:
    """Load a specific model version."""
    try:
        if version in model_cache:
            return True

        with MODEL_LOAD_TIME.time():
            model_uri = f"models:/KasminaModel/{version}"

            import mlflow.pytorch as mlflow_pytorch

            model = mlflow_pytorch.load_model(model_uri)
            model.eval()

            model_cache[version] = model
            logger.info("Loaded model version: %s", version)
            return True

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to load model version %s: %s", version, e)
        return False


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=current_model_version is not None,
        model_version=current_model_version,
        timestamp=str(time.time()),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/models", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models."""
    try:
        registry = ModelRegistry()
        versions = registry.list_model_versions()
        available_versions = [v.version for v in versions]

        return ModelInfo(
            current_version=current_model_version,
            available_versions=available_versions,
            model_name="KasminaModel",
        )
    except Exception as e:
        logger.error("Failed to get model info: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve model information") from e


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the loaded model."""
    start_time = time.time()

    try:
        # Determine which model version to use
        model_version = request.model_version or current_model_version

        if not model_version:
            raise HTTPException(status_code=503, detail="No model available")

        # Load model if not cached
        if model_version not in model_cache:
            if not await load_specific_model(model_version):
                raise HTTPException(
                    status_code=404, detail=f"Model version {model_version} not found"
                )

        model = model_cache[model_version]

        # Convert input data to tensor
        try:
            input_tensor = torch.tensor(request.data, dtype=torch.float32)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid input data: {e}") from e

        # Make prediction
        with MODEL_PREDICTION_TIME.time():
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

        # Convert to lists for JSON serialization
        pred_list = predictions.tolist()
        prob_list = probabilities.tolist()

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        return PredictionResponse(
            predictions=pred_list,
            probabilities=prob_list,
            model_version=model_version,
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.post("/reload-model")
async def reload_production_model():
    """Reload the production model from the registry."""
    try:
        success = await load_production_model()
        if success:
            return {
                "status": "success",
                "message": f"Reloaded model version {current_model_version}",
            }
        else:
            raise HTTPException(status_code=503, detail="Failed to reload model")
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error("Model reload failed: %s", e)
        raise HTTPException(status_code=500, detail="Model reload failed") from e


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """Global exception handler for better error reporting."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "morphogenetic_engine.inference_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )

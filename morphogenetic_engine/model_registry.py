"""
MLflow Model Registry integration for morphogenetic engine.

This module provides functionality for registering models, managing versions,
and tracking model metadata in MLflow's Model Registry.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import mlflow
import mlflow.exceptions
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model registration and versioning with MLflow Model Registry."""

    def __init__(self, model_name: str = "KasminaModel"):
        """Initialize the model registry manager.

        Args:
            model_name: Base name for registered models
        """
        self.model_name = model_name
        self.client = MlflowClient()

    def register_best_model(
        self,
        run_id: str,
        metrics: Dict[str, float],
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[ModelVersion]:
        """Register a model from a completed run.

        Args:
            run_id: MLflow run ID containing the model
            metrics: Final metrics from the run
            model_name: Override default model name
            description: Model version description
            tags: Additional tags for the model version

        Returns:
            ModelVersion object if successful, None otherwise
        """
        try:
            model_name = model_name or self.model_name
            model_uri = f"runs:/{run_id}/model"

            # Create description with key metrics
            if not description:
                val_acc = metrics.get("val_acc", 0.0)
                train_loss = metrics.get("train_loss", 0.0)
                seeds_activated = metrics.get("seeds_activated", False)
                description = (
                    f"Morphogenetic model - Val Acc: {val_acc:.4f}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Seeds Activated: {seeds_activated}"
                )

            # Register the model
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)

            # Update version description
            self.client.update_model_version(
                name=model_name, version=model_version.version, description=description
            )

            logger.info(
                "Registered model %s version %s from run %s",
                model_name,
                model_version.version,
                run_id,
            )

            return model_version

        except mlflow.exceptions.MlflowException as e:
            logger.error("Failed to register model: %s", e)
            return None
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected error registering model: %s", e)
            return None

    def get_best_model_version(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
        metric_name: str = "val_acc",
        higher_is_better: bool = True,
    ) -> Optional[ModelVersion]:
        """Find the best model version based on a metric.

        Args:
            model_name: Model name to search
            stage: Filter by stage ("Staging", "Production", etc.)
            metric_name: Metric to optimize
            higher_is_better: Whether higher metric values are better

        Returns:
            Best ModelVersion or None if not found
        """
        try:
            model_name = model_name or self.model_name
            versions = self._get_filtered_versions(model_name, stage)

            if not versions:
                logger.warning("No model versions found for %s", model_name)
                return None

            best_version, best_metric = self._find_best_version(
                versions, metric_name, higher_is_better
            )

            if best_version:
                logger.info(
                    "Best model version: %s with %s=%.4f",
                    best_version.version,
                    metric_name,
                    best_metric,
                )

            return best_version

        except mlflow.exceptions.MlflowException as e:
            logger.error("Failed to find best model: %s", e)
            return None

    def _get_filtered_versions(self, model_name: str, stage: Optional[str]) -> List[ModelVersion]:
        """Get model versions filtered by stage."""
        versions = self.client.search_model_versions(filter_string=f"name='{model_name}'")

        if stage:
            versions = [v for v in versions if v.current_stage == stage]

        return versions

    def _find_best_version(
        self, versions: List[ModelVersion], metric_name: str, higher_is_better: bool
    ):
        """Find the best version from a list based on a metric."""
        best_version = None
        best_metric = float("-inf") if higher_is_better else float("inf")

        for version in versions:
            if not version.run_id:
                continue

            try:
                metric_value = self._get_metric_value(version.run_id, metric_name)

                if self._is_better_metric(metric_value, best_metric, higher_is_better):
                    best_metric = metric_value
                    best_version = version

            except (ValueError, KeyError):
                continue

        return best_version, best_metric

    def _get_metric_value(self, run_id: str, metric_name: str) -> float:
        """Get metric value from a run."""
        run = self.client.get_run(run_id)
        return float(run.data.metrics.get(metric_name, 0))

    def _is_better_metric(self, current: float, best: float, higher_is_better: bool) -> bool:
        """Check if current metric is better than the best so far."""
        return (higher_is_better and current > best) or (not higher_is_better and current < best)

    def promote_model(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        stage: str = "Staging",
        archive_existing: bool = True,
    ) -> bool:
        """Promote a model version to a specific stage.

        Args:
            model_name: Model name
            version: Model version to promote
            stage: Target stage ("Staging", "Production", etc.)
            archive_existing: Whether to archive existing models in the stage

        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = model_name or self.model_name

            # If no version specified, get the best one
            if not version:
                best_version = self.get_best_model_version(model_name)
                if not best_version:
                    logger.error("No best model version found for promotion")
                    return False
                version = best_version.version

            # Archive existing models in the target stage
            if archive_existing:
                existing_versions = self.client.search_model_versions(
                    filter_string=f"name='{model_name}'"
                )
                for v in existing_versions:
                    if v.current_stage == stage:
                        self.client.transition_model_version_stage(
                            name=model_name, version=v.version, stage="Archived"
                        )

            # Promote the new version
            self.client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )

            logger.info("Promoted model %s v%s to %s", model_name, version, stage)
            return True

        except mlflow.exceptions.MlflowException as e:
            logger.error("Failed to promote model: %s", e)
            return False

    def list_model_versions(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> List[ModelVersion]:
        """List model versions with optional filtering.

        Args:
            model_name: Model name to filter by
            stage: Stage to filter by

        Returns:
            List of ModelVersion objects
        """
        try:
            model_name = model_name or self.model_name
            filter_string = f"name='{model_name}'"

            versions = self.client.search_model_versions(filter_string=filter_string)

            if stage:
                versions = [v for v in versions if v.current_stage == stage]

            return sorted(versions, key=lambda v: int(v.version), reverse=True)

        except mlflow.exceptions.MlflowException as e:
            logger.error("Failed to list model versions: %s", e)
            return []

    def get_production_model_uri(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get the URI of the current production model.

        Args:
            model_name: Model name

        Returns:
            Model URI or None if no production model exists
        """
        try:
            model_name = model_name or self.model_name
            versions = self.list_model_versions(model_name, stage="Production")

            if versions:
                latest_prod = versions[0]  # Already sorted by version desc
                return f"models:/{model_name}/{latest_prod.version}"

            return None

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to get production model URI: %s", e)
            return None

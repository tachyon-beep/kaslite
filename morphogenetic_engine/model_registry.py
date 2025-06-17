"""
MLflow Model Registry integration for morphogenetic engine.

This module provides functionality for registering models, managing versions,
and tracking model metadata in MLflow's Model Registry.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.exceptions
import mlflow.pytorch
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
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
                f"Registered model {model_name} version {model_version.version} "
                f"from run {run_id}"
            )

            return model_version

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to register model: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error registering model: {e}")
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

            # Get all model versions
            versions = self.client.search_model_versions(filter_string=f"name='{model_name}'")

            if stage:
                versions = [v for v in versions if v.current_stage == stage]

            if not versions:
                logger.warning(f"No model versions found for {model_name}")
                return None

            # Find best version by metric
            best_version = None
            best_metric = float("-inf") if higher_is_better else float("inf")

            for version in versions:
                if not version.run_id:
                    continue

                try:
                    # Get metrics from the run
                    run = self.client.get_run(version.run_id)
                    metric_value = float(run.data.metrics.get(metric_name, 0))

                    is_better = (higher_is_better and metric_value > best_metric) or (
                        not higher_is_better and metric_value < best_metric
                    )

                    if is_better:
                        best_metric = metric_value
                        best_version = version

                except (ValueError, KeyError):
                    continue

            if best_version:
                logger.info(
                    f"Best model version: {best_version.version} "
                    f"with {metric_name}={best_metric:.4f}"
                )

            return best_version

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to find best model: {e}")
            return None

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

            logger.info(f"Promoted model {model_name} v{version} to {stage}")
            return True

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to promote model: {e}")
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
            logger.error(f"Failed to list model versions: {e}")
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

        except Exception as e:
            logger.error(f"Failed to get production model URI: {e}")
            return None

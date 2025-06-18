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
                seeds_activated = bool(metrics.get("seeds_activated", False))
                description = (
                    f"Morphogenetic model - Val Acc: {val_acc:.4f}, "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Seeds Activated: {seeds_activated}"
                )

            # Register the model
            model_version = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)

            # Update version description (non-critical, don't fail if this doesn't work)
            try:
                self.client.update_model_version(
                    name=model_name, version=model_version.version, description=description
                )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Failed to update model description: %s", e)
                # Continue anyway - the model registration succeeded

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
        """Get model versions filtered by alias."""
        versions = self.client.search_model_versions(filter_string=f"name='{model_name}'")

        if stage:
            # Filter by alias instead of deprecated stage
            filtered_versions = []
            for v in versions:
                try:
                    # Handle both real aliases (list) and mock objects
                    aliases = getattr(v, "aliases", [])
                    if hasattr(aliases, "__contains__") and stage in aliases:
                        filtered_versions.append(v)
                except (TypeError, AttributeError):
                    # Skip if we can't check aliases (e.g., for mocks)
                    continue
            versions = filtered_versions

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

    def _find_existing_version(self, model_name: str, stage: str) -> Optional[ModelVersion]:
        """Find existing version with the given stage/alias."""
        logger.debug("Looking for existing version with stage/alias: %s", stage)

        # Try to find existing version with this alias first
        try:
            existing_version = self.client.get_model_version_by_alias(name=model_name, alias=stage)
            if existing_version:
                logger.debug("Found version via alias: %s", existing_version.version)
                return existing_version
        except mlflow.exceptions.MlflowException:
            logger.debug("No version found via alias for stage: %s", stage)

        # Fallback: search for versions and check aliases/stages
        logger.debug("Searching for versions with stage/alias: %s", stage)
        existing_versions = self.client.search_model_versions(filter_string=f"name='{model_name}'")
        logger.debug("Found %d total versions", len(existing_versions))

        for v in existing_versions:
            logger.debug("Checking version %s: aliases=%s", v.version, getattr(v, "aliases", "N/A"))

            try:
                aliases = getattr(v, "aliases", [])
                if hasattr(aliases, "__contains__") and stage in aliases:
                    logger.debug("Found version %s via aliases", v.version)
                    return v
            except (TypeError, AttributeError):
                pass

        logger.debug("No existing version found for stage: %s", stage)
        return None

    def _archive_existing_version(self, model_name: str, stage: str) -> None:
        """Archive existing version with the given stage/alias."""
        existing_version = self._find_existing_version(model_name, stage)

        if existing_version:
            logger.info(
                "Found existing version %s for stage %s, removing alias...",
                existing_version.version,
                stage,
            )

            # Remove the existing alias (modern alias-based approach)
            try:
                self.client.delete_registered_model_alias(name=model_name, alias=stage)
                logger.info("Deleted existing alias %s", stage)
            except mlflow.exceptions.RestException:
                # Alias doesn't exist, which is fine
                logger.info("No existing alias %s to delete", stage)
        else:
            logger.info("No existing version found for stage %s, skipping archiving", stage)

    def promote_model(
        self,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        stage: str = "Staging",
        archive_existing: bool = True,
    ) -> bool:
        """Promote a model version to a specific stage using modern MLflow aliases.

        Args:
            model_name: Model name
            version: Model version to promote
            stage: Target stage/alias ("Staging", "Production", etc.)
            archive_existing: Whether to archive existing versions and remove aliases

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
                version = str(best_version.version)

            # Convert version to string to ensure consistency
            version_str = str(version)

            # Handle archiving of existing versions if requested
            if archive_existing:
                self._archive_existing_version(model_name, stage)

            # Set the new alias (modern alias-based API)
            logger.debug(
                "Setting alias %s for model %s version %s (type: %s)",
                stage,
                model_name,
                version_str,
                type(version_str),
            )
            try:
                # Try with string version first
                self.client.set_registered_model_alias(
                    name=model_name, alias=stage, version=version_str
                )
                logger.debug(
                    "Successfully set alias %s for model %s version %s (string)",
                    stage,
                    model_name,
                    version_str,
                )
            except (mlflow.exceptions.MlflowException, mlflow.exceptions.RestException) as e:
                logger.debug("Failed with string version %s: %s. Trying integer...", version_str, e)
                # Try with integer version
                self.client.set_registered_model_alias(
                    name=model_name, alias=stage, version=int(version_str)
                )
                logger.debug(
                    "Successfully set alias %s for model %s version %s (integer)",
                    stage,
                    model_name,
                    version_str,
                )

            logger.info("Promoted model %s v%s to %s", model_name, version_str, stage)
            return True

        except mlflow.exceptions.MlflowException as e:
            logger.error("Failed to promote model: %s", e)
            return False
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected error promoting model: %s", e)
            return False

    def list_model_versions(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> List[ModelVersion]:
        """List model versions with optional filtering by alias.

        Args:
            model_name: Model name to filter by
            stage: Alias to filter by (replaces the old stage concept)

        Returns:
            List of ModelVersion objects
        """
        try:
            model_name = model_name or self.model_name
            filter_string = f"name='{model_name}'"

            versions = self.client.search_model_versions(filter_string=filter_string)

            if stage:
                # Filter by alias instead of deprecated stage
                filtered_versions = []
                for v in versions:
                    try:
                        # Handle both real aliases (list) and mock objects
                        aliases = getattr(v, "aliases", [])
                        if hasattr(aliases, "__contains__") and stage in aliases:
                            filtered_versions.append(v)
                    except (TypeError, AttributeError):
                        # Skip if we can't check aliases (e.g., for mocks)
                        continue
                versions = filtered_versions

            return sorted(versions, key=lambda v: int(v.version), reverse=True)

        except mlflow.exceptions.MlflowException as e:
            logger.error("Failed to list model versions: %s", e)
            return []

    def get_production_model_uri(self, model_name: Optional[str] = None) -> Optional[str]:
        """Get the URI of the current production model using aliases.

        Args:
            model_name: Model name

        Returns:
            Model URI or None if no production model exists
        """
        try:
            model_name = model_name or self.model_name

            # Get model version by alias instead of deprecated stage
            try:
                model_version = self.client.get_model_version_by_alias(
                    name=model_name, alias="Production"
                )
                return f"models:/{model_name}/{model_version.version}"
            except mlflow.exceptions.RestException:
                # No Production alias exists
                return None

        except mlflow.exceptions.MlflowException as e:
            logger.error("Failed to get production model URI: %s", e)
            return None

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to get production model URI: %s", e)
            return None

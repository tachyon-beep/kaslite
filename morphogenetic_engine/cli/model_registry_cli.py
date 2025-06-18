#!/usr/bin/env python3
"""
CLI tool for managing morphogenetic model registry operations.

This script provides command-line utilities for registering models,
promoting them between stages, and managing the model lifecycle.
"""

import argparse
import datetime
import logging
import sys

from mlflow.tracking import MlflowClient

from morphogenetic_engine.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def register_model(args):
    """Register a model from an MLflow run."""
    registry = ModelRegistry(args.model_name)

    # Prepare tags
    tags = {}
    if args.tags:
        for tag in args.tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                tags[key] = value

    # Prepare metrics (for description)
    metrics = {}
    if args.val_acc is not None:
        metrics["val_acc"] = args.val_acc
    if args.train_loss is not None:
        metrics["train_loss"] = args.train_loss
    if args.seeds_activated is not None:
        metrics["seeds_activated"] = args.seeds_activated

    model_version = registry.register_best_model(
        run_id=args.run_id, metrics=metrics, description=args.description, tags=tags
    )

    if model_version:
        print(f"✅ Successfully registered model version {model_version.version}")
        print(f"   Run ID: {args.run_id}")
        print(f"   Model Name: {args.model_name}")
        if metrics:
            print(f"   Metrics: {metrics}")
    else:
        print("❌ Failed to register model")
        sys.exit(1)


def promote_model(args):
    """Promote a model to a specific stage."""
    registry = ModelRegistry(args.model_name)

    success = registry.promote_model(
        version=args.version, stage=args.stage, archive_existing=args.archive_existing
    )

    if success:
        print(f"✅ Successfully promoted model {args.model_name} v{args.version} to {args.stage}")
    else:
        print("❌ Failed to promote model")
        sys.exit(1)


def list_models(args):
    """List model versions."""
    registry = ModelRegistry(args.model_name)

    versions = registry.list_model_versions(stage=args.stage)

    if not versions:
        print(f"No model versions found for {args.model_name}")
        if args.stage:
            print(f"(filtered by stage: {args.stage})")
        return

    print(f"\nModel versions for {args.model_name}:")
    print("-" * 80)
    print(f"{'Version':<10} {'Stage':<12} {'Run ID':<32} {'Created':<20}")
    print("-" * 80)

    for version in versions:
        created_time = version.creation_timestamp
        if created_time:
            created_str = datetime.datetime.fromtimestamp(created_time / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            created_str = "Unknown"

        print(
            f"{version.version:<10} {version.current_stage:<12} {version.run_id:<32} {created_str:<20}"
        )


def get_best_model(args):
    """Find the best model version based on a metric."""
    registry = ModelRegistry(args.model_name)

    best_version = registry.get_best_model_version(
        stage=args.stage, metric_name=args.metric, higher_is_better=args.higher_is_better
    )

    if best_version:
        print(f"✅ Best model version: {best_version.version}")
        print(f"   Stage: {best_version.current_stage}")
        print(f"   Run ID: {best_version.run_id}")
        print(f"   Metric: {args.metric}")

        # Get the metric value from the run
        try:
            client = MlflowClient()
            if best_version.run_id:
                run = client.get_run(best_version.run_id)
                metric_value = run.data.metrics.get(args.metric, "N/A")
                print(f"   {args.metric}: {metric_value}")
            else:
                print(f"   {args.metric}: N/A (no run ID)")
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Could not retrieve metric value: %s", e)
    else:
        print("❌ No model versions found matching criteria")


def get_production_model(args):
    """Get the current production model URI."""
    registry = ModelRegistry(args.model_name)

    model_uri = registry.get_production_model_uri()

    if model_uri:
        print(f"✅ Production model URI: {model_uri}")
    else:
        print(f"❌ No production model found for {args.model_name}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Morphogenetic Model Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-name", default="KasminaModel", help="Model name (default: KasminaModel)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register a model from MLflow run")
    register_parser.add_argument("run_id", help="MLflow run ID")
    register_parser.add_argument("--description", help="Model description")
    register_parser.add_argument("--val-acc", type=float, help="Validation accuracy")
    register_parser.add_argument("--train-loss", type=float, help="Training loss")
    register_parser.add_argument(
        "--seeds-activated", type=bool, help="Whether seeds were activated"
    )
    register_parser.add_argument("--tags", nargs="*", help="Tags in key=value format")

    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote model to stage")
    promote_parser.add_argument(
        "stage", choices=["Staging", "Production", "Archived"], help="Target stage"
    )
    promote_parser.add_argument("--version", help="Model version (defaults to best)")
    promote_parser.add_argument(
        "--no-archive",
        dest="archive_existing",
        action="store_false",
        help="Do not archive existing models in target stage",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List model versions")
    list_parser.add_argument("--stage", help="Filter by stage")

    # Best command
    best_parser = subparsers.add_parser("best", help="Find best model by metric")
    best_parser.add_argument(
        "--metric", default="val_acc", help="Metric to optimize (default: val_acc)"
    )
    best_parser.add_argument("--stage", help="Filter by stage")
    best_parser.add_argument(
        "--lower-is-better",
        dest="higher_is_better",
        action="store_false",
        help="Lower metric values are better",
    )

    # Production command
    subparsers.add_parser("production", help="Get production model URI")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "register":
            register_model(args)
        elif args.command == "promote":
            promote_model(args)
        elif args.command == "list":
            list_models(args)
        elif args.command == "best":
            get_best_model(args)
        elif args.command == "production":
            get_production_model(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Command failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

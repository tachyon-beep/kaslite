"""
Main experiment runner for morphogenetic architecture experiments.

This module provides the core experiment execution logic, including
experiment setup, data loading, training phases, and result collection.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from morphogenetic_engine import datasets
from morphogenetic_engine.cli_dashboard import RichDashboard
from morphogenetic_engine.experiment import build_model_and_agents
from morphogenetic_engine.logger import ExperimentLogger
from morphogenetic_engine.events import SeedState
from morphogenetic_engine.monitoring import cleanup_monitoring, initialize_monitoring
from morphogenetic_engine.training import execute_phase_1, execute_phase_2
from morphogenetic_engine.utils import (
    create_experiment_config,
    export_metrics_for_dvc,
    generate_experiment_slug,
)


def setup_experiment(args):
    """Configure the experiment environment based on the parsed arguments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Set device based on args and availability
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # Create configuration
    config = create_experiment_config(args, device)
    slug = generate_experiment_slug(args)

    # Initialize Prometheus monitoring
    initialize_monitoring(experiment_id=slug, port=8000)

    # Determine log location and initialise logger
    project_root = Path(__file__).parent.parent

    # Configure MLflow
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config.get("mlflow_uri", "file://" + str(mlruns_dir)))
    mlflow.set_experiment(config.get("experiment_name", args.problem_type))

    log_dir = project_root / "results"
    log_filename = f"results_{slug}.log"
    logger = ExperimentLogger(log_dir=log_dir, log_file=log_filename)

    # Create TensorBoard writer
    tb_dir = project_root / "runs" / slug
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    return logger, tb_writer, device, config, slug, project_root


def get_dataloaders(args):
    """Generate or load the specified dataset and create DataLoader instances."""
    # Dispatch on problem type to call appropriate generator
    if args.problem_type == "spirals":
        X, y = datasets.create_spirals(
            n_samples=args.n_samples,
            noise=args.noise,
            rotations=args.rotations,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "moons":
        X, y = datasets.create_moons(
            n_samples=args.n_samples,
            moon_noise=args.moon_noise,
            moon_sep=args.moon_sep,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "complex_moons":
        X, y = datasets.create_complex_moons(
            n_samples=args.n_samples,
            noise=args.moon_noise,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "clusters":
        X, y = datasets.create_clusters(
            cluster_count=args.cluster_count,
            cluster_size=args.n_samples // args.cluster_count,
            cluster_std=args.cluster_std,
            cluster_sep=args.cluster_sep,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "spheres":
        X, y = datasets.create_spheres(
            sphere_count=args.sphere_count,
            sphere_size=args.n_samples // args.sphere_count,
            sphere_radii=args.sphere_radii,
            sphere_noise=args.sphere_noise,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "cifar10":
        # Load CIFAR-10 dataset - use full training set
        X, y = datasets.create_cifar10(data_dir="data/cifar", train=True)
        # Override input_dim for CIFAR-10 (32*32*3 = 3072)
        args.input_dim = 3072
    else:
        raise ValueError(f"Unknown problem type: {args.problem_type}")

    # Apply normalization for synthetic datasets only
    # CIFAR-10 is already normalized to [0,1] in create_cifar10
    if args.problem_type != "cifar10":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X)
        X = scaler.transform(X).astype(np.float32)
    else:
        # Ensure X is float32 for consistency
        X = X.astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_size = int(args.train_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=0)

    return train_loader, val_loader, X.shape[0], X.shape[1:]


def log_final_summary(logger: ExperimentLogger, final_stats: dict[str, Any]):
    """Logs the final summary and system shutdown event."""
    if final_stats.get("acc_pre") is not None and final_stats.get("acc_post") is not None:
        logging.info(
            f"Final Accuracy: Pre-Adaptation {final_stats['acc_pre']:.4f}, "
            f"Post-Adaptation {final_stats['acc_post']:.4f}"
        )
    logger.log_system_shutdown(final_stats=final_stats)


def setup_mlflow_logging(config: Dict[str, Any], slug: str) -> None:
    """Setup MLflow run and log parameters."""
    try:
        # End any existing run first to avoid conflicts
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.start_run(run_name=slug)
        mlflow.log_params(config)
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        print(f"Warning: MLflow initialization failed: {e}")


def log_mlflow_metrics_and_artifacts(final_stats: Dict[str, Any], model, seed_manager, project_root: Path, slug: str) -> None:
    """Log metrics, artifacts, and model to MLflow."""
    try:
        # Log metrics
        if (best_acc := final_stats.get("best_acc")) is not None:
            mlflow.log_metric("final_best_acc", best_acc)
        if (accuracy_dip := final_stats.get("accuracy_dip")) is not None:
            mlflow.log_metric("accuracy_dip", accuracy_dip)
        if (recovery_time := final_stats.get("recovery_time")) is not None:
            mlflow.log_metric("recovery_time", recovery_time)

        mlflow.log_metric("total_seeds", len(seed_manager.seeds))
        active_seeds_count = sum(1 for info in seed_manager.seeds.values() if info["module"].state == SeedState.ACTIVE)
        mlflow.log_metric("active_seeds", active_seeds_count)
        mlflow.set_tag("seeds_activated", str(final_stats.get("seeds_activated", False)))

        # Log artifacts
        log_path = project_root / "results" / f"results_{slug}.log"
        if log_path.exists():
            mlflow.log_artifact(str(log_path))

        # Log TensorBoard logs
        tb_dir = project_root / "runs" / slug
        if tb_dir.exists():
            mlflow.log_artifacts(str(tb_dir), artifact_path="tensorboard")

        # Log and register model
        try:
            serializable_model = _create_serializable_model(model)
            mlflow.pytorch.log_model(serializable_model, "model")
        except RuntimeError as e:
            print(f"Warning: MLflow model logging failed: {e}")

        # Register model in Model Registry if validation accuracy meets threshold
        if final_stats.get("val_acc", 0) >= 0.7:
            print("Model validation accuracy exceeds threshold, but registry is disabled.")

    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        print(f"Warning: MLflow logging failed: {e}")


def _create_serializable_model(original_model):
    """Create a serializable version of BaseNet for MLflow logging."""

    class SerializableBaseNet(nn.Module):
        """Simplified, serializable version of BaseNet for model persistence."""

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers

            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.backbone = nn.Sequential(*layers)

        def forward(self, x):
            return self.backbone(x)

    serializable = SerializableBaseNet(
        input_dim=original_model.input_dim,
        hidden_dim=original_model.hidden_dim,
        output_dim=original_model.output_dim,
        num_layers=original_model.num_layers,
    )

    serializable_state = serializable.state_dict()
    original_state = original_model.state_dict()

    for key in serializable_state.keys():
        if key in original_state:
            serializable_state[key] = original_state[key]

    serializable.load_state_dict(serializable_state)
    serializable.eval()

    return serializable


def run_single_experiment(args) -> Dict[str, Any]:
    """Run a single experiment with the given arguments and return results."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger, tb_writer, device, config, slug, project_root = setup_experiment(args)
    logger.log_system_init(config=config)
    setup_mlflow_logging(config, slug)

    # Calculate total epochs and pass to the dashboard
    total_experiment_epochs = args.warm_up_epochs + args.adaptation_epochs
    dashboard_params = vars(args).copy()
    dashboard_params["epochs"] = total_experiment_epochs
    dashboard = RichDashboard(experiment_params=dashboard_params)

    # Connect the dashboard to the logger for real-time dispatching
    logger.dashboard = dashboard

    # Start the dashboard UI
    dashboard.start()

    # Initialize variables to ensure they exist for the `finally` block
    model, seed_manager, final_stats = None, None, {}

    try:
        train_loader, val_loader, n_samples, input_shape = get_dataloaders(args)
        config["n_samples"] = n_samples
        config["input_shape"] = input_shape

        model, seed_manager, karn, tamiyo = build_model_and_agents(
            logger=logger,
            tb_writer=tb_writer,
            dashboard=dashboard,
            **config,
        )

        # --- Phase 1: Warm-up ---
        final_stats = execute_phase_1(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            tb_writer=tb_writer,
            device=device,
            config=config,
            dashboard=dashboard,
            seed_manager=seed_manager,
        )

        # --- Phase 2: Adaptation ---
        if args.adaptation_epochs > 0:
            final_stats = execute_phase_2(
                model=model,
                seed_manager=seed_manager,
                karn=karn,
                tamiyo=tamiyo,
                train_loader=train_loader,
                val_loader=val_loader,
                logger=logger,
                tb_writer=tb_writer,
                device=device,
                config=config,
                dashboard=dashboard,
                final_stats=final_stats,
            )

    except (RuntimeError, ValueError, KeyError, TypeError) as e:
        logging.error(f"Experiment failed: {e}")
        # Return a result indicating failure
        return {"status": "failed", "error": str(e), **final_stats}

    finally:
        log_final_summary(logger, final_stats)
        if model and seed_manager:
            log_mlflow_metrics_and_artifacts(
                final_stats, model, seed_manager, project_root, slug
            )
        # Stop the live dashboard
        dashboard.stop()
        tb_writer.close()
        cleanup_monitoring()
        if mlflow.active_run():
            mlflow.end_run()

    # Export final metrics for DVC
    export_metrics_for_dvc(final_stats, slug, project_root)

    logging.info(f"Experiment {slug} completed.")
    return {**final_stats, "status": "completed"}

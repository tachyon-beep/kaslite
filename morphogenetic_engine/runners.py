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
from morphogenetic_engine.monitoring import cleanup_monitoring, initialize_monitoring
from morphogenetic_engine.training import execute_phase_1, execute_phase_2
from morphogenetic_engine.utils import (
    create_experiment_config,
    export_metrics_for_dvc,
    generate_experiment_slug,
    write_experiment_log_footer,
    write_experiment_log_header,
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
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"results_{slug}.log"

    logger = ExperimentLogger(str(log_path), config)
    log_f = log_path.open("w", encoding="utf-8")

    # Create TensorBoard writer
    tb_dir = project_root / "runs" / slug
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    return logger, tb_writer, log_f, device, config, slug, project_root


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

    return train_loader, val_loader


def log_final_summary(logger, final_stats, seed_manager, log_f):
    """Print the final report to the console and write the summary footer to the log file."""
    if final_stats["acc_pre"] is not None and final_stats["acc_post"] is not None:
        logger.log_accuracy_dip(0, final_stats["accuracy_dip"])

    write_experiment_log_footer(log_f, final_stats, seed_manager)


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
        mlflow.log_metric("final_best_acc", final_stats["best_acc"])
        if "accuracy_dip" in final_stats and final_stats["accuracy_dip"] is not None:
            mlflow.log_metric("accuracy_dip", final_stats["accuracy_dip"])
        if "recovery_time" in final_stats and final_stats["recovery_time"] is not None:
            mlflow.log_metric("recovery_time", final_stats["recovery_time"])

        mlflow.log_metric("total_seeds", len(seed_manager.seeds))
        active_seeds_count = sum(1 for info in seed_manager.seeds.values() if info["module"].state == "active")
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


def run_single_experiment(args, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Run a single experiment with the given arguments and return results."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger, tb_writer, log_f, device, config, slug, project_root = setup_experiment(args)
    setup_mlflow_logging(config, slug)

    # Calculate total epochs and pass to the dashboard
    total_experiment_epochs = args.warm_up_epochs + args.adaptation_epochs
    dashboard_params = vars(args).copy()
    dashboard_params["epochs"] = total_experiment_epochs
    dashboard = RichDashboard(experiment_params=dashboard_params)
    logger.dashboard = dashboard

    try:
        with log_f, dashboard:
            write_experiment_log_header(log_f, config, args)
            logger.log_experiment_start()

            train_loader, val_loader = get_dataloaders(args)
            loaders = (train_loader, val_loader)
            loss_fn = nn.CrossEntropyLoss()

            model, seed_manager, _, _ = build_model_and_agents(args, device)

            for sid in seed_manager.seeds:
                logger.log_seed_event(epoch=0, seed_id=sid, from_state="init", to_state="dormant")

            logger.log_phase_transition(
                epoch=0,
                from_phase="init",
                to_phase="phase_1",
                description="Warm-up",
                total_epochs=args.warm_up_epochs,
            )

            best_acc_phase1 = execute_phase_1(config, model, loaders, loss_fn, seed_manager, logger, tb_writer, log_f)

            logger.log_phase_transition(
                epoch=args.warm_up_epochs,
                from_phase="phase_1",
                to_phase="phase_2",
                description="Adaptation",
                total_epochs=args.adaptation_epochs,
            )

            final_stats = execute_phase_2(
                config, model, loaders, loss_fn, seed_manager, seed_manager, logger, tb_writer, log_f, best_acc_phase1
            )

            log_final_summary(logger, final_stats, seed_manager, log_f)
            log_mlflow_metrics_and_artifacts(final_stats, model, seed_manager, project_root, slug)
            export_metrics_for_dvc(final_stats, slug, project_root)

            print(f"Final best accuracy: {final_stats['best_acc']:.4f}")
            if final_stats.get("seeds_activated"):
                print("Seeds were activated during this experiment.")

            return {
                "run_id": run_id,
                "best_acc": final_stats["best_acc"],
                "seeds_activated": final_stats.get("seeds_activated", False),
            }

    except (RuntimeError, ValueError, KeyError) as e:
        import traceback

        traceback.print_exc()
        cleanup_monitoring()
        if mlflow.active_run():
            mlflow.end_run("FAILED")
        tb_writer.close()
        dashboard.stop()
        print(f"Experiment failed: {e}")
        return {"run_id": run_id, "error": str(e), "best_acc": 0.0, "seeds_activated": False}

"""
Main experiment runner for morphogenetic architecture experiments.

This module provides the core experiment execution logic, including
experiment setup, data loading, training phases, and result collection.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from morphogenetic_engine import datasets
from morphogenetic_engine.cli_dashboard import RichDashboard
from morphogenetic_engine.experiment import build_model_and_agents
from morphogenetic_engine.logger import ExperimentLogger
from morphogenetic_engine.training import execute_phase_1, execute_phase_2
from morphogenetic_engine.utils import (
    create_experiment_config,
    export_metrics_for_dvc,
    generate_experiment_slug,
    is_testing_mode,
    write_experiment_log_footer,
    write_experiment_log_header,
)

# MLflow integration - conditional import
TESTING_MODE = is_testing_mode()
MLFLOW_AVAILABLE = not TESTING_MODE

# Initialize MLflow variables
mlflow = None
mlflow_pytorch = None
ModelRegistry = None

if MLFLOW_AVAILABLE:
    try:
        import mlflow
        import mlflow.pytorch as mlflow_pytorch
        from morphogenetic_engine.model_registry import ModelRegistry
    except ImportError:
        MLFLOW_AVAILABLE = False
        mlflow = None
        mlflow_pytorch = None
        ModelRegistry = None


def setup_experiment(args):
    """Configure the experiment environment based on the parsed arguments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Set device based on args and availability
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # Create configuration
    config = create_experiment_config(args, device)
    slug = generate_experiment_slug(args)

    # Initialize Prometheus monitoring
    from morphogenetic_engine.monitoring import initialize_monitoring
    initialize_monitoring(experiment_id=slug, port=8000)

    # Determine log location and initialise logger
    project_root = Path(__file__).parent.parent

    # Configure MLflow if available
    if MLFLOW_AVAILABLE and mlflow is not None:
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
    else:
        raise ValueError(f"Unknown problem type: {args.problem_type}")

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype(np.float32)

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
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
        
    try:
        # End any existing run first to avoid conflicts
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.start_run(run_name=slug)
        mlflow.log_params(config)
    except (ImportError, AttributeError, RuntimeError, ValueError) as e:
        print(f"Warning: MLflow initialization failed: {e}")


def log_mlflow_metrics_and_artifacts(final_stats: Dict[str, Any], model, seed_manager, 
                                    project_root: Path, slug: str, args) -> None:
    """Log metrics, artifacts, and model to MLflow."""
    if not MLFLOW_AVAILABLE or mlflow is None or mlflow_pytorch is None:
        return
        
    try:
        # Log metrics
        mlflow.log_metric("final_best_acc", final_stats["best_acc"])
        if "accuracy_dip" in final_stats and final_stats["accuracy_dip"] is not None:
            mlflow.log_metric("accuracy_dip", final_stats["accuracy_dip"])
        if "recovery_time" in final_stats and final_stats["recovery_time"] is not None:
            mlflow.log_metric("recovery_time", final_stats["recovery_time"])

        mlflow.log_metric("total_seeds", len(seed_manager.seeds))
        active_seeds_count = sum(
            1 for info in seed_manager.seeds.values() if info["module"].state == "active"
        )
        mlflow.log_metric("active_seeds", active_seeds_count)
        mlflow.set_tag("seeds_activated", str(final_stats.get("seeds_activated", False)))

        # Log artifacts
        log_path = project_root / "results" / f"results_{slug}.log"
        if log_path.exists():
            mlflow.log_artifact(str(log_path))

        # Log TensorBoard logs
        tb_dir = project_root / "runs" / slug
        if tb_dir.exists():
            mlflow.log_artifacts(str(tb_dir), "tensorboard")

        # Log and register model
        try:
            mlflow_pytorch.log_model(model, "model")

            # Register model in Model Registry if validation accuracy meets threshold
            if final_stats.get("val_acc", 0) >= 0.7 and ModelRegistry is not None:
                try:
                    registry = ModelRegistry()
                    active_run = mlflow.active_run()
                    if active_run and active_run.info.run_id:
                        run_id = active_run.info.run_id
                        model_version = registry.register_best_model(
                            run_id=run_id,
                            metrics=final_stats,
                            description=f"Morphogenetic model trained on {args.problem_type}",
                            tags={
                                "problem_type": args.problem_type,
                                "device": str(args.device),
                                "seeds_activated": str(final_stats.get("seeds_activated", False)),
                                "training_mode": "single_experiment",
                            },
                        )
                        if model_version:
                            print(f"✅ Model registered: v{model_version.version} "
                                  f"(Val Acc: {final_stats.get('val_acc', 0):.4f})")

                            # Auto-promote to Staging if very good accuracy
                            if final_stats.get("val_acc", 0) >= 0.9:
                                registry.promote_model(
                                    version=model_version.version, stage="Staging"
                                )
                                print(f"🚀 Model promoted to Staging: v{model_version.version}")

                except Exception as e:
                    print(f"Warning: Model registration failed: {e}")

        except (ImportError, RuntimeError, ValueError, OSError) as e:
            print(f"Warning: Could not log model to MLflow: {e}")
            
    except (ImportError, AttributeError, RuntimeError, ValueError, OSError) as e:
        print(f"Warning: MLflow logging failed: {e}")


def run_single_experiment(args, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Run a single experiment with the given arguments and return results."""
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger, tb_writer, log_f, device, config, slug, project_root = setup_experiment(args)

    # Start MLflow run if available
    setup_mlflow_logging(config, slug)

    # Initialize Rich dashboard
    dashboard = RichDashboard()

    try:
        with log_f, dashboard:  # Ensure both log file and dashboard are properly closed
            # Write the detailed configuration header
            write_experiment_log_header(log_f, config, args)

            logger.log_experiment_start()

            train_loader, val_loader = get_dataloaders(args)
            loaders = (train_loader, val_loader)

            model, seed_manager, loss_fn, kasmina = build_model_and_agents(args, device)

            # Initialize seeds in dashboard
            for sid in seed_manager.seeds:
                dashboard.update_seed(sid, "dormant")

            # Phase 1
            logger.log_phase_transition(0, "init", "phase_1")
            tb_writer.add_text("phase/transitions", "Epoch 0: init → phase_1", 0)
            dashboard.show_phase_transition("phase_1", 0)

            # Log phase transition to MLflow
            if MLFLOW_AVAILABLE and mlflow is not None:
                try:
                    mlflow.set_tag("phase", "phase_1")
                except (ImportError, AttributeError, RuntimeError):
                    pass

            best_acc_phase1 = execute_phase_1(
                config, model, loaders, loss_fn, seed_manager, logger, tb_writer, log_f, dashboard
            )

            # Phase 2
            logger.log_phase_transition(config["warm_up_epochs"], "phase_1", "phase_2")
            tb_writer.add_text(
                "phase/transitions",
                f"Epoch {config['warm_up_epochs']}: phase_1 → phase_2",
                config["warm_up_epochs"],
            )
            dashboard.show_phase_transition("phase_2", config["warm_up_epochs"])

            # Log phase transition to MLflow
            if MLFLOW_AVAILABLE and mlflow is not None:
                try:
                    mlflow.set_tag("phase", "phase_2")
                except (ImportError, AttributeError, RuntimeError):
                    pass

            final_stats = execute_phase_2(
                config,
                model,
                loaders,
                loss_fn,
                seed_manager,
                kasmina,
                logger,
                tb_writer,
                log_f,
                best_acc_phase1,
                dashboard,
            )

            logger.log_experiment_end(config["warm_up_epochs"] + config["adaptation_epochs"])
            log_final_summary(logger, final_stats, seed_manager, log_f)

            # Log final metrics and artifacts to MLflow
            log_mlflow_metrics_and_artifacts(final_stats, model, seed_manager, project_root, slug, args)

            # Export metrics for DVC
            export_metrics_for_dvc(final_stats, slug, project_root)

            # Close TensorBoard writer
            tb_writer.close()

            # End MLflow run
            if MLFLOW_AVAILABLE and mlflow is not None:
                try:
                    mlflow.end_run()
                except (ImportError, AttributeError, RuntimeError):
                    pass

            # Cleanup monitoring
            from morphogenetic_engine.monitoring import cleanup_monitoring
            cleanup_monitoring()

            # Return results for sweep summary
            return {
                "run_id": run_id,
                "best_acc": final_stats["best_acc"],
                "accuracy_dip": final_stats.get("accuracy_dip"),
                "recovery_time": final_stats.get("recovery_time"),
                "seeds_activated": final_stats.get("seeds_activated", False),
                "total_seeds": len(seed_manager.seeds),
                "active_seeds": sum(
                    1 for info in seed_manager.seeds.values() if info["module"].state == "active"
                ),
            }
            
    except (RuntimeError, ValueError, KeyError, torch.cuda.OutOfMemoryError) as e:
        # Cleanup monitoring on error
        from morphogenetic_engine.monitoring import cleanup_monitoring
        cleanup_monitoring()

        # End MLflow run on error
        if MLFLOW_AVAILABLE and mlflow is not None:
            try:
                mlflow.end_run(status="FAILED")
            except (ImportError, AttributeError, RuntimeError):
                pass
        tb_writer.close()  # Ensure writer is closed even on error
        dashboard.stop()  # Ensure dashboard is stopped even on error
        print(f"Experiment failed: {e}")
        return {"run_id": run_id, "error": str(e), "best_acc": 0.0, "seeds_activated": False}

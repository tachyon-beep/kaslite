"""
Run a morphogenetic-architecture experiment on the two-spirals dataset.

• Phase 1 – train the full network for warm_up_epochs
• Phase 2 – freeze the trunk, let Kasmina germinate seeds on a plateau
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import logging
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import yaml

import numpy as np
import torch
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from morphogenetic_engine.components import BaseNet
from morphogenetic_engine.core import KasminaMicro, SeedManager
from morphogenetic_engine.logger import ExperimentLogger

_last_report: Dict[str, Optional[str]] = defaultdict(lambda: None)


# ---------- DATA -------------------------------------------------------------


def create_spirals(
    n_samples: int = 2000, noise: float = 0.25, rotations: int = 4, input_dim: int = 2
):
    """Generate the classic two-spirals toy dataset, optionally padded to input_dim."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    n = np.sqrt(rng.random(n_samples // 2)) * rotations * 2 * np.pi
    d1x = np.cos(n) * n + rng.random(n_samples // 2) * noise
    d1y = np.sin(n) * n + rng.random(n_samples // 2) * noise

    X = np.vstack((np.hstack((d1x, -d1x)), np.hstack((d1y, -d1y)))).T
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # Pad with independent N(0,1) features if input_dim > 2
    if input_dim > 2:
        padding = rng.standard_normal((n_samples, input_dim - 2))
        X = np.hstack((X, padding))

    return X.astype(np.float32), y.astype(np.int64)


def create_complex_moons(n_samples: int = 2000, noise: float = 0.1, input_dim: int = 2):
    """Generate complex moons dataset: two half-moons + two Gaussian clusters."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Generate two interleaved half-moons
    n_moons = n_samples // 2
    X_moons, y_moons = make_moons(n_samples=n_moons, noise=noise, random_state=42)

    # Generate two Gaussian clusters
    n_clusters = n_samples - n_moons
    n_cluster1 = n_clusters // 2
    n_cluster2 = n_clusters - n_cluster1

    # Cluster 1: centered at (2, 2)
    cluster1 = rng.multivariate_normal([2.0, 2.0], [[0.5, 0.1], [0.1, 0.5]], n_cluster1)
    y_cluster1 = np.zeros(n_cluster1)

    # Cluster 2: centered at (-2, -2)
    cluster2 = rng.multivariate_normal([-2.0, -2.0], [[0.5, -0.1], [-0.1, 0.5]], n_cluster2)
    y_cluster2 = np.ones(n_cluster2)

    # Concatenate all data
    X = np.vstack((X_moons, cluster1, cluster2))
    y = np.hstack((y_moons, y_cluster1, y_cluster2))

    # Shuffle the data
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    # Pad with independent N(0,1) features if input_dim > 2
    if input_dim > 2:
        padding = rng.standard_normal((len(X), input_dim - 2))
        X = np.hstack((X, padding))

    return X.astype(np.float32), y.astype(np.int64)


def create_moons(
    n_samples: int = 2000,
    moon_noise: float = 0.1,
    moon_sep: float = 0.5,
    input_dim: int = 2,
):
    """Generate two interleaved half-moons dataset."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Generate moons using sklearn
    X, y = make_moons(n_samples=n_samples, noise=moon_noise, random_state=42)

    # Adjust separation by scaling x-coordinates
    X[:, 0] *= moon_sep + 1.0

    # Pad with independent N(0,1) features if input_dim > 2
    if input_dim > 2:
        padding = rng.standard_normal((n_samples, input_dim - 2))
        X = np.hstack((X, padding))

    return X.astype(np.float32), y.astype(np.int64)


def create_clusters(
    cluster_count: int = 2,
    cluster_size: int = 500,
    cluster_std: float = 0.5,
    cluster_sep: float = 3.0,
    input_dim: int = 3,
):
    """Generate Gaussian clusters in n-dimensional space."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Use cluster_size as samples per cluster, but cap total based on realistic limits
    n_samples = cluster_count * cluster_size

    # Generate cluster centers
    centers = []
    for i in range(cluster_count):
        # Place centers in a circle/sphere pattern
        angle = 2 * np.pi * i / cluster_count
        if input_dim == 2:
            center = cluster_sep * np.array([np.cos(angle), np.sin(angle)])
        elif input_dim == 3:
            # Use spherical coordinates for 3D
            phi = np.pi * i / cluster_count  # elevation angle
            center = cluster_sep * np.array(
                [np.cos(angle) * np.sin(phi), np.sin(angle) * np.sin(phi), np.cos(phi)]
            )
        else:
            # For higher dimensions, randomize the remaining coordinates
            center = cluster_sep * np.concatenate(
                [[np.cos(angle), np.sin(angle)], rng.standard_normal(input_dim - 2)]
            )
        centers.append(center)

    # Generate data using make_blobs
    X, y = cast(  # pylint: disable=unbalanced-tuple-unpacking
        tuple[np.ndarray, np.ndarray],
        make_blobs(
            n_samples=n_samples,
            centers=np.array(centers),
            cluster_std=cluster_std,
            n_features=input_dim,
            random_state=42,
            return_centers=False,
        ),
    )

    # Convert to binary classification by grouping clusters
    # Odd clusters become class 0, even clusters become class 1
    y_binary = y % 2

    return X.astype(np.float32), y_binary.astype(np.int64)


def create_spheres(
    sphere_count: int = 2,
    sphere_size: int = 500,
    sphere_radii: str = "1,2",
    sphere_noise: float = 0.05,
    input_dim: int = 3,
):
    """Generate concentric spherical shells in n-dimensional space."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Parse radii string
    radii = [float(r.strip()) for r in sphere_radii.split(",")]
    if len(radii) != sphere_count:
        raise ValueError(f"Number of radii ({len(radii)}) must match sphere_count ({sphere_count})")

    X_list = []
    y_list = []

    for i, radius in enumerate(radii):
        # Generate points on unit sphere surface
        if input_dim == 2:
            # For 2D, generate points on circle
            angles = rng.uniform(0, 2 * np.pi, sphere_size)
            x = np.cos(angles)
            y_coord = np.sin(angles)
            points = np.column_stack([x, y_coord])
        elif input_dim == 3:
            # For 3D, use spherical coordinates
            u = rng.uniform(0, 1, sphere_size)
            v = rng.uniform(0, 1, sphere_size)
            theta = 2 * np.pi * u  # azimuthal angle
            phi = np.arccos(2 * v - 1)  # polar angle

            x = np.sin(phi) * np.cos(theta)
            y_coord = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points = np.column_stack([x, y_coord, z])
        else:
            # For higher dimensions, use Gaussian method and normalize
            points = rng.standard_normal((sphere_size, input_dim))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        # Scale to desired radius and add noise
        points = points * radius
        if sphere_noise > 0:
            noise = rng.normal(0, sphere_noise, points.shape)
            points += noise

        X_list.append(points)
        # Convert to binary classification by grouping spheres
        # Odd spheres become class 0, even spheres become class 1
        y_list.append(np.full(sphere_size, i % 2))

    # Concatenate all spheres
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle the data
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    return X.astype(np.float32), y.astype(np.int64)


# ---------- TRAIN / EVAL HELPERS ---------------------------------------------


def handle_seed_training(seed_manager, device):
    """Handle background seed training and blending for all seeds."""
    for info in seed_manager.seeds.values():
        seed = info["module"]
        if seed.state == "training":
            buf = info["buffer"]
            if len(buf) >= 10:
                sample_tensors = random.sample(list(buf), min(64, len(buf)))
                batch = torch.cat(sample_tensors, dim=0)
                if batch.size(0) > 64:
                    idx = torch.randperm(batch.size(0), device=batch.device)[:64]
                    batch = batch[idx]
                batch = batch.to(device)
                seed.train_child_step(batch)
        seed.update_blending()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    seed_manager: SeedManager,
    scheduler: Optional[LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if optimiser is not None:
            optimiser.zero_grad(set_to_none=True)

        preds = model(X)
        loss = criterion(preds, y)
        total_loss += loss.item()
        num_batches += 1

        if optimiser is not None and loss.requires_grad:
            loss.backward()
            optimiser.step()

        # Background seed training & blending
        handle_seed_training(seed_manager, device)

    # Now that we've done at least one optimizer.step(), it's safe to advance LR
    if scheduler is not None and optimiser is not None and num_batches > 0:
        scheduler.step()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Evaluate the model and return (loss, accuracy)."""
    model.eval()
    loss_accum, correct, total = 0.0, 0, 0
    num_batches = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss_accum += criterion(preds, y).item()
        correct += (preds.argmax(1) == y).sum().item()
        total += y.numel()
        num_batches += 1

    avg_loss = loss_accum / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ---------- MAIN -------------------------------------------------------------


def parse_arguments():
    """Define and parse all command-line arguments using argparse."""
    parser = argparse.ArgumentParser()

    # Sweep configuration
    parser.add_argument(
        "--sweep_config", "-s", 
        type=str, 
        default=None,
        help="Path to YAML sweep configuration file or directory"
    )

    # Existing morphogenetic parameters
    parser.add_argument("--blend_steps", type=int, default=30)
    parser.add_argument("--shadow_lr", type=float, default=1e-3)
    parser.add_argument("--progress_thresh", type=float, default=0.6)
    parser.add_argument(
        "--drift_warn",
        type=float,
        default=0.12,
        help="Drift warning threshold (0=disable)",
    )

    # Problem type and general parameters
    parser.add_argument(
        "--problem_type",
        choices=["spirals", "moons", "clusters", "spheres", "complex_moons"],
        default="spirals",
        help="Type of problem to solve",
    )
    parser.add_argument("--n_samples", type=int, default=2000, help="Total samples (split evenly)")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=3,
        help="Embedding/dimension for clusters and spheres",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8, help="Train/validation split fraction"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help='"cpu" or "cuda"')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Spirals-specific parameters
    parser.add_argument("--noise", type=float, default=0.25, help="Spirals noise")
    parser.add_argument("--rotations", type=int, default=4, help="Spirals turns")

    # Moons-specific parameters
    parser.add_argument("--moon_noise", type=float, default=0.1, help="Gaussian noise for moons")
    parser.add_argument("--moon_sep", type=float, default=0.5, help="Separation between half-moons")

    # Clusters-specific parameters
    parser.add_argument("--cluster_count", type=int, default=2, help="Number of Gaussian blobs")
    parser.add_argument("--cluster_size", type=int, default=500, help="Points per cluster")
    parser.add_argument("--cluster_std", type=float, default=0.5, help="Cluster standard deviation")
    parser.add_argument(
        "--cluster_sep",
        type=float,
        default=3.0,
        help="Distance between cluster centers",
    )

    # Spheres-specific parameters
    parser.add_argument(
        "--sphere_count",
        type=int,
        default=2,
        help="Number of concentric spherical shells",
    )
    parser.add_argument("--sphere_size", type=int, default=500, help="Points per sphere shell")
    parser.add_argument(
        "--sphere_radii", type=str, default="1,2", help="Comma-separated list of radii"
    )
    parser.add_argument(
        "--sphere_noise",
        type=float,
        default=0.05,
        help="Jitter magnitude on sphere surface",
    )

    # Additional hyper-parameters
    parser.add_argument(
        "--warm_up_epochs",
        type=int,
        default=50,
        help="Number of warm-up epochs before adaptation phase",
    )
    parser.add_argument(
        "--adaptation_epochs",
        type=int,
        default=200,
        help="Number of epochs for the adaptation phase",
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension size for the network",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Number of hidden layers in the network (each with a corresponding seed)",
    )
    parser.add_argument(
        "--seeds_per_layer",
        type=int,
        default=1,
        help="Number of sentinel seeds per layer (allows multiple adaptive paths per layer)",
    )
    parser.add_argument(
        "--acc_threshold",
        type=float,
        default=0.95,
        help="Accuracy threshold for Kasmina germination",
    )

    return parser.parse_args()


def setup_experiment(args):
    """Configure the experiment environment based on the parsed arguments."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Set device based on args and availability
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # Construct configuration for the experiment logger
    config = {
        "problem_type": args.problem_type,
        "n_samples": args.n_samples,
        "input_dim": args.input_dim,
        "train_frac": args.train_frac,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "warm_up_epochs": args.warm_up_epochs,
        "adaptation_epochs": args.adaptation_epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "seeds_per_layer": args.seeds_per_layer,
        "blend_steps": args.blend_steps,
        "shadow_lr": args.shadow_lr,
        "progress_thresh": args.progress_thresh,
        "drift_warn": args.drift_warn,
        "acc_threshold": args.acc_threshold,
    }

    slug = (
        f"{args.problem_type}_dim{args.input_dim}_{args.device}"
        f"_h{args.hidden_dim}_bs{args.blend_steps}"
        f"_lr{args.shadow_lr}_pt{args.progress_thresh}"
        f"_dw{args.drift_warn}"
    )

    # Determine log location and initialise logger
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "results"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"results_{slug}.log"

    logger = ExperimentLogger(str(log_path), config)
    log_f = log_path.open("w", encoding="utf-8")

    # Create TensorBoard writer
    tb_dir = project_root / "runs" / slug
    tb_writer = SummaryWriter(log_dir=str(tb_dir))

    return logger, tb_writer, log_f, device, config


def write_log_header(log_f, config, args):
    """Write the detailed configuration header to the log file."""
    _last_report.clear()

    # Write comprehensive configuration header
    log_f.write("# Morphogenetic Architecture Experiment Log\n")
    log_f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write("# Configuration:\n")
    log_f.write(f"# problem_type: {args.problem_type}\n")
    log_f.write(f"# n_samples: {args.n_samples}\n")
    log_f.write(f"# input_dim: {args.input_dim}\n")
    log_f.write(f"# train_frac: {args.train_frac}\n")
    log_f.write(f"# batch_size: {args.batch_size}\n")
    log_f.write(f"# device: {config['device']}\n")
    log_f.write(f"# seed: {args.seed}\n")

    # Dataset-specific parameters
    if args.problem_type == "spirals":
        log_f.write(f"# noise: {args.noise}\n")
        log_f.write(f"# rotations: {args.rotations}\n")
    elif args.problem_type in ["moons", "complex_moons"]:
        log_f.write(f"# moon_noise: {args.moon_noise}\n")
        if args.problem_type == "moons":
            log_f.write(f"# moon_sep: {args.moon_sep}\n")
    elif args.problem_type == "clusters":
        log_f.write(f"# cluster_count: {args.cluster_count}\n")
        log_f.write(f"# cluster_size: {args.cluster_size}\n")
        log_f.write(f"# cluster_std: {args.cluster_std}\n")
        log_f.write(f"# cluster_sep: {args.cluster_sep}\n")
    elif args.problem_type == "spheres":
        log_f.write(f"# sphere_count: {args.sphere_count}\n")
        log_f.write(f"# sphere_size: {args.sphere_size}\n")
        log_f.write(f"# sphere_radii: {args.sphere_radii}\n")
        log_f.write(f"# sphere_noise: {args.sphere_noise}\n")

    # Morphogenetic architecture parameters
    log_f.write(f"# warm_up_epochs: {args.warm_up_epochs}\n")
    log_f.write(f"# adaptation_epochs: {args.adaptation_epochs}\n")
    log_f.write(f"# lr: {args.lr}\n")
    log_f.write(f"# hidden_dim: {args.hidden_dim}\n")
    log_f.write(f"# num_layers: {args.num_layers}\n")
    log_f.write(f"# seeds_per_layer: {args.seeds_per_layer}\n")
    log_f.write(f"# blend_steps: {args.blend_steps}\n")
    log_f.write(f"# shadow_lr: {args.shadow_lr}\n")
    log_f.write(f"# progress_thresh: {args.progress_thresh}\n")
    log_f.write(f"# drift_warn: {args.drift_warn}\n")
    log_f.write(f"# acc_threshold: {args.acc_threshold}\n")
    log_f.write("#\n")
    log_f.write("# Data format: epoch,seed,state,alpha\n")
    log_f.write("epoch,seed,state,alpha\n")


def get_dataloaders(args):
    """Generate or load the specified dataset and create DataLoader instances."""
    # Dispatch on problem type to call appropriate generator
    if args.problem_type == "spirals":
        X, y = create_spirals(
            n_samples=args.n_samples,
            noise=args.noise,
            rotations=args.rotations,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "moons":
        X, y = create_moons(
            n_samples=args.n_samples,
            moon_noise=args.moon_noise,
            moon_sep=args.moon_sep,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "complex_moons":
        X, y = create_complex_moons(
            n_samples=args.n_samples,
            noise=args.moon_noise,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "clusters":
        X, y = create_clusters(
            cluster_count=args.cluster_count,
            cluster_size=args.n_samples // args.cluster_count,
            cluster_std=args.cluster_std,
            cluster_sep=args.cluster_sep,
            input_dim=args.input_dim,
        )
    elif args.problem_type == "spheres":
        X, y = create_spheres(
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


def build_model_and_agents(args, device):
    """Initialize the SeedManager, BaseNet model, loss function, and KasminaMicro."""
    seed_manager = SeedManager()
    model = BaseNet(
        args.hidden_dim,
        seed_manager=seed_manager,
        input_dim=args.input_dim,
        num_layers=args.num_layers,
        seeds_per_layer=args.seeds_per_layer,
        blend_steps=args.blend_steps,
        shadow_lr=args.shadow_lr,
        progress_thresh=args.progress_thresh,
        drift_warn=args.drift_warn,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    kasmina = KasminaMicro(seed_manager, patience=15, delta=5e-4, acc_threshold=args.acc_threshold)

    return model, seed_manager, loss_fn, kasmina


def should_log_seed_update(mod, prev):
    """Determine if a seed update should be logged."""
    tag = f"{mod.state}:{mod.alpha:.2f}" if mod.state == "blending" else mod.state

    if prev != tag:
        return True, tag

    if (
        mod.state == "blending"
        and prev
        and ":" in prev
        and float(prev.split(":")[1]) + 0.1 <= mod.alpha
    ):
        return True, tag

    return False, tag


def format_alpha_value(alpha_val):
    """Format alpha value for logging."""
    try:
        return f"{float(alpha_val):.3f}"
    except (TypeError, ValueError):
        return str(alpha_val)


def log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f):
    """Handle the repetitive logic of checking and logging seed state transitions."""
    for sid, info in seed_manager.seeds.items():
        mod = info["module"]
        prev = _last_report[sid]

        should_log, tag = should_log_seed_update(mod, prev)

        if should_log:
            _last_report[sid] = tag

            # Use the experiment logger instead of direct logging
            if mod.state == "blending":
                logger.log_blending_progress(epoch, sid, mod.alpha)
                # Log alpha values to TensorBoard
                tb_writer.add_scalar(f"seed/{sid}/alpha", mod.alpha, epoch)
            else:
                # For other states, use seed_event logging
                prev_state = prev.split(":")[0] if prev and ":" in prev else (prev or "unknown")
                logger.log_seed_event(epoch, sid, prev_state, mod.state)
                # Log state transitions to TensorBoard
                tb_writer.add_text(
                    f"seed/{sid}/events",
                    f"Epoch {epoch}: {prev_state} → {mod.state}",
                    epoch
                )

            alpha_str = format_alpha_value(getattr(mod, "alpha", 0.0))
            log_f.write(f"{epoch},{sid},{mod.state},{alpha_str}\n")


def execute_phase_1(config, model, loaders, loss_fn, seed_manager, logger, tb_writer, log_f):
    """Run the initial warm-up training phase."""
    train_loader, val_loader = loaders
    device = next(model.parameters()).device

    # Initialize optimizer and scheduler for the full model
    optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 20, 0.1)

    best_acc = 0.0
    warm_up_epochs = config["warm_up_epochs"]

    for epoch in range(1, warm_up_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, loss_fn, seed_manager, scheduler, device)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        best_acc = max(best_acc, val_acc)

        logger.log_epoch_progress(
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
            },
        )

        # Log to TensorBoard
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("validation/loss", val_loss, epoch)
        tb_writer.add_scalar("validation/accuracy", val_acc, epoch)
        tb_writer.add_scalar("validation/best_acc", best_acc, epoch)

        log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f)

    return best_acc


def handle_germination_tracking(epoch, germ_epoch, acc_pre, acc_post, val_acc, t_recover):
    """Handle the tracking of germination events and recovery time."""
    if germ_epoch and epoch == germ_epoch + 1:
        acc_post = val_acc
    if germ_epoch and t_recover is None and acc_pre is not None and val_acc >= acc_pre:
        t_recover = epoch - germ_epoch
    return acc_post, t_recover


def execute_phase_2(
    config, model, loaders, loss_fn, seed_manager, kasmina, logger, tb_writer, log_f, initial_best_acc
):
    """Run the adaptation phase where the backbone is frozen and seeds can germinate."""
    train_loader, val_loader = loaders
    device = next(model.parameters()).device
    warm_up_epochs = config["warm_up_epochs"]
    adaptation_epochs = config["adaptation_epochs"]
    lr = config["lr"]

    # Freeze the model's backbone
    model.freeze_backbone()

    # Define the rebuild_seed_opt helper function
    def rebuild_seed_opt():
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return None, None
        opt = torch.optim.Adam(params, lr=lr * 0.1, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1)
        return opt, sch

    optimiser, scheduler = rebuild_seed_opt()
    best_acc = initial_best_acc
    acc_pre = acc_post = t_recover = germ_epoch = None
    seeds_activated = False

    for epoch in range(warm_up_epochs + 1, warm_up_epochs + adaptation_epochs + 1):
        train_loss = 0.0
        if optimiser:
            train_loss = train_epoch(
                model,
                train_loader,
                optimiser,
                loss_fn,
                seed_manager,
                scheduler,
                device,
            )

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        if kasmina.step(val_loss, val_acc):
            seeds_activated = True
            germ_epoch, acc_pre = epoch, val_acc
            optimiser, scheduler = rebuild_seed_opt()

        acc_post, t_recover = handle_germination_tracking(
            epoch, germ_epoch, acc_pre, acc_post, val_acc, t_recover
        )

        if epoch % 10 == 0 or val_acc > best_acc:
            best_acc = max(best_acc, val_acc)
        status = ", ".join(f"{sid}:{info['status']}" for sid, info in seed_manager.seeds.items())
        logger.log_epoch_progress(
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
                "seeds": status,
            },
        )

        # Log to TensorBoard
        if train_loss > 0:  # Only log train loss if we actually trained
            tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("validation/loss", val_loss, epoch)
        tb_writer.add_scalar("validation/accuracy", val_acc, epoch)
        tb_writer.add_scalar("validation/best_acc", best_acc, epoch)

        log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f)

    return {
        "best_acc": best_acc,
        "accuracy_dip": (
            (acc_pre - acc_post) if acc_pre is not None and acc_post is not None else None
        ),
        "recovery_time": t_recover,
        "seeds_activated": seeds_activated,
        "acc_pre": acc_pre,
        "acc_post": acc_post,
    }


def log_final_summary(logger, final_stats, seed_manager, log_f):
    """Print the final report to the console and write the summary footer to the log file."""
    # The logger's generate_final_report() is automatically printed by the experiment_end event

    if final_stats["acc_pre"] is not None and final_stats["acc_post"] is not None:
        # Use the logger's accuracy dip method instead of direct logging
        logger.log_accuracy_dip(0, final_stats["accuracy_dip"])

    # Write log footer
    log_f.write("#\n")
    log_f.write("# Experiment completed successfully\n")
    log_f.write(f"# End timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_f.write(f"# Final best accuracy: {final_stats['best_acc']:.4f}\n")
    if final_stats["seeds_activated"]:
        active_seeds = sum(
            1 for info in seed_manager.seeds.values() if info["module"].state == "active"
        )
        log_f.write(f"# Seeds activated: {active_seeds}/{len(seed_manager.seeds)}\n")
    else:
        log_f.write(f"# Seeds activated: 0/{len(seed_manager.seeds)}\n")
    log_f.write("# ===== LOG COMPLETE =====\n")


# ---------- SWEEP CONFIGURATION ---------------------------------------------

def parse_value_list(value: Union[str, List, int, float]) -> List[Any]:
    """Parse a value that could be a comma-separated string, list, or single value."""
    if isinstance(value, str):
        # Handle comma-separated strings
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        else:
            return [value]
    elif isinstance(value, list):
        return value
    else:
        # Single numeric or other value
        return [value]


def validate_sweep_config(sweep_config: Dict[str, Any], valid_args: set) -> None:
    """Validate that all keys in the sweep config are valid argument names."""
    for key in sweep_config:
        # Remove leading dashes if present to match argument names
        clean_key = key.lstrip('-')
        if clean_key not in valid_args:
            raise ValueError(f"Unknown parameter in sweep config: '{key}'. Valid parameters: {sorted(valid_args)}")


def load_sweep_configs(config_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load YAML sweep configuration(s) from a file or directory."""
    config_path = Path(config_path)
    configs = []
    
    if config_path.is_file():
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                configs.append(yaml.safe_load(f))
        else:
            raise ValueError(f"Sweep config file must have .yml or .yaml extension: {config_path}")
    elif config_path.is_dir():
        yaml_files = list(config_path.glob('*.yml')) + list(config_path.glob('*.yaml'))
        if not yaml_files:
            raise ValueError(f"No YAML files found in directory: {config_path}")
        for yaml_file in sorted(yaml_files):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                configs.append(yaml.safe_load(f))
    else:
        raise ValueError(f"Sweep config path does not exist: {config_path}")
    
    return configs


def expand_grid(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a sweep configuration into a grid of parameter combinations."""
    if not sweep_config:
        return [{}]
    
    # Parse all values into lists
    param_lists = {}
    for key, value in sweep_config.items():
        param_lists[key] = parse_value_list(value)
    
    # Create cartesian product
    keys = list(param_lists.keys())
    value_combinations = itertools.product(*param_lists.values())
    
    # Convert to list of dictionaries
    grid = []
    for combination in value_combinations:
        combo_dict = dict(zip(keys, combination))
        grid.append(combo_dict)
    
    return grid


def merge_args_with_combo(base_args: argparse.Namespace, combo: Dict[str, Any]) -> argparse.Namespace:
    """Merge base CLI arguments with a parameter combination from the sweep grid."""
    # Create a new namespace with base args
    merged_args = argparse.Namespace(**vars(base_args))
    
    # Override with combo values, converting types appropriately
    for key, value in combo.items():
        # Convert string values to appropriate types based on the original arg type
        if hasattr(merged_args, key):
            original_value = getattr(merged_args, key)
            if isinstance(original_value, bool):
                # Handle boolean conversion
                if isinstance(value, str):
                    converted_value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    converted_value = bool(value)
            elif isinstance(original_value, int):
                converted_value = int(value)
            elif isinstance(original_value, float):
                converted_value = float(value)
            else:
                converted_value = value
            
            setattr(merged_args, key, converted_value)
        else:
            # New parameter not in base args - add as-is
            setattr(merged_args, key, value)
    
    return merged_args


def generate_run_slug(combo: Dict[str, Any], run_index: int) -> str:
    """Generate a unique slug for a parameter combination."""
    # Create a short hash of the parameter combination
    combo_str = "_".join(f"{k}={v}" for k, v in sorted(combo.items()))
    combo_hash = hashlib.md5(combo_str.encode()).hexdigest()[:8]
    
    return f"run_{run_index:03d}_{combo_hash}"


def create_sweep_results_summary(sweep_runs: List[Dict[str, Any]], sweep_dir: Path) -> None:
    """Create a CSV summary of all sweep runs and their results."""
    if not sweep_runs:
        return
    
    summary_path = sweep_dir / "sweep_summary.csv"
    
    # Collect all parameter names and result keys
    all_params = set()
    all_results = set()
    
    for run in sweep_runs:
        all_params.update(run.get('parameters', {}).keys())
        all_results.update(run.get('results', {}).keys())
    
    # Write CSV
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['run_id', 'run_slug'] + sorted(all_params) + sorted(all_results)
        writer.writerow(header)
        
        # Data rows
        for run in sweep_runs:
            row = [
                run.get('run_id', ''),
                run.get('run_slug', '')
            ]
            
            # Add parameter values
            params = run.get('parameters', {})
            for param in sorted(all_params):
                row.append(params.get(param, ''))
            
            # Add result values
            results = run.get('results', {})
            for result in sorted(all_results):
                row.append(results.get(result, ''))
            
            writer.writerow(row)
    
    print(f"Sweep summary saved to: {summary_path}")


def get_valid_argument_names() -> set:
    """Get the set of valid argument names for validation."""
    # Return the set of valid argument names
    valid_args = {
        'sweep_config', 'blend_steps', 'shadow_lr', 'progress_thresh', 'drift_warn',
        'problem_type', 'n_samples', 'input_dim', 'train_frac', 'batch_size', 
        'device', 'seed', 'noise', 'rotations', 'moon_noise', 'moon_sep',
        'cluster_count', 'cluster_size', 'cluster_std', 'cluster_sep',
        'sphere_count', 'sphere_size', 'sphere_radii', 'sphere_noise',
        'warm_up_epochs', 'adaptation_epochs', 'lr', 'hidden_dim',
        'num_layers', 'seeds_per_layer', 'acc_threshold'
    }
    
    return valid_args


def run_single_experiment(args: argparse.Namespace, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Run a single experiment with the given arguments and return results."""
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    logger, tb_writer, log_f, device, config = setup_experiment(args)

    try:
        with log_f:  # Ensure the log file is always closed
            # Write the detailed configuration header
            write_log_header(log_f, config, args)

            logger.log_experiment_start()

            train_loader, val_loader = get_dataloaders(args)
            loaders = (train_loader, val_loader)

            model, seed_manager, loss_fn, kasmina = build_model_and_agents(args, device)

            logger.log_phase_transition(0, "init", "phase_1")
            tb_writer.add_text("phase/transitions", "Epoch 0: init → phase_1", 0)
            best_acc_phase1 = execute_phase_1(
                config, model, loaders, loss_fn, seed_manager, logger, tb_writer, log_f
            )
            logger.log_phase_transition(config["warm_up_epochs"], "phase_1", "phase_2")
            tb_writer.add_text("phase/transitions", f"Epoch {config['warm_up_epochs']}: phase_1 → phase_2", config["warm_up_epochs"])

            final_stats = execute_phase_2(
                config, model, loaders, loss_fn, seed_manager, kasmina, logger, tb_writer, log_f, best_acc_phase1
            )

            logger.log_experiment_end(config["warm_up_epochs"] + config["adaptation_epochs"])
            log_final_summary(logger, final_stats, seed_manager, log_f)
            
            # Close TensorBoard writer
            tb_writer.close()
            
            # Return results for sweep summary
            return {
                'run_id': run_id,
                'best_acc': final_stats['best_acc'],
                'accuracy_dip': final_stats.get('accuracy_dip'),
                'recovery_time': final_stats.get('recovery_time'),
                'seeds_activated': final_stats.get('seeds_activated', False),
                'total_seeds': len(seed_manager.seeds),
                'active_seeds': sum(1 for info in seed_manager.seeds.values() 
                                   if info["module"].state == "active")
            }
    except Exception as e:
        tb_writer.close()  # Ensure writer is closed even on error
        print(f"Experiment failed: {e}")
        return {
            'run_id': run_id,
            'error': str(e),
            'best_acc': 0.0,
            'seeds_activated': False
        }


def create_run_directory_and_setup(sweep_dir: Path, run_slug: str, original_setup_func):
    """Create run directory and setup function for a single sweep run."""
    run_dir = sweep_dir / run_slug
    run_dir.mkdir(exist_ok=True)
    
    def setup_experiment_for_sweep(args_inner, run_dir_param=run_dir, original_setup_param=original_setup_func):
        """Modified setup_experiment that puts logs in the run directory."""
        logger_inner, tb_writer_inner, log_f_inner, device_inner, config_inner = original_setup_param(args_inner)
        
        # Move the log file to the run directory
        original_log_path = Path(log_f_inner.name)
        new_log_path = run_dir_param / original_log_path.name
        log_f_inner.close()
        
        if original_log_path.exists():
            shutil.move(str(original_log_path), str(new_log_path))
        
        # Reopen with new path
        log_f_new = new_log_path.open("w", encoding="utf-8")
        
        # Close the original TensorBoard writer and create a new one in the run directory
        tb_writer_inner.close()
        tb_dir_new = run_dir_param / "tensorboard"
        tb_writer_new = SummaryWriter(log_dir=str(tb_dir_new))
        
        return logger_inner, tb_writer_new, log_f_new, device_inner, config_inner
    
    return run_dir, setup_experiment_for_sweep


def process_single_sweep_config(config_idx: int, sweep_config: Dict[str, Any], 
                               args: argparse.Namespace, sweep_dir: Path, 
                               run_counter: int, valid_args: set) -> tuple[List[Dict[str, Any]], int]:
    """Process a single sweep configuration and return results and updated counter."""
    print(f"\nProcessing sweep config {config_idx + 1}")
    
    # Validate configuration
    try:
        validate_sweep_config(sweep_config, valid_args)
    except ValueError as e:
        print(f"Error in sweep config {config_idx + 1}: {e}")
        return [], run_counter
    
    # Expand parameter grid
    param_grid = expand_grid(sweep_config)
    print(f"Generated {len(param_grid)} parameter combinations")
    
    config_runs = []
    original_setup = setup_experiment
    
    # Run experiments for this config
    for combo in param_grid:
        run_counter += 1
        run_slug = generate_run_slug(combo, run_counter)
        
        print(f"\nRun {run_counter}: {run_slug}")
        print(f"Parameters: {combo}")
        
        # Merge CLI args with parameter combination
        run_args = merge_args_with_combo(args, combo)
        
        # Create run-specific setup
        _, setup_func = create_run_directory_and_setup(sweep_dir, run_slug, original_setup)
        
        # Temporarily replace setup_experiment
        globals()['setup_experiment'] = setup_func
        
        try:
            # Run the experiment
            experiment_results = run_single_experiment(run_args, run_slug)
            
            # Separate parameters from results
            run_record = {
                'run_id': run_slug,
                'run_slug': run_slug,
                'parameters': combo,
                'results': {k: v for k, v in experiment_results.items() 
                           if k not in ['run_id', 'parameters']}
            }
            config_runs.append(run_record)
            
            print(f"  Final accuracy: {experiment_results.get('best_acc', 0.0):.4f}")
            if experiment_results.get('error'):
                print(f"  Error: {experiment_results['error']}")
                
        finally:
            # Restore original setup_experiment
            globals()['setup_experiment'] = original_setup
    
    return config_runs, run_counter


def run_parameter_sweep(args: argparse.Namespace) -> None:
    """Run a parameter sweep based on the sweep configuration."""
    print(f"Loading sweep configuration from: {args.sweep_config}")
    
    # Load and validate sweep configs
    try:
        sweep_configs = load_sweep_configs(args.sweep_config)
    except Exception as e:
        print(f"Error loading sweep config: {e}")
        return
    
    valid_args = get_valid_argument_names()
    
    # Create sweep results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent
    sweep_dir = project_root / "results" / "sweeps" / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sweep results will be saved to: {sweep_dir}")
    
    # Process all sweep configs
    all_runs = []
    run_counter = 0
    
    for config_idx, sweep_config in enumerate(sweep_configs):
        config_runs, run_counter = process_single_sweep_config(
            config_idx, sweep_config, args, sweep_dir, run_counter, valid_args
        )
        all_runs.extend(config_runs)
    
    # Create summary and print results
    create_sweep_results_summary(all_runs, sweep_dir)
    
    print(f"\nSweep completed! {len(all_runs)} experiments run.")
    print(f"Results saved to: {sweep_dir}")
    
    # Print quick summary
    if all_runs:
        successful_runs = [r for r in all_runs if not r.get('error')]
        if successful_runs:
            best_run = max(successful_runs, key=lambda x: x.get('best_acc', 0.0))
            print(f"Best accuracy: {best_run.get('best_acc', 0.0):.4f} (run: {best_run.get('run_slug', 'unknown')})")


def main():
    """Main function to orchestrate single experiments or parameter sweeps."""
    args = parse_arguments()
    
    if args.sweep_config:
        # Run parameter sweep
        run_parameter_sweep(args)
    else:
        # Run single experiment
        run_single_experiment(args)


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()

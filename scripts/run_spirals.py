"""
Run a morphogenetic-architecture experiment on the two-spirals dataset.

• Phase 1 – train the full network for warm_up_epochs
• Phase 2 – freeze the trunk, let Kasmina germinate seeds on a plateau
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.datasets import make_blobs, make_moons
except ImportError:
    raise ImportError(
        "This script requires scikit-learn. Please install it with: pip install scikit-learn"
    )

from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset

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
    cluster2 = rng.multivariate_normal(
        [-2.0, -2.0], [[0.5, -0.1], [-0.1, 0.5]], n_cluster2
    )
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
    X, y = make_blobs(
        n_samples=n_samples,
        centers=np.array(centers),
        cluster_std=cluster_std,
        n_features=input_dim,
        random_state=42,
        return_centers=False,
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
        raise ValueError(
            f"Number of radii ({len(radii)}) must match sphere_count ({sphere_count})"
        )

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
            # Scheduler step moved outside the batch loop

        # background seed training & blending
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

    # Step the scheduler once per epoch after all batches
    # Only step if we have an optimizer and we've processed batches
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
def main():
    """Main function to run the morphogenetic architecture experiment."""
    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--n_samples", type=int, default=2000, help="Total samples (split evenly)"
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=3,
        help="Embedding/dimension for clusters and spheres",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8, help="Train/validation split fraction"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="DataLoader batch size"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help='"cpu" or "cuda"'
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Spirals-specific parameters
    parser.add_argument("--noise", type=float, default=0.25, help="Spirals noise")
    parser.add_argument("--rotations", type=int, default=4, help="Spirals turns")

    # Moons-specific parameters
    parser.add_argument(
        "--moon_noise", type=float, default=0.1, help="Gaussian noise for moons"
    )
    parser.add_argument(
        "--moon_sep", type=float, default=0.5, help="Separation between half-moons"
    )

    # Clusters-specific parameters
    parser.add_argument(
        "--cluster_count", type=int, default=2, help="Number of Gaussian blobs"
    )
    parser.add_argument(
        "--cluster_size", type=int, default=500, help="Points per cluster"
    )
    parser.add_argument(
        "--cluster_std", type=float, default=0.5, help="Cluster standard deviation"
    )
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
    parser.add_argument(
        "--sphere_size", type=int, default=500, help="Points per sphere shell"
    )
    parser.add_argument(
        "--sphere_radii", type=str, default="1,2", help="Comma-separated list of radii"
    )
    parser.add_argument(
        "--sphere_noise",
        type=float,
        default=0.05,
        help="Jitter magnitude on sphere surface",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Set device based on args and availability
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )

    # ---------------- hyper-parameters ----------------
    warm_up_epochs = 50
    adaptation_epochs = 200
    lr = 3e-3
    hidden_dim = 128
    acc_threshold = 0.95
    # --------------------------------------------------

    # Construct configuration for the experiment logger
    config = {
        "problem_type": args.problem_type,
        "n_samples": args.n_samples,
        "input_dim": args.input_dim,
        "train_frac": args.train_frac,
        "batch_size": args.batch_size,
        "device": str(device),
        "seed": args.seed,
        "warm_up_epochs": warm_up_epochs,
        "adaptation_epochs": adaptation_epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "blend_steps": args.blend_steps,
        "shadow_lr": args.shadow_lr,
        "progress_thresh": args.progress_thresh,
        "drift_warn": args.drift_warn,
        "acc_threshold": acc_threshold,
    }

    slug = (
        f"{args.problem_type}_dim{args.input_dim}_{args.device}"
        f"_h{hidden_dim}_bs{args.blend_steps}"
        f"_lr{args.shadow_lr}_pt{args.progress_thresh}"
        f"_dw{args.drift_warn}"
    )

    # Determine log location and initialise logger
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"results_{slug}.log"

    logger = ExperimentLogger(str(log_path), config)

    # ---- open log file with context-manager ----
    with log_path.open("w", encoding="utf-8") as log_f:
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
        log_f.write(f"# device: {device}\n")
        log_f.write(f"# seed: {args.seed}\n")

        logger.log_experiment_start()

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
        log_f.write(f"# warm_up_epochs: {warm_up_epochs}\n")
        log_f.write(f"# adaptation_epochs: {adaptation_epochs}\n")
        log_f.write(f"# lr: {lr}\n")
        log_f.write(f"# hidden_dim: {hidden_dim}\n")
        log_f.write(f"# blend_steps: {args.blend_steps}\n")
        log_f.write(f"# shadow_lr: {args.shadow_lr}\n")
        log_f.write(f"# progress_thresh: {args.progress_thresh}\n")
        log_f.write(f"# drift_warn: {args.drift_warn}\n")
        log_f.write(f"# acc_threshold: {acc_threshold}\n")
        log_f.write("#\n")
        log_f.write("# Data format: epoch,seed,state,alpha\n")
        log_f.write("epoch,seed,state,alpha\n")

        # ---------- data ----------
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

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
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=0)

        # ---------- model & agents ----------
        seed_manager = SeedManager()
        model = BaseNet(
            hidden_dim,
            input_dim=args.input_dim,
            blend_steps=args.blend_steps,
            shadow_lr=args.shadow_lr,
            progress_thresh=args.progress_thresh,
            drift_warn=args.drift_warn,
        ).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        kasmina = KasminaMicro(
            seed_manager, patience=15, delta=5e-4, acc_threshold=acc_threshold
        )

        # ---------- optimiser & scheduler (phase 1) ----------
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 20, 0.1)

        best_acc = 0.0
        acc_pre = acc_post = t_recover = germ_epoch = None

        # ---------------- Phase 1 ----------------
        logger.log_phase_transition(0, "init", "phase_1")
        for epoch in range(1, warm_up_epochs + 1):
            train_epoch(
                model, train_loader, optimiser, loss_fn, seed_manager, scheduler, device
            )

            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            best_acc = max(best_acc, val_acc)

            logger.log_epoch_progress(
                epoch,
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_acc": best_acc,
                },
            )

            for sid, info in seed_manager.seeds.items():
                mod = info["module"]
                tag = (
                    f"{mod.state}:{mod.alpha:.2f}"
                    if mod.state == "blending"
                    else mod.state
                )
                prev = _last_report[sid]
                should = (prev != tag) or (
                    mod.state == "blending"
                    and prev
                    and ":" in prev
                    and float(prev.split(":")[1]) + 0.1 <= mod.alpha
                )
                if should:
                    _last_report[sid] = tag
                    logging.info("epoch %d %s %s", epoch, sid, tag)
                    log_f.write(f"{epoch},{sid},{mod.state},{mod.alpha:.3f}\n")

        logger.log_phase_transition(warm_up_epochs, "phase_1", "phase_2")
        model.freeze_backbone()

        # ---------- phase-2 optimiser builder ----------
        def rebuild_seed_opt():
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                return None, None
            opt = torch.optim.Adam(params, lr=lr * 0.1, weight_decay=0.0)
            sch = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1)
            return opt, sch

        optimiser, scheduler = rebuild_seed_opt()
        seeds_activated = False

        # ---------------- Phase 2 ----------------
        logger.log_phase_transition(warm_up_epochs, "phase_1_frozen", "phase_2")
        for epoch in range(warm_up_epochs + 1, warm_up_epochs + adaptation_epochs + 1):
            if optimiser:
                train_epoch(
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

            if germ_epoch and epoch == germ_epoch + 1:
                acc_post = val_acc
            if (
                germ_epoch
                and t_recover is None
                and acc_pre is not None
                and val_acc >= acc_pre
            ):
                t_recover = epoch - germ_epoch

            if epoch % 10 == 0 or val_acc > best_acc:
                best_acc = max(best_acc, val_acc)
            status = ", ".join(
                f"{sid}:{info['status']}" for sid, info in seed_manager.seeds.items()
            )
            logger.log_epoch_progress(
                epoch,
                {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_acc": best_acc,
                    "seeds": status,
                },
            )

            for sid, info in seed_manager.seeds.items():
                mod = info["module"]
                tag = (
                    f"{mod.state}:{mod.alpha:.2f}"
                    if mod.state == "blending"
                    else mod.state
                )
                prev = _last_report[sid]
                should = (prev != tag) or (
                    mod.state == "blending"
                    and prev
                    and ":" in prev
                    and float(prev.split(":")[1]) + 0.1 <= mod.alpha
                )
                if should:
                    _last_report[sid] = tag
                    logging.info("epoch %d %s %s", epoch, sid, tag)
                    alpha_str = f"{mod.alpha:.3f}" if mod.state == "blending" else ""
                    log_f.write(f"{epoch},{sid},{mod.state},{alpha_str}\n")

        # ------------- final stats -------------
        logger.log_experiment_end(warm_up_epochs + adaptation_epochs)
        print(logger.generate_final_report())

        if acc_pre is not None and acc_post is not None:
            logging.info(
                "accuracy dip %.3f, recovery %s epochs", acc_pre - acc_post, t_recover
            )

        # ------------- log footer -------------
        log_f.write("#\n")
        log_f.write("# Experiment completed successfully\n")
        log_f.write(
            f"# End timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_f.write(f"# Final best accuracy: {best_acc:.4f}\n")
        if seeds_activated:
            active_seeds = sum(
                1
                for info in seed_manager.seeds.values()
                if info["module"].state == "active"
            )
            log_f.write(
                f"# Seeds activated: {active_seeds}/{len(seed_manager.seeds)}\n"
            )
        else:
            log_f.write("# Seeds activated: 0/{}\n".format(len(seed_manager.seeds)))
        log_f.write("# ===== LOG COMPLETE =====\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()

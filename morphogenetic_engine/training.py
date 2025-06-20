"""
Training and evaluation utilities for morphogenetic architecture experiments.

This module provides functions for training neural networks, evaluating performance,
and executing different phases of morphogenetic experiments, reporting all progress
exclusively through the logger.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from typing import Any, Dict

import mlflow
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .core import KasminaMicro, SeedManager
from .events import SeedInfo, SeedState
from .logger import ExperimentLogger

# Check if we're in testing mode to conditionally disable MLflow
TESTING_MODE = "pytest" in sys.modules or "unittest" in sys.modules
MLFLOW_AVAILABLE = not TESTING_MODE

# Global variable to track seed update reporting
_last_report: Dict[str, str] = defaultdict(lambda: "")


def handle_seed_training(seed_manager: "SeedManager", device: torch.device, epoch: int | None = None):
    """Handle background seed training and blending for all seeds."""
    for info in seed_manager.seeds.values():
        seed = info["module"]

        # Store the latest epoch for logging purposes
        if epoch is not None:
            info['last_epoch'] = epoch

        if seed.state == SeedState.TRAINING.value:
            buf = info["buffer"]
            if len(buf) >= 10:
                sample_tensors = random.sample(list(buf), min(64, len(buf)))
                batch = torch.cat(sample_tensors, dim=0)
                if batch.size(0) > 64:
                    idx = torch.randperm(batch.size(0), device=batch.device)[:64]
                    batch = batch[idx]
                batch = batch.to(device)
                seed.train_child_step(batch, epoch=epoch)
        seed.update_blending(epoch)
        seed.update_shadowing(epoch)
        seed.update_probationary(epoch)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    seed_manager: "SeedManager",
    device: torch.device,
    scheduler: LRScheduler | None = None,
    epoch: int | None = None,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad(set_to_none=True)
        preds = model(X)
        loss = criterion(preds, y)
        if loss.requires_grad:
            loss.backward()
            optimiser.step()

        total_loss += loss.item()
        handle_seed_training(seed_manager, device, epoch)

    if scheduler:
        scheduler.step()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Evaluate the model and return (loss, accuracy)."""
    model.eval()
    loss_accum, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss_accum += criterion(preds, y).item()
        correct += (preds.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_accum / len(loader), correct / total


def execute_phase_1(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: ExperimentLogger,
    tb_writer: "SummaryWriter",
    device: torch.device,
    config: dict[str, Any],
    seed_manager: SeedManager,
    tamiyo: KasminaMicro,
) -> dict[str, Any]:
    """
    Runs the initial warm-up phase, reporting all progress via the logger.
    """
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 20, 0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    best_acc = 0.0

    logger.log_phase_update(
        epoch=0,
        from_phase="Init",
        to_phase="Warm-up",
        total_epochs_in_phase=config["warm_up_epochs"],
    )

    for epoch in range(1, config["warm_up_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, seed_manager, device, scheduler, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_acc = max(best_acc, val_acc)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
        }
        logger.log_metrics_update(epoch, metrics)

        # Assess seeds and log their state and metrics
        # This single call now handles all detailed seed logging.
        if hasattr(tamiyo, "assess_and_update_seeds"):
            tamiyo.assess_and_update_seeds(epoch)

        # Update other loggers
        tb_writer.add_scalar("train/loss_phase1", train_loss, epoch)
        tb_writer.add_scalar("validation/loss_phase1", val_loss, epoch)
        tb_writer.add_scalar("validation/accuracy_phase1", val_acc, epoch)
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
                step=epoch,
            )

    return {"best_acc": best_acc}


def execute_phase_2(
    model: nn.Module,
    seed_manager: SeedManager,
    karn: Any,  # Placeholder for Karn agent
    tamiyo: KasminaMicro,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: ExperimentLogger,
    tb_writer: "SummaryWriter",
    device: torch.device,
    config: dict[str, Any],
    final_stats: dict[str, Any],
) -> Dict[str, Any]:
    """
    Runs the adaptation phase, reporting all progress via the logger.
    """
    warm_up_epochs = config["warm_up_epochs"]
    adaptation_epochs = config["adaptation_epochs"]
    criterion = nn.CrossEntropyLoss().to(device)

    model.freeze_backbone()

    def rebuild_opt(m):
        params = [p for p in m.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=config["lr"] * 0.1, weight_decay=config.get("weight_decay", 1e-4)) if params else None

    optimiser = rebuild_opt(model)

    # Carry over stats from phase 1
    best_acc = final_stats.get("best_acc", 0.0)
    acc_pre, acc_post, t_recover, germ_epoch = None, None, None, None
    seeds_activated = False

    logger.log_phase_update(
        epoch=warm_up_epochs,
        from_phase="Warm-up",
        to_phase="Adaptation",
        total_epochs_in_phase=adaptation_epochs,
    )

    for epoch in range(warm_up_epochs + 1, warm_up_epochs + adaptation_epochs + 1):
        train_loss = 0.0
        if optimiser:
            train_loss = train_epoch(model, train_loader, optimiser, criterion, seed_manager, device, epoch=epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_acc = max(best_acc, val_acc)

        # Assess seeds BEFORE the step function, so Tamiyo has the latest state
        if hasattr(tamiyo, "assess_and_update_seeds"):
            tamiyo.assess_and_update_seeds(epoch)

        if not seeds_activated and tamiyo.step(epoch, val_loss, val_acc):
            seeds_activated, germ_epoch, acc_pre = True, epoch, val_acc
            optimiser = rebuild_opt(model)

        if germ_epoch:
            if epoch == germ_epoch + 1:
                acc_post = val_acc
            if t_recover is None and acc_pre is not None and val_acc >= acc_pre:
                t_recover = epoch - germ_epoch

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
        }
        logger.log_metrics_update(epoch, metrics)

        # NOTE: The comprehensive seed state update is now handled by
        # tamiyo.assess_and_update_seeds() called earlier in the loop.

        # Update other loggers
        tb_writer.add_scalar("train/loss_phase2", train_loss, epoch)
        tb_writer.add_scalar("validation/loss_phase2", val_loss, epoch)
        tb_writer.add_scalar("validation/accuracy_phase2", val_acc, epoch)
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics(
                {"train_loss_p2": train_loss, "val_loss_p2": val_loss, "val_acc_p2": val_acc},
                step=epoch,
            )

    # Combine results
    final_stats.update(
        {
            "best_acc": best_acc,
            "accuracy_dip": (acc_pre - acc_post) if acc_pre is not None and acc_post is not None else None,
            "recovery_time": t_recover,
            "seeds_activated": seeds_activated,
            "acc_pre": acc_pre,
            "acc_post": acc_post,
        }
    )
    return final_stats


def clear_seed_report_cache():
    """Clear the global seed report cache. Useful for test isolation."""
    _last_report.clear()

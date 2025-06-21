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


def _get_seed_training_batch(info: SeedInfo, device: torch.device) -> torch.Tensor | None:
    """Get a training batch from a seed's buffer if available."""
    buf = info["buffer"]
    if len(buf) < 10:  # Require a minimum buffer size to train
        return None

    sample_tensors = random.sample(list(buf), min(64, len(buf)))
    batch = torch.cat(sample_tensors, dim=0)
    if batch.size(0) > 64:
        idx = torch.randperm(batch.size(0), device=batch.device)[:64]
        batch = batch[idx]
    return batch.to(device)


def _perform_per_step_seed_updates(
    seed_manager: "SeedManager", device: torch.device, epoch: int | None
):
    """Handle continuous, per-step seed activities like training and grafting."""
    for info in seed_manager.seeds.values():
        seed = info["module"]
        if epoch is not None:
            info["last_epoch"] = epoch

        # Match on the seed's state to determine the primary action.
        match seed.state:
            case SeedState.TRAINING.value:
                batch = _get_seed_training_batch(info, device)
                if batch is not None and hasattr(seed, "train_child_step"):
                    seed.train_child_step(batch, epoch=epoch)
            case SeedState.FINE_TUNING.value:
                if hasattr(seed, "perform_fine_tuning_step"):
                    seed.perform_fine_tuning_step(epoch=epoch)

        # Grafting is a continuous process that can happen alongside other states.
        if hasattr(seed, "update_grafting"):
            seed.update_grafting(epoch)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    seed_manager: "SeedManager",
    device: torch.device,
    scheduler: LRScheduler | None = None,
    epoch: int | None = None,
    logger: ExperimentLogger | None = None,  # logger for per-batch progress
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    steps_total = len(loader)
    # roughly 1% of steps, but at least every batch if <100
    update_interval = max(1, steps_total // 100)
    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad(set_to_none=True)
        # Pass labels to the model for fine-tuning data collection
        preds = model(X, y)
        loss = criterion(preds, y)
        if loss.requires_grad:
            loss.backward()
            optimiser.step()

        total_loss += loss.item()

        # Handle continuous, per-step seed activities
        _perform_per_step_seed_updates(seed_manager, device, epoch)

        # Emit progress every ~1% of steps, or every batch if <100, and always on last batch
        if logger is not None and epoch is not None and (
            (i + 1) % update_interval == 0 or (i + 1) == steps_total
        ):
            logger.log_step_update(epoch, i + 1, steps_total)

    # After all batches for an epoch are done, major state transitions
    # are handled by KasminaMicro.assess_and_update_seeds.

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
        # Pass labels to the model to ensure consistent forward pass
        preds = model(X, y)
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

    # Set the validation loader and loss function on the model for fine-tuning evaluation
    if hasattr(model, "val_loader"):
        model.val_loader = val_loader
    if hasattr(model, "loss_function"):
        model.loss_function = criterion

    logger.log_phase_update(
        epoch=0,
        from_phase="Init",
        to_phase="Warm-up",
        total_epochs_in_phase=config["warm_up_epochs"],
    )

    for epoch in range(1, config["warm_up_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, seed_manager, device, scheduler, epoch, logger)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_acc = max(best_acc, val_acc)
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
            "steps_per_epoch": len(train_loader),
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


def _rebuild_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer | None:
    """Rebuilds the optimizer to only include trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None
    return torch.optim.AdamW(params, lr=config["lr"] * 0.1, weight_decay=config.get("weight_decay", 1e-4))


def _update_post_germination_stats(
    epoch: int,
    germ_epoch: int,
    acc_pre: float,
    val_acc: float,
    acc_post: float | None,
    t_recover: int | None,
) -> tuple[float | None, int | None]:
    """Updates accuracy dip and recovery time stats after seed germination."""
    if epoch == germ_epoch + 1:
        acc_post = val_acc
    if t_recover is None and acc_pre is not None and val_acc >= acc_pre:
        t_recover = epoch - germ_epoch
    return acc_post, t_recover


def _log_phase2_metrics(
    epoch: int,
    metrics: dict[str, Any],
    logger: ExperimentLogger,
    tb_writer: "SummaryWriter",
):
    """Logs all relevant metrics for a single epoch in phase 2."""
    logger.log_metrics_update(epoch, metrics)
    tb_writer.add_scalar("train/loss_phase2", metrics["train_loss"], epoch)
    tb_writer.add_scalar("validation/loss_phase2", metrics["val_loss"], epoch)
    tb_writer.add_scalar("validation/accuracy_phase2", metrics["val_acc"], epoch)
    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_metrics(
            {"train_loss_p2": metrics["train_loss"], "val_loss_p2": metrics["val_loss"], "val_acc_p2": metrics["val_acc"]},
            step=epoch,
        )


def execute_phase_2(
    model: nn.Module,
    seed_manager: SeedManager,
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
    adaptation_epochs = config["adaptation_epochs"]
    criterion = nn.CrossEntropyLoss().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 1e-4)
    )

    # Initialize tracking variables
    best_acc = final_stats.get("best_acc", 0.0)
    acc_pre, acc_post, t_recover, germ_epoch = None, None, None, None
    seeds_activated = False

    logger.log_phase_update(
        epoch=config["warm_up_epochs"],
        from_phase="Warm-up",
        to_phase="Adaptation",
        total_epochs_in_phase=adaptation_epochs,
    )

    for epoch_offset in range(1, adaptation_epochs + 1):
        epoch = config["warm_up_epochs"] + epoch_offset
        # Train for one epoch first.
        train_loss = train_epoch(model, train_loader, optimiser, criterion, seed_manager, device, epoch=epoch, logger=logger)
        # Now, evaluate performance *after* training.
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_acc = max(best_acc, val_acc)

        # Let Tamiyo decide if it's time to activate seeds based on the *current* performance.
        if not seeds_activated and tamiyo.step(epoch, val_loss, val_acc):
            seeds_activated, germ_epoch, acc_pre = True, epoch, val_acc

        # Assess seeds and handle all other epoch-level state transitions.
        if hasattr(tamiyo, "assess_and_update_seeds"):
            tamiyo.assess_and_update_seeds(epoch)

        # NEW: After assessing states, try to start the next training job.
        if hasattr(tamiyo, "start_training_next_seed"):
            tamiyo.start_training_next_seed(epoch)

        if germ_epoch:
            acc_post, t_recover = _update_post_germination_stats(
                epoch, germ_epoch, acc_pre, val_acc, acc_post, t_recover
            )

        metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "best_acc": best_acc, "steps_per_epoch": len(train_loader)}
        _log_phase2_metrics(epoch, metrics, logger, tb_writer)

    # Combine and return final results
    final_stats.update(
        {"best_acc": best_acc}
    )
    if acc_post is not None:
        final_stats.update({"acc_post_germination": acc_post, "time_to_recover": t_recover})
    return final_stats


def clear_seed_report_cache():
    """Clear the global seed report cache. Useful for test isolation."""
    _last_report.clear()

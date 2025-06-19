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

# Check if we're in testing mode to conditionally disable MLflow
TESTING_MODE = "pytest" in sys.modules or "unittest" in sys.modules
MLFLOW_AVAILABLE = not TESTING_MODE

# Global variable to track seed update reporting
_last_report: Dict[str, str] = defaultdict(lambda: "")


def handle_seed_training(seed_manager: "SeedManager", device: torch.device):
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
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    seed_manager: "SeedManager",
    device: torch.device,
    scheduler: LRScheduler | None = None,
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
        handle_seed_training(seed_manager, device)

    if scheduler:
        scheduler.step()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
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


def log_seed_updates(epoch: int, seed_manager: "SeedManager", logger: "ExperimentLogger", tb_writer: "SummaryWriter"):
    """
    REFACTORED: Checks and logs all seed state transitions through the logger.
    This is now the single source of truth for seed status reporting.
    """
    for sid, info in seed_manager.seeds.items():
        module = info["module"]
        current_state = module.state
        current_alpha = getattr(module, "alpha", 0.0)
        
        # Create a unique tag for the current state to detect changes
        current_tag = f"{current_state}:{current_alpha:.3f}" if current_state == "blending" else current_state
        last_tag = _last_report[sid]

        if current_tag != last_tag:
            from_state = last_tag.split(":")[0] if ":" in last_tag else last_tag
            if from_state == "": from_state = "init"
            
            # Use the correct, specific logger methods
            if current_state == "blending":
                logger.log_blending_progress(epoch, sid, current_alpha)
            else:
                logger.log_seed_event(epoch, sid, from_state, current_state)
            
            # Also log to TensorBoard
            if current_state == "blending":
                tb_writer.add_scalar(f"seed/{sid}/alpha", current_alpha, epoch)
            else:
                tb_writer.add_text(f"seed/{sid}/events", f"Epoch {epoch}: {from_state} → {current_state}", epoch)

            _last_report[sid] = current_tag


def execute_phase_1(
    config: Dict[str, Any], model: nn.Module, loaders: tuple, loss_fn: nn.Module, 
    seed_manager: "SeedManager", logger: "ExperimentLogger", tb_writer: "SummaryWriter", log_f
) -> float:
    """
    REFACTORED: Runs the initial warm-up phase, reporting all progress via the logger.
    The `dashboard` parameter has been removed.
    """
    train_loader, val_loader = loaders
    device = config["device"]
    optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 20, 0.1)
    best_acc = 0.0

    for epoch in range(1, config["warm_up_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimiser, loss_fn, seed_manager, device, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        best_acc = max(best_acc, val_acc)

        metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "best_acc": best_acc}
        
        # <<< CHANGE: Single call to the logger handles all UI and file logging.
        logger.log_epoch_progress(epoch, metrics)
        
        log_seed_updates(epoch, seed_manager, logger, tb_writer)

        # Log to other platforms
        tb_writer.add_scalar("train/loss_phase1", train_loss, epoch)
        tb_writer.add_scalar("validation/loss_phase1", val_loss, epoch)
        tb_writer.add_scalar("validation/accuracy_phase1", val_acc, epoch)
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)

    return best_acc


def execute_phase_2(
    config: Dict[str, Any], model: nn.Module, loaders: tuple, loss_fn: nn.Module,
    seed_manager: "SeedManager", kasmina, logger: "ExperimentLogger", tb_writer: "SummaryWriter", 
    log_f, initial_best_acc: float
) -> Dict[str, Any]:
    """
    REFACTORED: Runs the adaptation phase, reporting all progress via the logger.
    The `dashboard` parameter has been removed.
    """
    train_loader, val_loader = loaders
    device = config["device"]
    warm_up_epochs = config["warm_up_epochs"]
    adaptation_epochs = config["adaptation_epochs"]
    
    model.freeze_backbone()
    
    def rebuild_opt(m):
        params = [p for p in m.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=config["lr"] * 0.1) if params else None
        
    optimiser = rebuild_opt(model)
    best_acc, acc_pre, acc_post, t_recover, germ_epoch = initial_best_acc, None, None, None, None
    seeds_activated = False

    for epoch in range(warm_up_epochs + 1, warm_up_epochs + adaptation_epochs + 1):
        train_loss = 0.0
        if optimiser:
            train_loss = train_epoch(model, train_loader, optimiser, loss_fn, seed_manager, device)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        best_acc = max(best_acc, val_acc)

        if not seeds_activated and kasmina.step(val_loss, val_acc):
            seeds_activated, germ_epoch, acc_pre = True, epoch, val_acc
            logger.log_germination(epoch, kasmina.get_last_germinated_seed_id())
            optimiser = rebuild_opt(model)
        
        if germ_epoch:
            if epoch == germ_epoch + 1: acc_post = val_acc
            if t_recover is None and acc_pre is not None and val_acc >= acc_pre:
                t_recover = epoch - germ_epoch

        metrics = {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "best_acc": best_acc}
        
        # <<< CHANGE: Single call to the logger handles all UI and file logging.
        logger.log_epoch_progress(epoch, metrics)
        
        log_seed_updates(epoch, seed_manager, logger, tb_writer)

        # Log to other platforms
        tb_writer.add_scalar("train/loss_phase2", train_loss, epoch)
        tb_writer.add_scalar("validation/loss_phase2", val_loss, epoch)
        tb_writer.add_scalar("validation/accuracy_phase2", val_acc, epoch)
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics({"train_loss_p2": train_loss, "val_loss_p2": val_loss, "val_acc_p2": val_acc}, step=epoch)

    return {
        "best_acc": best_acc,
        "accuracy_dip": (acc_pre - acc_post) if acc_pre is not None and acc_post is not None else None,
        "recovery_time": t_recover, "seeds_activated": seeds_activated,
        "acc_pre": acc_pre, "acc_post": acc_post,
    }

def clear_seed_report_cache():
    """Clear the global seed report cache. Useful for test isolation."""
    _last_report.clear()
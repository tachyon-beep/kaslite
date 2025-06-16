"""
Training and evaluation utilities for morphogenetic architecture experiments.

This module provides functions for training neural networks, evaluating performance,
and executing different phases of morphogenetic experiments.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from morphogenetic_engine.core import SeedManager


# Global variable to track seed update reporting
_last_report: dict[str, Optional[str]] = defaultdict(lambda: None)


def handle_seed_training(seed_manager: SeedManager, device: torch.device):
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


def _update_dashboard_seed_state(dashboard, sid, mod):
    """Update dashboard with seed state if dashboard is available."""
    if dashboard:
        alpha_val = getattr(mod, "alpha", 0.0)
        dashboard.update_seed(sid, mod.state, alpha_val)


def _log_seed_state_change(epoch, sid, mod, prev, logger, tb_writer):
    """Log seed state changes to experiment logger and TensorBoard."""
    if mod.state == "blending":
        logger.log_blending_progress(epoch, sid, mod.alpha)
        tb_writer.add_scalar(f"seed/{sid}/alpha", mod.alpha, epoch)
    else:
        prev_state = prev.split(":")[0] if prev and ":" in prev else (prev or "unknown")
        logger.log_seed_event(epoch, sid, prev_state, mod.state)
        tb_writer.add_text(
            f"seed/{sid}/events",
            f"Epoch {epoch}: {prev_state} → {mod.state}",
            epoch
        )


def log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f, dashboard=None):
    """Handle the repetitive logic of checking and logging seed state transitions."""
    for sid, info in seed_manager.seeds.items():
        mod = info["module"]
        prev = _last_report[sid]

        should_log, tag = should_log_seed_update(mod, prev)

        if should_log:
            _last_report[sid] = tag

            _update_dashboard_seed_state(dashboard, sid, mod)
            _log_seed_state_change(epoch, sid, mod, prev, logger, tb_writer)

            alpha_str = format_alpha_value(getattr(mod, "alpha", 0.0))
            log_f.write(f"{epoch},{sid},{mod.state},{alpha_str}\n")


def execute_phase_1(config, model, loaders, loss_fn, seed_manager, logger, tb_writer, log_f, dashboard=None):
    """Run the initial warm-up training phase."""
    train_loader, val_loader = loaders
    device = next(model.parameters()).device

    # Initialize optimizer and scheduler for the full model
    optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 20, 0.1)

    best_acc = 0.0
    warm_up_epochs = config["warm_up_epochs"]

    # Start phase 1 in dashboard if available
    if dashboard:
        dashboard.start_phase("phase_1", warm_up_epochs, "🔥 Warm-up Training")

    for epoch in range(1, warm_up_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, loss_fn, seed_manager, scheduler, device)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        best_acc = max(best_acc, val_acc)

        # Prepare metrics for logging
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
        }

        logger.log_epoch_progress(epoch, metrics)

        # Update dashboard
        if dashboard:
            dashboard.update_progress(epoch, metrics)

        # Log to TensorBoard
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("validation/loss", val_loss, epoch)
        tb_writer.add_scalar("validation/accuracy", val_acc, epoch)
        tb_writer.add_scalar("validation/best_acc", best_acc, epoch)

        log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f, dashboard)

    return best_acc


def handle_germination_tracking(epoch, germ_epoch, acc_pre, acc_post, val_acc, t_recover):
    """Handle the tracking of germination events and recovery time."""
    if germ_epoch and epoch == germ_epoch + 1:
        acc_post = val_acc
    if germ_epoch and t_recover is None and acc_pre is not None and val_acc >= acc_pre:
        t_recover = epoch - germ_epoch
    return acc_post, t_recover


def _setup_phase_2_optimizer(lr: float):
    """Create and return optimizer and scheduler for phase 2."""
    def rebuild_seed_opt(model):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return None, None
        opt = torch.optim.Adam(params, lr=lr * 0.1, weight_decay=0.0)
        sch = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1)
        return opt, sch
    return rebuild_seed_opt


def _handle_germination_step(kasmina, val_loss, val_acc, epoch, dashboard, seed_manager, rebuild_opt_fn, model):
    """Handle germination logic and return updated state."""
    seeds_activated = False
    germ_epoch = None
    acc_pre = None
    optimiser = None
    scheduler = None
    
    if kasmina.step(val_loss, val_acc):
        seeds_activated = True
        germ_epoch, acc_pre = epoch, val_acc
        optimiser, scheduler = rebuild_opt_fn(model)
        
        # Show germination event in dashboard
        if dashboard:
            # Find which seed(s) just became active
            for sid, info in seed_manager.seeds.items():
                if info["module"].state in ["blending", "active"]:
                    dashboard.show_germination_event(sid, epoch)
                    break
    
    return seeds_activated, germ_epoch, acc_pre, optimiser, scheduler


def _log_phase_2_metrics(epoch, metrics, logger, tb_writer, train_loss, dashboard, warm_up_epochs):
    """Log metrics for phase 2."""
    logger.log_epoch_progress(epoch, metrics)

    # Update dashboard
    if dashboard:
        # Convert epoch to phase-relative epoch for progress bar
        phase_epoch = epoch - warm_up_epochs
        dashboard.update_progress(phase_epoch, metrics)

    # Log to TensorBoard
    if train_loss > 0:  # Only log train loss if we actually trained
        tb_writer.add_scalar("train/loss", train_loss, epoch)
    tb_writer.add_scalar("validation/loss", metrics["val_loss"], epoch)
    tb_writer.add_scalar("validation/accuracy", metrics["val_acc"], epoch)
    tb_writer.add_scalar("validation/best_acc", metrics["best_acc"], epoch)


def execute_phase_2(
    config, model, loaders, loss_fn, seed_manager, kasmina, logger, tb_writer, log_f, initial_best_acc, dashboard=None
):
    """Run the adaptation phase where the backbone is frozen and seeds can germinate."""
    train_loader, val_loader = loaders
    device = next(model.parameters()).device
    warm_up_epochs = config["warm_up_epochs"]
    adaptation_epochs = config["adaptation_epochs"]
    lr = config["lr"]

    # Freeze the model's backbone
    model.freeze_backbone()

    # Start phase 2 in dashboard if available
    if dashboard:
        dashboard.start_phase("phase_2", adaptation_epochs, "🧬 Adaptation Phase")

    # Setup optimizer rebuilding function
    rebuild_opt_fn = _setup_phase_2_optimizer(lr)
    optimiser, scheduler = rebuild_opt_fn(model)
    
    # Initialize state variables
    best_acc = initial_best_acc
    acc_pre = acc_post = t_recover = germ_epoch = None
    seeds_activated = False

    for epoch in range(warm_up_epochs + 1, warm_up_epochs + adaptation_epochs + 1):
        # Training step
        train_loss = 0.0
        if optimiser:
            train_loss = train_epoch(model, train_loader, optimiser, loss_fn, seed_manager, scheduler, device)

        # Evaluation step
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        # Germination check
        step_seeds_activated, step_germ_epoch, step_acc_pre, new_opt, new_sch = _handle_germination_step(
            kasmina, val_loss, val_acc, epoch, dashboard, seed_manager, rebuild_opt_fn, model
        )
        
        if step_seeds_activated:
            seeds_activated = True
            germ_epoch, acc_pre = step_germ_epoch, step_acc_pre
            optimiser, scheduler = new_opt, new_sch

        # Recovery tracking
        acc_post, t_recover = handle_germination_tracking(epoch, germ_epoch, acc_pre, acc_post, val_acc, t_recover)

        # Update best accuracy
        if epoch % 10 == 0 or val_acc > best_acc:
            best_acc = max(best_acc, val_acc)
        
        # Prepare and log metrics
        status = ", ".join(f"{sid}:{info['status']}" for sid, info in seed_manager.seeds.items())
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": best_acc,
            "seeds": status,
        }
        
        _log_phase_2_metrics(epoch, metrics, logger, tb_writer, train_loss, dashboard, warm_up_epochs)
        log_seed_updates(epoch, seed_manager, logger, tb_writer, log_f, dashboard)

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


def clear_seed_report_cache():
    """Clear the global seed report cache. Useful for test isolation."""
    _last_report.clear()

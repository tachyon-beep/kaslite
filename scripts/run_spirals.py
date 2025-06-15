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
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Default device - can be overridden for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_device_for_testing(test_device: str = "cpu"):
    """Override device for testing purposes."""
    global device
    device = torch.device(test_device)

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LRScheduler

from morphogenetic_engine.core import SeedManager, KasminaMicro
from morphogenetic_engine.components import BaseNet

# Default device - can be overridden for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_last_report: Dict[str, Optional[str]] = defaultdict(lambda: None)


# ---------- DATA -------------------------------------------------------------

def create_spirals(n_samples: int = 2000, noise: float = 0.25, rotations: int = 4):
    """Generate the classic two-spirals toy dataset."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    n = np.sqrt(rng.random(n_samples // 2)) * rotations * 2 * np.pi
    d1x = np.cos(n) * n + rng.random(n_samples // 2) * noise
    d1y = np.sin(n) * n + rng.random(n_samples // 2) * noise

    X = np.vstack((np.hstack((d1x, -d1x)), np.hstack((d1y, -d1y)))).T
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    return X.astype(np.float32), y.astype(np.int64)


# ---------- TRAIN / EVAL HELPERS ---------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    seed_manager: SeedManager,
    scheduler: Optional[LRScheduler] = None,
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
    model: nn.Module, loader: DataLoader, criterion: nn.Module
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
    """Main function to run the spirals experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend_steps", type=int, default=30)
    parser.add_argument("--shadow_lr", type=float, default=1e-3)
    parser.add_argument("--progress_thresh", type=float, default=0.6)
    parser.add_argument(
        "--drift_warn", type=float, default=0.12,
        help="Drift warning threshold (0=disable)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ---------------- hyper-parameters ----------------
    warm_up_epochs    = 50
    adaptation_epochs = 200
    lr                = 3e-3
    hidden_dim        = 128
    acc_threshold     = 0.95
    # --------------------------------------------------

    print("=== Morphogenetic Architecture Experiment ===")
    print(f"Warm-up   : {warm_up_epochs} epochs")
    print(f"Seed phase: {adaptation_epochs} epochs")
    print(f"LR        : {lr}")
    print(f"Hidden dim: {hidden_dim}\n")

    slug = (
        f"h{hidden_dim}_bs{args.blend_steps}"
        f"_lr{args.shadow_lr}_pt{args.progress_thresh}"
    )

    # ---- open log file with context-manager ----
    with Path(f"results_{slug}.log").open("w", encoding="utf-8") as log_f:
        _last_report.clear()
        log_f.write("epoch,seed,state,alpha\n")

        # ---------- data ----------
        X, y = create_spirals()
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X).astype(np.float32)

        dataset  = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=128, num_workers=0)

        # ---------- model & agents ----------
        seed_manager = SeedManager()
        model = BaseNet(
            hidden_dim,
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
        print("----- Phase 1 : full-model training -----")
        for epoch in range(1, warm_up_epochs + 1):
            train_epoch(model, train_loader, optimiser, loss_fn, seed_manager, scheduler)

            val_loss, val_acc = evaluate(model, val_loader, loss_fn)
            best_acc = max(best_acc, val_acc)

            if epoch % 5 == 0 or epoch in (1, warm_up_epochs):
                print(f"Ep {epoch:>2}: loss {val_loss:.4f}  acc {val_acc:.4f}  best {best_acc:.4f}")

            for sid, info in seed_manager.seeds.items():
                mod = info["module"]
                tag = f"{mod.state}:{mod.alpha:.2f}" if mod.state == "blending" else mod.state
                prev = _last_report[sid]
                should = (
                    (prev != tag) or
                    (mod.state == "blending" and prev and ":" in prev and
                     float(prev.split(":")[1]) + 0.1 <= mod.alpha)
                )
                if should:
                    _last_report[sid] = tag
                    logging.info("epoch %d %s %s", epoch, sid, tag)
                    log_f.write(f"{epoch},{sid},{mod.state},{mod.alpha:.3f}\n")

        print(f"\nWarm-up complete — trunk frozen at {best_acc:.4f} accuracy\n")
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
        print("----- Phase 2 : seed adaptation -----")
        for epoch in range(warm_up_epochs + 1, warm_up_epochs + adaptation_epochs + 1):
            if optimiser:
                train_epoch(model, train_loader, optimiser, loss_fn, seed_manager, scheduler)

            val_loss, val_acc = evaluate(model, val_loader, loss_fn)

            if kasmina.step(val_loss, val_acc):
                seeds_activated = True
                germ_epoch, acc_pre = epoch, val_acc
                print(f"[!] Germination at epoch {epoch}")
                optimiser, scheduler = rebuild_seed_opt()

            if germ_epoch and epoch == germ_epoch + 1:
                acc_post = val_acc
            if germ_epoch and t_recover is None and acc_pre is not None and val_acc >= acc_pre:
                t_recover = epoch - germ_epoch

            if epoch % 10 == 0 or val_acc > best_acc:
                best_acc = max(best_acc, val_acc)
                status = ", ".join(
                    f"{sid}:{info['status']}" for sid, info in seed_manager.seeds.items()
                )
                print(
                    f"Ep {epoch:>3}: loss {val_loss:.4f}  acc {val_acc:.4f}  "
                    f"best {best_acc:.4f}  seeds [{status}]"
                )

            for sid, info in seed_manager.seeds.items():
                mod = info["module"]
                tag = f"{mod.state}:{mod.alpha:.2f}" if mod.state == "blending" else mod.state
                prev = _last_report[sid]
                should = (
                    (prev != tag) or
                    (mod.state == "blending" and prev and ":" in prev and
                     float(prev.split(":")[1]) + 0.1 <= mod.alpha)
                )
                if should:
                    _last_report[sid] = tag
                    logging.info("epoch %d %s %s", epoch, sid, tag)
                    alpha_str = f"{mod.alpha:.3f}" if mod.state == "blending" else ""
                    log_f.write(f"{epoch},{sid},{mod.state},{alpha_str}\n")

        # ------------- final stats -------------
        print("\n===== Final =====")
        print(f"Best accuracy: {best_acc:.4f}")
        if seeds_activated:
            print("Seed events:")
            for ev in seed_manager.germination_log:
                t = time.strftime("%H:%M:%S", time.localtime(ev["timestamp"]))
                if "success" in ev:
                    print(f"  {ev['seed_id']} germination {'✓' if ev['success'] else '✗'} @ {t}")
                else:
                    print(f"  {ev['seed_id']} {ev['from']}→{ev['to']} @ {t}")

        if acc_pre is not None and acc_post is not None:
            logging.info("accuracy dip %.3f, recovery %s epochs", acc_pre - acc_post, t_recover)


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()

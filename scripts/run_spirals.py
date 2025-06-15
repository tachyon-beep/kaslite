"""
Run a morphogenetic-architecture experiment on the two-spirals dataset.

• Phase 1 – train the full network for `warm_up_epochs`
• Phase 2 – freeze the trunk, let Kasmina germinate seeds on a plateau
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from morphogenetic_engine.core import SeedManager, KasminaMicro
from morphogenetic_engine.components import BaseNet


# ---------- DATA -------------------------------------------------------------

def create_spirals(n_samples: int = 2000, noise: float = 0.25, rotations: int = 4):
    """Generate the classic two-spirals toy dataset."""
    n = np.sqrt(np.random.rand(n_samples // 2)) * rotations * 2 * np.pi
    d1x = np.cos(n) * n + np.random.rand(n_samples // 2) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples // 2) * noise

    X = np.vstack((np.hstack((d1x, -d1x)), np.hstack((d1y, -d1y)))).T
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    return X.astype(np.float32), y.astype(np.int64)


# ---------- TRAIN / EVAL HELPERS ---------------------------------------------

def train_epoch(model: nn.Module,
               loader: DataLoader,
               optimiser: torch.optim.Optimizer,
               criterion: nn.Module):
    model.train()
    for X, y in loader:
        optimiser.zero_grad(set_to_none=True)
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimiser.step()


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    loss_accum, correct, total = 0.0, 0, 0
    for X, y in loader:
        preds = model(X)
        loss_accum += criterion(preds, y).item()
        correct += (preds.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_accum / len(loader), correct / total


# ---------- MAIN -------------------------------------------------------------

def main():
    # ---------------- hyper-parameters ----------------
    warm_up_epochs    = 50
    adaptation_epochs = 200
    lr                = 3e-3          # calmer than 0.01
    hidden_dim        = 128
    acc_threshold     = 0.95
    # --------------------------------------------------

    print("=== Morphogenetic Architecture Experiment ===")
    print(f"Warm-up  : {warm_up_epochs} epochs")
    print(f"Seed phase : {adaptation_epochs} epochs")
    print(f"LR        : {lr}")
    print(f"Hidden dim: {hidden_dim}\n")

    # ---------- data ----------
    X, y = create_spirals()
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype(np.float32)

    dataset   = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128)
    # ---------------------------

    # ---------- model & agents ----------
    seed_manager = SeedManager()
    model   = BaseNet(hidden_dim)              # SentinelSeed is widened in components.py
    loss_fn = nn.CrossEntropyLoss()
    kasmina = KasminaMicro(seed_manager,
                           patience=15,
                           delta=5e-4,
                           acc_threshold=acc_threshold)
    # -------------------------------------

    # ---------- optimiser & scheduler (phase 1) ----------
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 20, 0.1)
    # -----------------------------------------------------

    best_acc = 0.0

    print("----- Phase 1 : full-model training -----")
    for epoch in range(1, warm_up_epochs + 1):
        train_epoch(model, train_loader, optimiser, loss_fn)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        best_acc = max(best_acc, val_acc)

        if epoch % 5 == 0 or epoch == 1 or epoch == warm_up_epochs:
            print(f"Ep {epoch:>2}: loss {val_loss:.4f}  acc {val_acc:.4f}  best {best_acc:.4f}")

    print(f"\nWarm-up complete — trunk frozen at {best_acc:.4f} accuracy\n")
    model.freeze_backbone()

    # ---------- switch to seed-only training ----------
    def rebuild_seed_opt():
        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            opt  = torch.optim.Adam(params, lr=lr * 0.1)  # cooler LR for seeds
            sch  = torch.optim.lr_scheduler.StepLR(opt, 20, 0.1)
            return opt, sch
        return None, None

    optimiser, scheduler = rebuild_seed_opt()
    # ---------------------------------------------------

    seeds_activated = False

    print("----- Phase 2 : seed adaptation -----")
    for epoch in range(warm_up_epochs + 1,
                       warm_up_epochs + adaptation_epochs + 1):

        if optimiser:
            train_epoch(model, train_loader, optimiser, loss_fn)
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        # germination check
        if kasmina.step(val_loss, val_acc):
            seeds_activated = True
            print(f"[!] Germination at epoch {epoch}")
            optimiser, scheduler = rebuild_seed_opt()

        if epoch % 10 == 0 or val_acc > best_acc:
            best_acc = max(best_acc, val_acc)
            status = ", ".join(f"{sid}:{info['status'][0]}"
                               for sid, info in seed_manager.seeds.items())
            print(f"Ep {epoch:>3}: loss {val_loss:.4f}  acc {val_acc:.4f} "
                  f"best {best_acc:.4f}  seeds [{status}]")

    print("\n===== Final =====")
    print(f"Best accuracy: {best_acc:.4f}")
    if seeds_activated:
        print("Seed events:")
        for ev in seed_manager.germination_log:
            t = time.strftime('%H:%M:%S', time.localtime(ev['timestamp']))
            print(f"  {ev['seed_id']} – {'OK' if ev['success'] else 'FAIL'} at {t}")


# ---------- entry-point guard ----------

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    main()

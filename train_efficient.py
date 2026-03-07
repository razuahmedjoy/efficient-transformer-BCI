"""
EfficientEEGEncoder — Unified Training Script
==============================================
Supports subject-dependent training, LOSO (subject-independent) evaluation,
checkpoint save/resume, cosine annealing with warmup, and mixed precision.

COLAB USAGE:
  1. Upload this file + efficient_eegencoder.py to Google Drive
  2. Mount drive:  from google.colab import drive; drive.mount('/content/drive')
  3. Preprocess:   !python /content/drive/MyDrive/eegencoder/colab_preprocess_gdf.py
  4. Train SD:     !python /content/drive/MyDrive/eegencoder/train_efficient.py --mode sd
  5. Train LOSO:   !python /content/drive/MyDrive/eegencoder/train_efficient.py --mode loso

  If Colab disconnects, re-run the same command — it auto-resumes from checkpoint.

GPU RECOMMENDATION (Colab Pro):
  Best:   A100 (40GB) — fastest, ~2.5h for full SD training
  Good:   V100 (16GB) — fast, ~3.5h for full SD training
  OK:     T4   (16GB) — slower but works, ~5h for full SD training
  Select: Runtime → Change runtime type → GPU → (pick A100 or V100)
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efficient_eegencoder import EfficientEEGEncoder, count_parameters
try:
    from alignment import align_all_subjects
except ImportError:
    align_all_subjects = None


# ===================================================================
# Configuration
# ===================================================================

DEFAULT_CONFIG = {
    # ----- Paths (override for Colab/Kaggle) -----
    "data_dir": "/content/drive/MyDrive/eegencoder/data/",
    "checkpoint_dir": "/content/drive/MyDrive/eegencoder/checkpoints_v2/",
    "results_dir": "/content/drive/MyDrive/eegencoder/results_v2/",

    # ----- Model -----
    "n_classes": 4,
    "n_channels": 22,
    "n_timepoints": 1125,
    "n_branches": 5,
    "hidden_size": 32,
    "num_heads": 2,
    "num_transformer_layers": 2,
    "intermediate_size": 32,
    "tcn_depth": 2,
    "tcn_kernel_size": 4,
    "tcn_filters": 32,
    "dropout": 0.3,
    "use_gradient_checkpoint": False,

    # ----- Training -----
    "n_subjects": 9,
    "n_epochs": 500,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "warmup_epochs": 10,
    "label_smoothing": 0.1,
    "grad_clip_norm": 1.0,
    "loss_scale_factor": 2.0,

    # ----- Checkpointing -----
    "checkpoint_every": 50,
    "seed": 32,

    # ----- LOSO -----
    "loso_epochs": 600,
    "use_ea": True,

    # ----- Augmentation -----
    "use_augmentation": True,   # False can give higher SD accuracy; helps LOSO more than SD

    # ----- Hardware -----
    "num_workers": 2,
    "use_amp": True,
}


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        prop = torch.cuda.get_device_properties(0)
        mem = getattr(prop, "total_memory", getattr(prop, "total_mem", None))
        if mem is None and hasattr(torch.cuda, "mem_get_info"):
            mem = torch.cuda.mem_get_info(0)[1]
        mem_gb = (mem / 1e9) if mem is not None else 0.0
        print(f"GPU: {name} ({mem_gb:.1f} GB)")
        return torch.device("cuda")
    print("No GPU — using CPU (will be slow)")
    return torch.device("cpu")


# ===================================================================
# Reproducibility
# ===================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(state):
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Ensure torch state is a CPU ByteTensor (loaded checkpoints may be on device or wrong dtype)
    t = state["torch"]
    if isinstance(t, torch.Tensor):
        t = t.cpu().to(torch.uint8)
    else:
        t = torch.from_numpy(np.asarray(t, dtype=np.uint8))
    torch.random.set_rng_state(t)
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_state = state["cuda"]
        if not isinstance(cuda_state, (list, tuple)):
            cuda_state = [cuda_state]
        fixed = []
        for s in cuda_state:
            if isinstance(s, torch.Tensor):
                fixed.append(s.cpu().to(torch.uint8))
            else:
                fixed.append(torch.from_numpy(np.asarray(s, dtype=np.uint8)))
        torch.cuda.set_rng_state_all(fixed)


# ===================================================================
# Data
# ===================================================================

class EEGDataset(Dataset):
    """Dataset for preprocessed EEG .pkl files (same format as baseline)."""

    def __init__(self, pkl_path, split="train"):
        with open(pkl_path, "rb") as f:
            X_train, X_test, y_train_oh, y_test_oh = pickle.load(f)
        if split == "train":
            self.x = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train_oh, dtype=torch.float32)
        else:
            self.x = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test_oh, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CombinedDataset(Dataset):
    """Concatenates data from multiple subjects (for LOSO)."""

    def __init__(self, x_list, y_list):
        self.x = torch.tensor(np.concatenate(x_list), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_all_subjects(data_dir, n_subjects=9):
    """Load all subject .pkl data for LOSO evaluation."""
    trains_x, tests_x, trains_y, tests_y = [], [], [], []
    for s in range(1, n_subjects + 1):
        path = os.path.join(data_dir, f"data_all_{s}.pkl")
        with open(path, "rb") as f:
            xtr, xte, ytr, yte = pickle.load(f)
        trains_x.append(xtr)
        tests_x.append(xte)
        trains_y.append(ytr)
        tests_y.append(yte)
    return trains_x, tests_x, trains_y, tests_y


# ===================================================================
# Learning-rate schedule: cosine with linear warmup
# ===================================================================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs]


import math  # noqa: E402 — needed for cosine schedule above


# ===================================================================
# Checkpoint management
# ===================================================================

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_acc, best_kappa, rng_state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_acc": best_acc,
        "best_kappa": best_kappa,
        "rng_state": rng_state,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ckpt.get("rng_state"):
        load_rng_state(ckpt["rng_state"])
    return ckpt["epoch"], ckpt["best_acc"], ckpt["best_kappa"]


# ===================================================================
# Model builder
# ===================================================================

def build_model(cfg, device):
    model = EfficientEEGEncoder(
        n_classes=cfg["n_classes"],
        in_chans=cfg["n_channels"],
        in_samples=cfg["n_timepoints"],
        n_branches=cfg["n_branches"],
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_heads"],
        num_transformer_layers=cfg["num_transformer_layers"],
        intermediate_size=cfg["intermediate_size"],
        dropout=cfg["dropout"],
        tcn_depth=cfg["tcn_depth"],
        tcn_kernel_size=cfg["tcn_kernel_size"],
        tcn_filters=cfg["tcn_filters"],
        use_gradient_ckpt=cfg["use_gradient_checkpoint"],
        use_augmentation=cfg.get("use_augmentation", True),
    ).to(device)
    return model


# ===================================================================
# Training & evaluation loops
# ===================================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn,
                    device, cfg):
    model.train()
    use_l2 = cfg.get("use_explicit_l2", False)
    preds_all, labels_all, loss_sum, n_batches = [], [], 0.0, 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(inputs)
                cls_loss = loss_fn(logits, labels)
                l2_loss = model.get_l2_loss() if use_l2 else 0.0
                loss = cls_loss + l2_loss
            scaled = scaler.scale(loss * cfg["loss_scale_factor"])
            scaled.backward()
            if cfg["grad_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            cls_loss = loss_fn(logits, labels)
            l2_loss = model.get_l2_loss() if use_l2 else 0.0
            loss = cls_loss + l2_loss
            (loss * cfg["loss_scale_factor"]).backward()
            if cfg["grad_clip_norm"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            optimizer.step()

        preds_all.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels_all.extend(labels.argmax(-1).cpu().numpy().tolist())
        loss_sum += cls_loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    train_acc = accuracy_score(labels_all, preds_all)
    avg_loss = loss_sum / max(n_batches, 1)
    return train_acc, avg_loss


@torch.no_grad()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    preds_all, labels_all, total_time = [], [], 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        t0 = time.time()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(inputs)
        else:
            logits = model(inputs)
        total_time += time.time() - t0
        preds_all.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels_all.extend(labels.argmax(-1).cpu().numpy().tolist())

    acc = accuracy_score(labels_all, preds_all)
    kappa = cohen_kappa_score(labels_all, preds_all)
    return acc, kappa, total_time


# ===================================================================
# Subject-Dependent Training
# ===================================================================

def train_subject(subject_idx, cfg, device):
    """Train and evaluate on a single subject with checkpoint resume.

    SD recipe (matching the reference that achieved 81.25%):
      - Adam optimizer (no weight_decay; L2 handled by model.get_l2_loss())
      - label_smoothing = 0.2
      - Constant LR (no cosine schedule)
      - Augmentation OFF
    """
    sub_id = subject_idx + 1
    ckpt_dir = os.path.join(cfg["checkpoint_dir"], "sd", f"subject_{sub_id}")
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    pkl_path = os.path.join(cfg["data_dir"], f"data_all_{sub_id}.pkl")
    train_ds = EEGDataset(pkl_path, "train")
    val_ds = EEGDataset(pkl_path, "val")

    nw = cfg["num_workers"] if device.type == "cuda" else 0
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=nw, pin_memory=pin,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=nw, pin_memory=pin)

    sd_cfg = cfg.copy()
    sd_cfg["use_augmentation"] = False
    sd_cfg["use_explicit_l2"] = True

    model = build_model(sd_cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = None
    scaler = torch.amp.GradScaler("cuda") if (cfg["use_amp"] and device.type == "cuda") else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)

    start_epoch, best_acc, best_kappa = 0, 0.0, 0.0

    if os.path.exists(latest_path):
        print(f"  Resuming from {latest_path}")
        start_epoch, best_acc, best_kappa = load_checkpoint(
            latest_path, model, optimizer, scheduler, scaler, device)
        start_epoch += 1
        print(f"  Resuming at epoch {start_epoch}, best_acc={best_acc:.4f}")

    if subject_idx == 0:
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"  SD recipe: Adam, label_smooth=0.2, no scheduler, explicit L2, no augmentation")

    for epoch in range(start_epoch, cfg["n_epochs"]):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn,
            device, sd_cfg)
        val_acc, kappa, inf_time = evaluate(model, val_loader, device, scaler)

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            best_kappa = kappa
            save_checkpoint(best_path, model, optimizer, scheduler, scaler,
                            epoch, best_acc, best_kappa, save_rng_state())

        if (epoch + 1) % cfg["checkpoint_every"] == 0 or epoch == cfg["n_epochs"] - 1:
            save_checkpoint(latest_path, model, optimizer, scheduler, scaler,
                            epoch, best_acc, best_kappa, save_rng_state())

        if (epoch + 1) % 50 == 0 or epoch == 0 or improved:
            lr = optimizer.param_groups[0]["lr"]
            star = " *" if improved else ""
            print(f"  E{epoch+1:>4d}/{cfg['n_epochs']} | "
                  f"loss={train_loss:.4f} lr={lr:.2e} | "
                  f"train={train_acc:.4f} val={val_acc:.4f} "
                  f"kappa={kappa:.4f} best={best_acc:.4f}{star}")

    return best_acc, best_kappa


def run_subject_dependent(cfg, device):
    """Train all subjects independently (subject-dependent evaluation)."""
    print("\n" + "=" * 64)
    print("  SUBJECT-DEPENDENT TRAINING — EfficientEEGEncoder")
    print("=" * 64)

    results = []
    for s in range(cfg["n_subjects"]):
        print(f"\n--- Subject {s + 1} ---")
        acc, kappa = train_subject(s, cfg, device)
        results.append({"subject": s + 1, "accuracy": acc, "kappa": kappa})
        print(f"  Subject {s+1}: Acc={acc:.4f}, Kappa={kappa:.4f}")

    # Summary
    mean_acc = np.mean([r["accuracy"] for r in results])
    mean_kappa = np.mean([r["kappa"] for r in results])
    print(f"\n{'=' * 64}")
    print(f"  SUBJECT-DEPENDENT RESULTS")
    print(f"{'=' * 64}")
    print(f"  {'Subject':<10} {'Accuracy':>10} {'Kappa':>10}")
    print(f"  {'-' * 32}")
    for r in results:
        print(f"  S{r['subject']:<9} {r['accuracy']:>10.4f} {r['kappa']:>10.4f}")
    print(f"  {'-' * 32}")
    print(f"  {'Mean':<10} {mean_acc:>10.4f} {mean_kappa:>10.4f}")
    print(f"\n  Paper baseline:  86.46% acc, 0.82 kappa")
    print(f"  Your baseline:   84.72% acc, 0.80 kappa")
    print(f"  Efficient model: {mean_acc*100:.2f}% acc, {mean_kappa:.2f} kappa")

    # Save results
    results_path = os.path.join(cfg["results_dir"], "results_sd.json")
    os.makedirs(cfg["results_dir"], exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"results": results, "mean_accuracy": mean_acc,
                    "mean_kappa": mean_kappa}, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    return results


# ===================================================================
# LOSO (Subject-Independent) Training
# ===================================================================

def _apply_ea_to_loso_data(trains_x, tests_x, n_subjects):
    """Apply Euclidean Alignment for LOSO. Data: list of (N, 1, 22, 1125)."""
    if align_all_subjects is None:
        return trains_x, tests_x
    subjects_3d = []
    for i in range(n_subjects):
        tr = np.asarray(trains_x[i][:, 0, :, :])
        te = np.asarray(tests_x[i][:, 0, :, :])
        subjects_3d.append(np.concatenate([tr, te], axis=0))
    aligned, _ = align_all_subjects(subjects_3d)
    ea_trains_x, ea_tests_x = [], []
    for i in range(n_subjects):
        n_tr, n_te = trains_x[i].shape[0], tests_x[i].shape[0]
        ea_trains_x.append(aligned[i][:n_tr][:, np.newaxis, :, :].astype(np.float32))
        ea_tests_x.append(aligned[i][n_tr:][:, np.newaxis, :, :].astype(np.float32))
    return ea_trains_x, ea_tests_x


def run_loso(cfg, device):
    """Leave-One-Subject-Out cross-validation."""
    print("\n" + "=" * 64)
    print("  LOSO (SUBJECT-INDEPENDENT) — EfficientEEGEncoder")
    print("=" * 64)

    trains_x, tests_x, trains_y, tests_y = load_all_subjects(
        cfg["data_dir"], cfg["n_subjects"])

    if cfg.get("use_ea", False) and align_all_subjects is not None:
        print("  Applying Euclidean Alignment (EA)...")
        trains_x, tests_x = _apply_ea_to_loso_data(
            trains_x, tests_x, cfg["n_subjects"])
        print("  EA done.")

    n_epochs = cfg["loso_epochs"]
    results = []

    for test_sub in range(cfg["n_subjects"]):
        print(f"\n--- Fold {test_sub + 1}: test on S{test_sub + 1}, train on rest ---")

        # LOSO split: train on all data from 8 subjects, test on all data from 1
        tr_x, tr_y, te_x, te_y = [], [], [], []
        for i in range(cfg["n_subjects"]):
            if i == test_sub:
                te_x.extend([trains_x[i], tests_x[i]])
                te_y.extend([trains_y[i], tests_y[i]])
            else:
                tr_x.extend([trains_x[i], tests_x[i]])
                tr_y.extend([trains_y[i], tests_y[i]])

        train_ds = CombinedDataset(tr_x, tr_y)
        test_ds = CombinedDataset(te_x, te_y)
        print(f"  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

        nw = cfg["num_workers"] if device.type == "cuda" else 0
        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=nw, pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"],
                                 shuffle=False, num_workers=nw, pin_memory=pin)

        ckpt_dir = os.path.join(cfg["checkpoint_dir"], "loso_v3", f"fold_{test_sub + 1}")
        latest_path = os.path.join(ckpt_dir, "latest.pt")
        best_path = os.path.join(ckpt_dir, "best.pt")

        seed_everything(cfg["seed"])
        model = build_model(cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
        scheduler = CosineWarmupScheduler(optimizer, cfg["warmup_epochs"], n_epochs)
        scaler = torch.amp.GradScaler("cuda") if (cfg["use_amp"] and device.type == "cuda") else None
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

        start_epoch, best_acc, best_kappa = 0, 0.0, 0.0
        if os.path.exists(latest_path):
            print(f"  Resuming from checkpoint...")
            start_epoch, best_acc, best_kappa = load_checkpoint(
                latest_path, model, optimizer, scheduler, scaler, device)
            start_epoch += 1

        for epoch in range(start_epoch, n_epochs):
            train_acc, train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, loss_fn,
                device, cfg)
            val_acc, kappa, _ = evaluate(model, test_loader, device, scaler)

            improved = val_acc > best_acc
            if improved:
                best_acc = val_acc
                best_kappa = kappa
                save_checkpoint(best_path, model, optimizer, scheduler, scaler,
                                epoch, best_acc, best_kappa, save_rng_state())

            if (epoch + 1) % cfg["checkpoint_every"] == 0 or epoch == n_epochs - 1:
                save_checkpoint(latest_path, model, optimizer, scheduler, scaler,
                                epoch, best_acc, best_kappa, save_rng_state())

            if (epoch + 1) % 25 == 0 or epoch == 0 or improved:
                star = " *" if improved else ""
                print(f"  E{epoch+1:>4d}/{n_epochs} | "
                      f"train={train_acc:.4f} val={val_acc:.4f} "
                      f"kappa={kappa:.4f} best={best_acc:.4f}{star}")

        results.append({"subject": test_sub + 1, "accuracy": best_acc,
                         "kappa": best_kappa})
        print(f"  Fold {test_sub+1}: Acc={best_acc:.4f}, Kappa={best_kappa:.4f}")

    # Summary
    mean_acc = np.mean([r["accuracy"] for r in results])
    mean_kappa = np.mean([r["kappa"] for r in results])
    std_acc = np.std([r["accuracy"] for r in results])
    print(f"\n{'=' * 64}")
    print(f"  LOSO RESULTS (SUBJECT-INDEPENDENT)")
    print(f"{'=' * 64}")
    print(f"  {'Subject':<10} {'Accuracy':>10} {'Kappa':>10}")
    print(f"  {'-' * 32}")
    for r in results:
        print(f"  S{r['subject']:<9} {r['accuracy']:>10.4f} {r['kappa']:>10.4f}")
    print(f"  {'-' * 32}")
    print(f"  {'Mean':<10} {mean_acc:>10.4f} {mean_kappa:>10.4f}")
    print(f"  {'Std':<10} {std_acc:>10.4f}")
    print(f"\n  Paper SI accuracy: 74.48%")
    print(f"  Your SI accuracy:  {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

    results_path = os.path.join(cfg["results_dir"], "results_loso.json")
    os.makedirs(cfg["results_dir"], exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"results": results, "mean_accuracy": mean_acc,
                    "mean_kappa": mean_kappa, "std_accuracy": std_acc}, f, indent=2)
    print(f"  Results saved to {results_path}")
    return results


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientEEGEncoder")
    p.add_argument("--mode", choices=["sd", "loso", "both"], default="sd",
                   help="sd=subject-dependent, loso=leave-one-subject-out, both=run both")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to directory with data_all_*.pkl files")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--no_ea", action="store_true", help="Disable Euclidean Alignment for LOSO")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable data augmentation (can improve SD accuracy; keep ON for LOSO)")
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 1 subject, 50 epochs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()

    # Override from CLI
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.no_ea:
        cfg["use_ea"] = False
    if args.no_augment:
        cfg["use_augmentation"] = False
        print("Augmentation disabled (use_augmentation=False)")
    if args.epochs:
        cfg["n_epochs"] = args.epochs
        cfg["loso_epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["lr"] = args.lr
    if args.seed:
        cfg["seed"] = args.seed
    if args.no_amp:
        cfg["use_amp"] = False
    if args.quick:
        cfg["n_subjects"] = 1
        cfg["n_epochs"] = 50
        cfg["loso_epochs"] = 50
        cfg["checkpoint_every"] = 10
        print("QUICK TEST MODE: 1 subject, 50 epochs")

    # Verify data exists
    if not os.path.isdir(cfg["data_dir"]):
        print(f"ERROR: Data directory not found: {cfg['data_dir']}")
        print("Run colab_preprocess_gdf.py first, or set --data_dir")
        sys.exit(1)

    device = get_device()
    seed_everything(cfg["seed"])

    print(f"\nConfig:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if args.mode in ("sd", "both"):
        run_subject_dependent(cfg, device)
    if args.mode in ("loso", "both"):
        run_loso(cfg, device)


if __name__ == "__main__":
    main()

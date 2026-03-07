"""
EfficientEEGEncoder — Complete Verification Script
===================================================
Run this on Google Colab to verify ALL results in one go.

This script:
  1. Detects GPU and prints specs
  2. Finds your .pkl data files (auto-detects path)
  3. Compares baseline vs efficient model (params, speed, memory)
  4. Runs a quick smoke test (1 subject, 30 epochs)
  5. Prints what to run next for full results

Usage on Colab:
  !python /content/drive/MyDrive/eegencoder/run_verification.py
"""

import os
import sys
import time
import json

# Make sure we can import from the same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def find_data_dir():
    """Auto-detect where the .pkl data files are stored on Google Drive."""
    candidates = [
        "/content/drive/MyDrive/eegencoder/data/",
        "/content/drive/MyDrive/eegencoder/datasets/mat/data/",
        "/content/drive/MyDrive/eegencoder/datasets/data/",
        os.path.join(SCRIPT_DIR, "data/"),
        os.path.join(SCRIPT_DIR, "EEGEncoder-main/data/"),
    ]
    for path in candidates:
        check = os.path.join(path, "data_all_1.pkl")
        if os.path.exists(check):
            n_files = sum(1 for f in os.listdir(path) if f.startswith("data_all_") and f.endswith(".pkl"))
            print(f"  Found {n_files} .pkl files at: {path}")
            return path
    return None


def check_gpu():
    """Print GPU information."""
    import torch
    print("\n" + "=" * 64)
    print("  STEP 1: GPU CHECK")
    print("=" * 64)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        prop = torch.cuda.get_device_properties(0)
        mem_bytes = getattr(prop, 'total_memory', getattr(prop, 'total_mem', None))
        if mem_bytes is None:
            mem_bytes = torch.cuda.mem_get_info(0)[1]
        mem = mem_bytes / 1e9
        cap = torch.cuda.get_device_capability(0)
        print(f"  GPU:        {name}")
        print(f"  Memory:     {mem:.1f} GB")
        print(f"  Capability: sm_{cap[0]}{cap[1]}")
        print(f"  PyTorch:    {torch.__version__}")
        print(f"  CUDA:       {torch.version.cuda}")
        return torch.device("cuda")
    else:
        print("  WARNING: No GPU detected! Training will be very slow.")
        print("  Go to: Runtime -> Change runtime type -> GPU")
        return torch.device("cpu")


def compare_models(device):
    """Compare parameter count, inference speed, and memory."""
    import torch
    print("\n" + "=" * 64)
    print("  STEP 2: MODEL COMPARISON (Baseline vs Efficient)")
    print("=" * 64)

    from efficient_eegencoder import EfficientEEGEncoder, count_parameters, model_summary

    efficient = EfficientEEGEncoder().to(device)
    ep = count_parameters(efficient)
    print("\n  --- EfficientEEGEncoder ---")
    model_summary(efficient)

    # Try to load baseline for comparison
    try:
        from colab_train_baseline import EEGEncoder
        from colab_train_baseline import count_parameters as cp_base
        baseline = EEGEncoder().to(device)
        bp = cp_base(baseline)
        print(f"\n  --- Comparison ---")
        print(f"  Baseline EEGEncoder:    {bp:>10,} params")
        print(f"  EfficientEEGEncoder:    {ep:>10,} params")
        print(f"  Reduction:              {(1 - ep / bp) * 100:>9.1f}%")

        # Speed comparison
        x = torch.randn(16, 1, 22, 1125).to(device)
        baseline.eval()
        efficient.eval()

        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = baseline(x)
                    _ = efficient(x)
                    torch.cuda.synchronize()

                    t0 = time.time()
                    for _ in range(50):
                        _ = baseline(x)
                    torch.cuda.synchronize()
                    bt = (time.time() - t0) / 50

                    t0 = time.time()
                    for _ in range(50):
                        _ = efficient(x)
                    torch.cuda.synchronize()
                    et = (time.time() - t0) / 50
            else:
                _ = baseline(x)
                _ = efficient(x)
                t0 = time.time()
                for _ in range(20):
                    _ = baseline(x)
                bt = (time.time() - t0) / 20
                t0 = time.time()
                for _ in range(20):
                    _ = efficient(x)
                et = (time.time() - t0) / 20

        print(f"\n  Inference (batch=16, {'AMP' if device.type == 'cuda' else 'FP32'}):")
        print(f"  Baseline:               {bt * 1000:>9.2f} ms")
        print(f"  Efficient:              {et * 1000:>9.2f} ms")
        print(f"  Speedup:                {bt / et:>9.2f}x")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda"):
                _ = baseline(x)
            baseline_mem = torch.cuda.max_memory_allocated() / 1e6

            del baseline
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda"):
                _ = efficient(x)
            efficient_mem = torch.cuda.max_memory_allocated() / 1e6

            print(f"\n  Peak GPU Memory:")
            print(f"  Baseline:               {baseline_mem:>9.1f} MB")
            print(f"  Efficient:              {efficient_mem:>9.1f} MB")
            print(f"  Reduction:              {(1 - efficient_mem / baseline_mem) * 100:>9.1f}%")

        has_baseline = True
    except Exception as e:
        print(f"\n  Could not load baseline for comparison: {e}")
        print(f"  (This is OK — baseline comparison is optional)")
        print(f"\n  EfficientEEGEncoder: {ep:,} params")
        has_baseline = False

    return has_baseline


def quick_smoke_test(data_dir, device):
    """Run 1 subject for 30 epochs to verify everything works."""
    import torch
    print("\n" + "=" * 64)
    print("  STEP 3: QUICK SMOKE TEST (1 subject, 30 epochs)")
    print("=" * 64)

    from train_efficient import (
        DEFAULT_CONFIG, build_model, EEGDataset, train_one_epoch,
        evaluate, CosineWarmupScheduler, seed_everything,
    )
    import torch.nn as nn
    from torch.utils.data import DataLoader

    cfg = DEFAULT_CONFIG.copy()
    cfg["data_dir"] = data_dir
    cfg["n_epochs"] = 30
    cfg["warmup_epochs"] = 3
    cfg["checkpoint_every"] = 999  # Don't save checkpoints for smoke test

    seed_everything(cfg["seed"])
    model = build_model(cfg, device)
    pkl_path = os.path.join(data_dir, "data_all_1.pkl")
    train_ds = EEGDataset(pkl_path, "train")
    val_ds = EEGDataset(pkl_path, "val")

    nw = cfg["num_workers"] if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=nw,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=nw,
                            pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = CosineWarmupScheduler(optimizer, cfg["warmup_epochs"], cfg["n_epochs"])
    scaler = torch.amp.GradScaler("cuda") if (cfg["use_amp"] and device.type == "cuda") else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"  Training for {cfg['n_epochs']} epochs on Subject 1...\n")

    t_start = time.time()
    best_acc = 0
    for epoch in range(cfg["n_epochs"]):
        train_acc, train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, device, cfg)
        val_acc, kappa, _ = evaluate(model, val_loader, device, scaler)
        if val_acc > best_acc:
            best_acc = val_acc
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3d}/30 | loss={train_loss:.4f} | "
                  f"train={train_acc:.4f} val={val_acc:.4f} kappa={kappa:.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Smoke test completed in {elapsed:.1f}s")
    print(f"  Best val accuracy (30 epochs): {best_acc:.4f}")
    print(f"  Estimated time for 500 epochs: {elapsed / 30 * 500 / 60:.1f} min per subject")
    print(f"  Estimated time for 9 subjects: {elapsed / 30 * 500 * 9 / 3600:.1f} hours")

    if best_acc > 0.30:
        print("\n  SMOKE TEST PASSED — model is learning")
        return True
    else:
        print("\n  WARNING: accuracy is very low — something may be wrong")
        return False


def print_next_steps(data_dir):
    """Print the exact commands for full training."""
    print("\n" + "=" * 64)
    print("  STEP 4: NEXT STEPS")
    print("=" * 64)
    print(f"""
  Everything is working! Now run the full experiments:

  --- SUBJECT-DEPENDENT (full 9 subjects, 500 epochs) ---
  !python {SCRIPT_DIR}/train_efficient.py \\
      --mode sd \\
      --data_dir "{data_dir}"

  --- LOSO / SUBJECT-INDEPENDENT (9 folds, 200 epochs) ---
  !python {SCRIPT_DIR}/train_efficient.py \\
      --mode loso \\
      --data_dir "{data_dir}"

  --- OR RUN BOTH IN SEQUENCE ---
  !python {SCRIPT_DIR}/train_efficient.py \\
      --mode both \\
      --data_dir "{data_dir}"

  If Colab disconnects, just re-run the SAME command.
  It auto-resumes from the last checkpoint (saved to Google Drive).

  After training completes, share these files with me:
    1. The terminal output (copy-paste the full results table)
    2. {SCRIPT_DIR}/results/results_sd.json
    3. {SCRIPT_DIR}/results/results_loso.json
""")


def main():
    print("=" * 64)
    print("  EfficientEEGEncoder — Complete Verification")
    print("=" * 64)

    # Step 1: GPU
    device = check_gpu()

    # Step 2: Find data
    print("\n" + "=" * 64)
    print("  DATA CHECK")
    print("=" * 64)
    data_dir = find_data_dir()
    if data_dir is None:
        print("  ERROR: Could not find data_all_*.pkl files!")
        print("  Please run colab_preprocess_gdf.py first.")
        print("  Or set the path manually in DEFAULT_CONFIG['data_dir']")
        sys.exit(1)

    # Step 3: Model comparison
    compare_models(device)

    # Step 4: Smoke test
    passed = quick_smoke_test(data_dir, device)

    # Step 5: Next steps
    if passed:
        print_next_steps(data_dir)
    else:
        print("\n  Fix the issue above before running full training.")


if __name__ == "__main__":
    main()

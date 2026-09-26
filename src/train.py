
import os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses import BCEDiceLoss
from src.reproducibility import set_seed
from src.datasets import SliceDataset
from src.metrics import validate_3d


def run_training(run_id, budget, strategy, model_class, selected_ids, validation_ids,
                  volume_cache, device, batch_size=8, max_epochs=100, patience=15, min_delta=0.001,
                  checkpoint_dir="checkpoints", log_dir="logs"):

    # Leakage check (Section 11, step 3)
    overlap = set(selected_ids) & set(validation_ids)
    assert len(overlap) == 0, f"LEAKAGE DETECTED in {run_id}: {overlap}"

    train_ds = SliceDataset(selected_ids)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    set_seed(42)
    model = model_class().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    param_audit = {"run_id": run_id, "trainable_params": trainable_params,
                   "total_params": total_params, "frozen_params": total_params - trainable_params}

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_val_dsc = -1.0
    epochs_without_improvement = 0
    loss_curve, val_dsc_curve, epoch_times = [], [], []
    oom_fallback_used = False

    torch.cuda.reset_peak_memory_stats()
    run_start = time.time()
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        try:
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
        except torch.cuda.OutOfMemoryError:
            if not oom_fallback_used:
                batch_size = max(1, batch_size // 2)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
                oom_fallback_used = True
                torch.cuda.empty_cache()
                print(f"[{run_id}] OOM -> batch_size reduced to {batch_size}")
                continue
            else:
                raise

        mean_train_loss = float(np.mean(epoch_losses))
        loss_curve.append(mean_train_loss)

        mean_val_dsc, _ = validate_3d(model, validation_ids, volume_cache, device)
        val_dsc_curve.append(mean_val_dsc)
        epoch_times.append(time.time() - epoch_start)

        improved = mean_val_dsc > (best_val_dsc + min_delta)
        print(f"[{run_id}] Epoch {epoch:3d} | loss={mean_train_loss:.4f} | val_DSC={mean_val_dsc:.4f} "
              f"| best={best_val_dsc:.4f}" + (" *BEST*" if improved else ""))

        if improved:
            best_val_dsc = mean_val_dsc
            epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_dsc": mean_val_dsc, "run_id": run_id},
                       f"{checkpoint_dir}/{run_id}_best.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[{run_id}] Early stopping at epoch {epoch}")
            break

    total_runtime = time.time() - run_start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    run_log = {
        "run_id": run_id, "budget": budget, "strategy": strategy, "model": model_class.__name__,
        "selected_ids": selected_ids, "batch_size_used": batch_size, "oom_fallback_used": oom_fallback_used,
        "epochs_run": epoch, "best_val_dsc": best_val_dsc, "total_runtime_sec": round(total_runtime, 1),
        "peak_gpu_memory_gb": round(peak_mem_gb, 2),
        "mean_epoch_time_sec": round(float(np.mean(epoch_times)), 1) if epoch_times else None,
        "loss_curve": loss_curve, "val_dsc_curve": val_dsc_curve, "param_audit": param_audit
    }
    with open(f"{log_dir}/{run_id}.json", "w") as jf:
        json.dump(run_log, jf, indent=2)

    return model, run_log

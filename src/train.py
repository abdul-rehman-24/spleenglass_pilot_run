import os
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses import BCEDiceLoss
from src.reproducibility import set_seed
from src.datasets import SliceDataset
from src.metrics import validate_3d


def run_training(
    run_id,
    budget,
    strategy,
    model_class,
    selected_ids,
    validation_ids,
    volume_cache,
    device,
    batch_size=8,
    max_epochs=100,
    patience=15,
    min_delta=0.001,
    checkpoint_dir="checkpoints_v2",
    log_dir="logs_v2",
    *,
    training_seed=42,
    selection_seed=None,
):
    selected_ids = list(selected_ids)
    validation_ids = list(validation_ids)
    device = torch.device(device)
    use_cuda = device.type == "cuda"

    # Input validation
    if max_epochs < 1:
        raise ValueError(f"[{run_id}] max_epochs must be >= 1")
    if batch_size < 1:
        raise ValueError(f"[{run_id}] batch_size must be >= 1")
    if patience < 1:
        raise ValueError(f"[{run_id}] patience must be >= 1")
    if min_delta < 0:
        raise ValueError(f"[{run_id}] min_delta must be >= 0")
    if not selected_ids:
        raise ValueError(f"[{run_id}] selected_ids is empty")
    if not validation_ids:
        raise ValueError(f"[{run_id}] validation_ids is empty")
    if len(selected_ids) != budget:
        raise ValueError(
            f"[{run_id}] len(selected_ids)={len(selected_ids)} "
            f"does not match budget={budget}"
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"[{run_id}] Duplicate training volume IDs")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError(f"[{run_id}] Duplicate validation volume IDs")

    overlap = set(selected_ids) & set(validation_ids)
    if overlap:
        raise ValueError(
            f"[{run_id}] Train/validation leakage: {sorted(overlap)}"
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ckpt_path = os.path.join(
        checkpoint_dir, f"{run_id}_best.pt"
    )
    log_path = os.path.join(log_dir, f"{run_id}.json")

    if os.path.exists(ckpt_path):
        raise FileExistsError(
            f"Checkpoint already exists: {ckpt_path}"
        )
    if os.path.exists(log_path):
        raise FileExistsError(
            f"Log already exists: {log_path}"
        )

    training_seed = int(training_seed)
    selection_seed = (
        None if selection_seed is None else int(selection_seed)
    )

    run_log = {
        "protocol_version": "epoch_baseline_v2",
        "status": "running",
        "run_id": run_id,
        "budget": int(budget),
        "strategy": strategy,
        "model": getattr(
            model_class, "__name__", type(model_class).__name__
        ),
        "selected_ids": selected_ids,
        "validation_ids": validation_ids,
        "training_seed": training_seed,
        "selection_seed": selection_seed,
        "batch_size_used": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "oom_fallback_used": False,
        "epochs_run": 0,
        "best_epoch": None,
        "best_val_dsc": None,
        "best_checkpoint_optimizer_steps": None,
        "optimizer_steps": 0,
        "loss_curve": [],
        "val_dsc_curve": [],
        "epoch_times_sec": [],
    }

    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            f"Cannot JSON-serialize {type(value).__name__}"
        )

    def save_log():
        with open(log_path, "w") as handle:
            json.dump(
                run_log,
                handle,
                indent=2,
                default=json_default,
            )

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(device)

    save_log()

    optimizer_steps = 0
    memory_stats_started = False
    run_start = time.perf_counter()

    try:
        # Seed before dataset/model creation.
        set_seed(training_seed)

        train_ds = SliceDataset(selected_ids)
        if len(train_ds) == 0:
            raise ValueError(
                f"[{run_id}] Training dataset is empty"
            )

        train_generator = torch.Generator()
        train_generator.manual_seed(training_seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=train_generator,
        )

        model = model_class().to(device)
        criterion = BCEDiceLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        total_params = sum(
            p.numel() for p in model.parameters()
        )

        run_log.update({
            "total_training_slices": len(train_ds),
            "steps_per_epoch": len(train_loader),
            "param_audit": {
                "run_id": run_id,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "frozen_params": total_params - trainable_params,
            },
        })
        save_log()

        best_val_dsc = -float("inf")
        es_reference_dsc = -float("inf")
        epochs_without_improvement = 0

        if use_cuda:
            synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            memory_stats_started = True

        for epoch in range(1, max_epochs + 1):
            synchronize()
            epoch_start = time.perf_counter()

            model.train()
            weighted_loss_sum = 0.0
            sample_count = 0

            # Fixed batch size; OOM propagates to failure handling.
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(imgs)
                loss = criterion(logits, lbls)

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"[{run_id}] Non-finite training loss"
                    )

                loss.backward()
                optimizer.step()
                optimizer_steps += 1

                n = imgs.size(0)
                weighted_loss_sum += loss.item() * n
                sample_count += n

            mean_train_loss = (
                weighted_loss_sum / sample_count
            )

            previous_mode = model.training
            model.eval()
            try:
                mean_val_dsc, per_volume_dsc = validate_3d(
                    model,
                    validation_ids,
                    volume_cache,
                    device,
                )
            finally:
                model.train(previous_mode)

            mean_val_dsc = float(mean_val_dsc)
            if not np.isfinite(mean_val_dsc):
                raise FloatingPointError(
                    f"[{run_id}] Non-finite validation DSC"
                )

            # Save every actual new maximum.
            is_best = mean_val_dsc > best_val_dsc

            if is_best:
                checkpoint = {
                    "protocol_version": "epoch_baseline_v2",
                    "run_id": run_id,
                    "epoch": epoch,
                    "best_epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dsc": mean_val_dsc,
                    "per_volume_val_dsc": [
                        float(x) for x in per_volume_dsc
                    ],
                    "training_seed": training_seed,
                    "selection_seed": selection_seed,
                    "selected_ids": selected_ids,
                    "validation_ids": validation_ids,
                }
                torch.save(checkpoint, ckpt_path)

                best_val_dsc = mean_val_dsc
                run_log.update({
                    "best_epoch": epoch,
                    "best_val_dsc": best_val_dsc,
                    "best_checkpoint_optimizer_steps": optimizer_steps,
                    "best_per_volume_val_dsc": [
                        float(x) for x in per_volume_dsc
                    ],
                })

            # min_delta affects patience, not best-model saving.
            if mean_val_dsc > es_reference_dsc + min_delta:
                es_reference_dsc = mean_val_dsc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            synchronize()
            epoch_seconds = (
                time.perf_counter() - epoch_start
            )

            run_log["loss_curve"].append(
                float(mean_train_loss)
            )
            run_log["val_dsc_curve"].append(mean_val_dsc)
            run_log["epoch_times_sec"].append(epoch_seconds)
            run_log["epochs_run"] = epoch
            run_log["optimizer_steps"] = optimizer_steps
            save_log()

            print(
                f"[{run_id}] Epoch {epoch:3d} "
                f"| loss={mean_train_loss:.4f} "
                f"| val_DSC={mean_val_dsc:.4f} "
                f"| best={best_val_dsc:.4f} "
                f"| steps={optimizer_steps} "
                f"| time={epoch_seconds:.1f}s"
                + (" *BEST*" if is_best else "")
            )

            if epochs_without_improvement >= patience:
                print(
                    f"[{run_id}] Early stopping at epoch {epoch}"
                )
                break

        # Load only the trusted checkpoint generated by this run.
        best_ckpt = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_ckpt["model_state_dict"])

        run_log["returned_model_checkpoint"] = "best_validation"
        run_log["status"] = "completed"

    except Exception as error:
        run_log.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise

    finally:
        run_log["optimizer_steps"] = optimizer_steps
        run_log["total_runtime_sec"] = round(
            time.perf_counter() - run_start, 1
        )

        times = run_log["epoch_times_sec"]
        run_log["mean_epoch_time_sec"] = (
            round(float(np.mean(times)), 1)
            if times else None
        )

        try:
            run_log["peak_gpu_memory_gb"] = (
                round(
                    torch.cuda.max_memory_allocated(device) / 1e9,
                    2,
                )
                if memory_stats_started else None
            )
        except Exception:
            run_log["peak_gpu_memory_gb"] = None

        # Preserve the original training exception if logging fails.
        try:
            save_log()
        except Exception as log_error:
            if run_log["status"] != "failed":
                raise
            print(
                f"[{run_id}] Could not save failure log: "
                f"{log_error}"
            )

    return model, run_log

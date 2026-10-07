from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    HipMRIDataset,
    discover_volume_pairs,
    split_by_patient,
    write_split_manifest,
)
from losses import (
    DeepSupervisionLoss,
    DiceCrossEntropyLoss,
    dice_from_statistics,
    dice_statistics,
)
from modules import ResidualUNet3D, UNet3D, count_trainable_parameters


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("baseline", "proposed"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 128, 128))
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--foreground-probability", type=float, default=0.75)
    parser.add_argument("--class-weights", type=float, nargs=6)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    return parser


def create_model(
    model_name: str,
    base_channels: int,
) -> nn.Module:
    if model_name == "baseline":
        return UNet3D(base_channels=base_channels)
    if model_name == "proposed":
        return ResidualUNet3D(base_channels=base_channels)
    raise ValueError(f"Unknown model: {model_name}")


def create_data_loaders(
    data_root: Path,
    patch_size: tuple[int, int, int],
    batch_size: int,
    workers: int,
    seed: int,
    foreground_probability: float,
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], dict[str, Any]]:
    pairs = discover_volume_pairs(data_root)
    splits = split_by_patient(pairs, seed=seed)
    train_dataset = HipMRIDataset(
        splits["train"],
        patch_size=patch_size,
        training=True,
        foreground_probability=foreground_probability,
        augment=True,
    )
    validation_dataset = HipMRIDataset(
        splits["validation"],
        patch_size=patch_size,
        training=False,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_arguments = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_arguments,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_arguments,
    )
    return train_loader, validation_loader, splits


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader[dict[str, object]],
    criterion: nn.Module,
    device: torch.device,
    training: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_clip: float,
    max_batches: int | None,
) -> dict[str, Any]:
    if training and optimizer is None:
        raise ValueError("Training requires an optimizer")
    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches must be positive, received {max_batches}")

    model.train(training)
    loss_total = 0.0
    sample_count = 0
    intersection_total: torch.Tensor | None = None
    denominator_total: torch.Tensor | None = None

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image = _require_tensor(batch, "image").to(device, non_blocking=True)
        label = _require_tensor(batch, "label").to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(image)
                loss = criterion(outputs, label)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at batch {batch_index}: {float(loss.detach())}"
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

        batch_size = image.shape[0]
        loss_total += float(loss.detach()) * batch_size
        sample_count += batch_size
        batch_intersection, batch_denominator = dice_statistics(
            _main_logits(outputs).detach(),
            label,
        )
        batch_intersection = batch_intersection.cpu()
        batch_denominator = batch_denominator.cpu()
        if intersection_total is None:
            intersection_total = batch_intersection
            denominator_total = batch_denominator
        else:
            intersection_total += batch_intersection
            denominator_total += batch_denominator

    if sample_count == 0 or intersection_total is None or denominator_total is None:
        raise RuntimeError("The epoch processed no samples")
    dice = dice_from_statistics(intersection_total, denominator_total)
    foreground_dice = dice[1:]
    finite_foreground = foreground_dice[torch.isfinite(foreground_dice)]
    if finite_foreground.numel() == 0:
        raise RuntimeError("No foreground class had a defined Dice score")
    return {
        "loss": loss_total / sample_count,
        "per_class_dice": [
            None if not torch.isfinite(value) else float(value)
            for value in dice
        ],
        "mean_foreground_dice": float(finite_foreground.mean()),
        "samples": sample_count,
    }


def train(arguments: argparse.Namespace) -> None:
    _validate_arguments(arguments)
    output_directory = arguments.output_dir.expanduser().resolve()
    _prepare_output_directory(output_directory)
    configuration = _serialise_arguments(arguments)
    (output_directory / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )

    _set_seed(arguments.seed, arguments.deterministic)
    device = _resolve_device(arguments.device)
    use_amp = arguments.amp and device.type == "cuda"
    patch_size = tuple(arguments.patch_size)
    train_loader, validation_loader, splits = create_data_loaders(
        data_root=arguments.data_root,
        patch_size=patch_size,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        seed=arguments.seed,
        foreground_probability=arguments.foreground_probability,
    )
    write_split_manifest(splits, output_directory / "split_manifest.json")

    model = create_model(arguments.model, arguments.base_channels).to(device)
    base_loss = DiceCrossEntropyLoss(class_weights=arguments.class_weights)
    criterion = DeepSupervisionLoss(base_loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=arguments.epochs,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    metrics_path = output_directory / "metrics.jsonl"
    best_score = -1.0

    for epoch in range(1, arguments.epochs + 1):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            training=True,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip=arguments.gradient_clip,
            max_batches=arguments.max_train_batches,
        )
        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            training=False,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip=arguments.gradient_clip,
            max_batches=arguments.max_validation_batches,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed_seconds = time.perf_counter() - epoch_start
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        )
        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_mb": peak_memory_mb,
            "train": training_metrics,
            "validation": validation_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(epoch_metrics) + "\n")
        print(json.dumps(epoch_metrics, indent=2), flush=True)

        state = {
            "epoch": epoch,
            "model_name": arguments.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "configuration": configuration,
            "validation_mean_foreground_dice": validation_metrics[
                "mean_foreground_dice"
            ],
        }
        _save_checkpoint(state, output_directory / "last.pt")
        current_score = validation_metrics["mean_foreground_dice"]
        if current_score > best_score:
            best_score = current_score
            _save_checkpoint(state, output_directory / "best.pt")

    summary = {
        "best_validation_mean_foreground_dice": best_score,
        "device": str(device),
        "mixed_precision": use_amp,
        "trainable_parameters": count_trainable_parameters(model),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _main_logits(outputs: object) -> torch.Tensor:
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise TypeError("Model must return (logits, auxiliary_logits)")
    logits = outputs[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Main model output must be a tensor")
    return logits


def _require_tensor(batch: dict[str, object], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key} must be a tensor")
    return value


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive_values = {
        "epochs": arguments.epochs,
        "base_channels": arguments.base_channels,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "gradient_clip": arguments.gradient_clip,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, received {value}")
    if arguments.weight_decay < 0:
        raise ValueError(
            f"weight_decay must be non-negative, received {arguments.weight_decay}"
        )
    if arguments.workers < 0:
        raise ValueError(f"workers must be non-negative, received {arguments.workers}")


def _prepare_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    reserved_names = (
        "best.pt",
        "last.pt",
        "config.json",
        "metrics.jsonl",
        "split_manifest.json",
        "summary.json",
    )
    existing = [name for name in reserved_names if (path / name).exists()]
    if existing:
        raise FileExistsError(
            f"Output directory contains existing run files: {existing}"
        )


def _serialise_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(arguments).items()
    }


def _save_checkpoint(state: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    arguments = build_argument_parser().parse_args()
    train(arguments)


if __name__ == "__main__":
    main()

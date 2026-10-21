"""Train and validate the matched slice-wise 2D U-Net baseline.

Author: Jiacheng Yu (s4916091).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, discover_volume_pairs, split_by_patient, write_split_manifest
from losses import DeepSupervisionLoss, DiceCrossEntropyLoss, dice_from_statistics, dice_statistics
from modules import UNet2D, count_trainable_parameters
from train import (
    _prepare_output_directory,
    _resolve_device,
    _save_checkpoint,
    _serialise_arguments,
    _set_seed,
)
from visualization import plot_training_history


NUM_CLASSES = 6
MODEL_NAME = "baseline_2d"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--slices-per-volume", type=int, default=16)
    parser.add_argument("--slice-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--foreground-probability", type=float, default=0.75)
    parser.add_argument("--class-weights", type=float, nargs=NUM_CLASSES)
    parser.add_argument("--selection-class", type=int, default=5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-train-volumes", type=int)
    parser.add_argument("--max-validation-volumes", type=int)
    return parser


def create_data_loaders(
    data_root: Path,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], dict[str, Any]]:
    pairs = discover_volume_pairs(data_root)
    splits = split_by_patient(pairs, seed=seed)
    train_dataset = HipMRIDataset(splits["train"], training=False)
    validation_dataset = HipMRIDataset(splits["validation"], training=False)
    loader_arguments = {
        "batch_size": 1,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
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
    slices_per_volume: int,
    slice_batch_size: int,
    foreground_probability: float,
    max_volumes: int | None,
) -> dict[str, Any]:
    if training and optimizer is None:
        raise ValueError("Training requires an optimizer")
    model.train(training)
    loss_total = 0.0
    sample_count = 0
    volume_count = 0
    intersection_total: torch.Tensor | None = None
    denominator_total: torch.Tensor | None = None

    for volume_index, batch in enumerate(data_loader):
        if max_volumes is not None and volume_index >= max_volumes:
            break
        image_volume = _require_tensor(batch, "image")[0]
        label_volume = _require_tensor(batch, "label")[0]
        if training:
            indices = _select_training_slices(
                label_volume,
                slices_per_volume,
                foreground_probability,
            )
        else:
            indices = torch.arange(label_volume.shape[0])

        image_slices = image_volume[:, indices].movedim(1, 0)
        label_slices = label_volume[indices]
        if training:
            image_slices, label_slices = _random_in_plane_flip(
                image_slices,
                label_slices,
            )

        for start in range(0, len(indices), slice_batch_size):
            image = image_slices[start : start + slice_batch_size].to(
                device,
                non_blocking=True,
            )
            label = label_slices[start : start + slice_batch_size].to(
                device,
                non_blocking=True,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(image)
                    loss = criterion(outputs, label)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at volume {volume_index}: {float(loss.detach())}"
                    )
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()

            chunk_size = image.shape[0]
            loss_total += float(loss.detach()) * chunk_size
            sample_count += chunk_size
            intersection, denominator = dice_statistics(outputs[0].detach(), label)
            intersection = intersection.cpu()
            denominator = denominator.cpu()
            if intersection_total is None:
                intersection_total = intersection
                denominator_total = denominator
            else:
                intersection_total += intersection
                denominator_total += denominator
        volume_count += 1

    if sample_count == 0 or intersection_total is None or denominator_total is None:
        raise RuntimeError("The epoch processed no slices")
    dice = dice_from_statistics(intersection_total, denominator_total)
    finite_foreground = dice[1:][torch.isfinite(dice[1:])]
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
        "volumes": volume_count,
    }


def train(arguments: argparse.Namespace) -> None:
    _validate_arguments(arguments)
    output_directory = arguments.output_dir.expanduser().resolve()
    _prepare_output_directory(output_directory)
    configuration = _serialise_arguments(arguments)
    configuration["model"] = MODEL_NAME
    configuration["dimensionality"] = "2d"
    (output_directory / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )

    device = _resolve_device(arguments.device)
    _set_seed(arguments.seed, arguments.deterministic, device)
    use_amp = arguments.amp and device.type == "cuda"
    train_loader, validation_loader, splits = create_data_loaders(
        data_root=arguments.data_root,
        workers=arguments.workers,
        seed=arguments.seed,
        pin_memory=device.type == "cuda",
    )
    write_split_manifest(splits, output_directory / "split_manifest.json")

    model = UNet2D(base_channels=arguments.base_channels).to(device)
    _set_seed(arguments.seed, arguments.deterministic, device)
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
            model,
            train_loader,
            criterion,
            device,
            True,
            optimizer,
            scaler,
            use_amp,
            arguments.gradient_clip,
            arguments.slices_per_volume,
            arguments.slice_batch_size,
            arguments.foreground_probability,
            arguments.max_train_volumes,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            False,
            None,
            scaler,
            use_amp,
            arguments.gradient_clip,
            arguments.slices_per_volume,
            arguments.slice_batch_size,
            arguments.foreground_probability,
            arguments.max_validation_volumes,
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
        plot_training_history(
            metrics_path,
            output_directory / "training_curves.png",
            arguments.selection_class,
        )
        print(json.dumps(epoch_metrics, indent=2), flush=True)

        selection_score = validation_metrics["per_class_dice"][arguments.selection_class]
        if selection_score is None:
            raise RuntimeError(
                "Validation Dice is undefined for checkpoint selection class "
                f"{arguments.selection_class}"
            )
        state = {
            "epoch": epoch,
            "model_name": MODEL_NAME,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "configuration": configuration,
            "validation_mean_foreground_dice": validation_metrics[
                "mean_foreground_dice"
            ],
            "validation_selection_class": arguments.selection_class,
            "validation_selection_dice": selection_score,
        }
        _save_checkpoint(state, output_directory / "last.pt")
        if selection_score > best_score:
            best_score = selection_score
            _save_checkpoint(state, output_directory / "best.pt")

    summary = {
        "selection_class": arguments.selection_class,
        "best_validation_selection_dice": best_score,
        "device": str(device),
        "mixed_precision": use_amp,
        "trainable_parameters": count_trainable_parameters(model),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def _select_training_slices(
    label_volume: torch.Tensor,
    count: int,
    foreground_probability: float,
) -> torch.Tensor:
    selected = []
    depth = label_volume.shape[0]
    for _ in range(count):
        if torch.rand(()) < foreground_probability:
            class_id = int(torch.randint(3, NUM_CLASSES, (1,)))
            candidates = torch.nonzero(
                torch.any(label_volume == class_id, dim=(1, 2)),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                raise ValueError(f"Foreground class {class_id} is absent from a volume")
            selected.append(candidates[int(torch.randint(len(candidates), (1,)))])
        else:
            selected.append(torch.randint(depth, (1,))[0])
    return torch.stack(selected)


def _random_in_plane_flip(
    image: torch.Tensor,
    label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for dimension in (-2, -1):
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(dimension,))
            label = torch.flip(label, dims=(dimension,))
    return image, label


def _require_tensor(batch: dict[str, object], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch field {key} must be a tensor")
    return value


def _validate_arguments(arguments: argparse.Namespace) -> None:
    positive_values = {
        "epochs": arguments.epochs,
        "base_channels": arguments.base_channels,
        "slices_per_volume": arguments.slices_per_volume,
        "slice_batch_size": arguments.slice_batch_size,
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
    if not 0.0 <= arguments.foreground_probability <= 1.0:
        raise ValueError(
            "foreground_probability must be in [0, 1], "
            f"received {arguments.foreground_probability}"
        )
    if not 0 <= arguments.selection_class < NUM_CLASSES:
        raise ValueError(
            f"selection_class must be in [0, 5], received {arguments.selection_class}"
        )
    for name in ("max_train_volumes", "max_validation_volumes"):
        value = getattr(arguments, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive, received {value}")


def main() -> None:
    arguments = build_argument_parser().parse_args()
    train(arguments)


if __name__ == "__main__":
    main()

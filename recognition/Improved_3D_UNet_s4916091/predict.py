from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from dataset import HipMRIDataset, discover_volume_pairs, split_by_patient
from metrics import aggregate_case_metrics, evaluate_segmentation
from modules import ResidualUNet3D, UNet3D


NUM_CLASSES = 6


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 128, 128))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--window-batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hd95", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int)
    return parser


def sliding_window_logits(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    overlap: float,
    window_batch_size: int,
    use_amp: bool,
) -> torch.Tensor:
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError(
            "Expected one image with shape [1, channels, depth, height, width], "
            f"received {tuple(image.shape)}"
        )
    if len(patch_size) != 3 or any(size < 32 for size in patch_size):
        raise ValueError(f"Invalid patch size: {patch_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), received {overlap}")
    if window_batch_size <= 0:
        raise ValueError(
            f"window_batch_size must be positive, received {window_batch_size}"
        )

    original_size = tuple(image.shape[2:])
    padded_image, padding_before = _pad_image(image, patch_size)
    padded_size = tuple(padded_image.shape[2:])
    starts = [
        _scan_starts(image_size, window_size, overlap)
        for image_size, window_size in zip(padded_size, patch_size, strict=True)
    ]
    coordinates = list(itertools.product(*starts))
    importance = _gaussian_importance_map(
        patch_size,
        device=image.device,
    )
    accumulated_logits: torch.Tensor | None = None
    accumulated_weights = torch.zeros(
        (1, 1, *padded_size),
        dtype=torch.float32,
        device=image.device,
    )

    for offset in range(0, len(coordinates), window_batch_size):
        coordinate_batch = coordinates[offset : offset + window_batch_size]
        patches = torch.cat(
            [
                padded_image[
                    :,
                    :,
                    depth : depth + patch_size[0],
                    height : height + patch_size[1],
                    width : width + patch_size[2],
                ]
                for depth, height, width in coordinate_batch
            ],
            dim=0,
        )
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=image.device.type,
                enabled=use_amp,
            ):
                patch_logits = model(patches)[0]
        patch_logits = patch_logits.float()
        if accumulated_logits is None:
            accumulated_logits = torch.zeros(
                (1, patch_logits.shape[1], *padded_size),
                dtype=torch.float32,
                device=image.device,
            )
        for patch_index, (depth, height, width) in enumerate(coordinate_batch):
            slices = (
                slice(depth, depth + patch_size[0]),
                slice(height, height + patch_size[1]),
                slice(width, width + patch_size[2]),
            )
            accumulated_logits[(slice(None), slice(None), *slices)] += (
                patch_logits[patch_index : patch_index + 1] * importance
            )
            accumulated_weights[(slice(None), slice(None), *slices)] += importance

    if accumulated_logits is None:
        raise RuntimeError("Sliding-window inference generated no windows")
    if torch.any(accumulated_weights == 0):
        raise RuntimeError("Sliding-window inference left uncovered voxels")
    accumulated_logits /= accumulated_weights
    depth, height, width = padding_before
    return accumulated_logits[
        :,
        :,
        depth : depth + original_size[0],
        height : height + original_size[1],
        width : width + original_size[2],
    ]


def evaluate_checkpoint(arguments: argparse.Namespace) -> None:
    _validate_arguments(arguments)
    device = _resolve_device(arguments.device)
    use_amp = arguments.amp and device.type == "cuda"
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model_name, base_channels, seed = _checkpoint_configuration(checkpoint)
    model = _create_model(model_name, base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pairs = discover_volume_pairs(arguments.data_root)
    splits = split_by_patient(pairs, seed=seed)
    selected_pairs = splits[arguments.split]
    if arguments.max_cases is not None:
        selected_pairs = selected_pairs[: arguments.max_cases]
    dataset = HipMRIDataset(selected_pairs, training=False)

    output_directory = arguments.output_dir.expanduser().resolve()
    metrics_path = output_directory / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Evaluation metrics already exist: {metrics_path}")
    prediction_directory = output_directory / "predictions"
    output_directory.mkdir(parents=True, exist_ok=True)
    if arguments.save_predictions:
        prediction_directory.mkdir(parents=True, exist_ok=True)

    case_results = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = _require_tensor(sample, "image").unsqueeze(0).to(device)
        target = _require_tensor(sample, "label").cpu().numpy()
        spacing_xyz = _require_tensor(sample, "spacing_xyz").tolist()
        spacing_zyx = tuple(reversed(spacing_xyz))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        logits = sliding_window_logits(
            model=model,
            image=image,
            patch_size=tuple(arguments.patch_size),
            overlap=arguments.overlap,
            window_batch_size=arguments.window_batch_size,
            use_amp=use_amp,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_start
        prediction = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        classes = evaluate_segmentation(
            prediction=prediction,
            target=target,
            num_classes=NUM_CLASSES,
            spacing_zyx=spacing_zyx,
            calculate_hd95=arguments.hd95,
        )
        patient_id = _require_string(sample, "patient_id")
        visit = _require_string(sample, "visit")
        case_name = f"{patient_id}_{visit}"
        prediction_path = None
        if arguments.save_predictions:
            prediction_path = prediction_directory / f"{case_name}_PREDICTION.nii.gz"
            _save_prediction(
                prediction,
                _require_string(sample, "image_path"),
                prediction_path,
            )
        case_result = {
            "case": case_name,
            "patient_id": patient_id,
            "visit": visit,
            "inference_seconds": inference_seconds,
            "prediction_path": None if prediction_path is None else str(prediction_path),
            "classes": classes,
        }
        case_results.append(case_result)
        print(json.dumps(case_result, indent=2), flush=True)

    aggregate_classes = aggregate_case_metrics(case_results, NUM_CLASSES)
    foreground_dice = [
        entry["dice"]
        for entry in aggregate_classes[1:]
        if entry["dice"] is not None
    ]
    result = {
        "checkpoint": str(arguments.checkpoint.expanduser().resolve()),
        "model": model_name,
        "base_channels": base_channels,
        "split": arguments.split,
        "cases": len(case_results),
        "patch_size": list(arguments.patch_size),
        "overlap": arguments.overlap,
        "window_batch_size": arguments.window_batch_size,
        "device": str(device),
        "mixed_precision": use_amp,
        "mean_inference_seconds": float(
            np.mean([case["inference_seconds"] for case in case_results])
        ),
        "mean_foreground_dice": float(np.mean(foreground_dice)),
        "aggregate_classes": aggregate_classes,
        "case_results": case_results,
    }
    metrics_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def _scan_starts(image_size: int, window_size: int, overlap: float) -> list[int]:
    if image_size < window_size:
        raise ValueError(
            f"Image dimension {image_size} is smaller than window {window_size}"
        )
    stride = max(int(window_size * (1.0 - overlap)), 1)
    starts = list(range(0, image_size - window_size + 1, stride))
    final_start = image_size - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _pad_image(
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    padding = []
    before_values = []
    for dimension, patch in reversed(tuple(zip(image.shape[2:], patch_size, strict=True))):
        missing = max(patch - dimension, 0)
        before = missing // 2
        after = missing - before
        padding.extend((before, after))
        before_values.append(before)
    padded = functional.pad(image, padding, mode="constant", value=0)
    return padded, tuple(reversed(before_values))


def _gaussian_importance_map(
    patch_size: tuple[int, int, int],
    device: torch.device,
    sigma_scale: float = 0.125,
) -> torch.Tensor:
    axes = []
    for size in patch_size:
        coordinates = torch.arange(size, dtype=torch.float32, device=device)
        coordinates -= (size - 1) / 2.0
        sigma = size * sigma_scale
        axes.append(torch.exp(-0.5 * (coordinates / sigma) ** 2))
    importance = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )
    importance /= importance.max()
    importance = importance.clamp_min(1e-3)
    return importance[None, None]


def _checkpoint_configuration(checkpoint: dict[str, Any]) -> tuple[str, int, int]:
    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Checkpoint has no valid configuration")
    model_name = checkpoint.get("model_name")
    if model_name not in ("baseline", "proposed"):
        raise ValueError(f"Checkpoint has invalid model name: {model_name}")
    base_channels = configuration.get("base_channels")
    seed = configuration.get("seed")
    if not isinstance(base_channels, int) or base_channels <= 0:
        raise ValueError(f"Checkpoint has invalid base_channels: {base_channels}")
    if not isinstance(seed, int):
        raise ValueError(f"Checkpoint has invalid seed: {seed}")
    return model_name, base_channels, seed


def _create_model(model_name: str, base_channels: int) -> nn.Module:
    if model_name == "baseline":
        return UNet3D(base_channels=base_channels)
    return ResidualUNet3D(base_channels=base_channels)


def _save_prediction(
    prediction_zyx: np.ndarray,
    source_image_path: str,
    output_path: Path,
) -> None:
    source = nib.load(source_image_path)
    prediction_xyz = np.ascontiguousarray(prediction_zyx.transpose(2, 1, 0))
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(prediction_xyz, source.affine, header)
    nib.save(output, output_path)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _require_tensor(sample: dict[str, object], key: str) -> torch.Tensor:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Sample field {key} must be a tensor")
    return value


def _require_string(sample: dict[str, object], key: str) -> str:
    value = sample[key]
    if not isinstance(value, str):
        raise TypeError(f"Sample field {key} must be a string")
    return value


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_cases is not None and arguments.max_cases <= 0:
        raise ValueError(f"max_cases must be positive, received {arguments.max_cases}")


def main() -> None:
    arguments = build_argument_parser().parse_args()
    evaluate_checkpoint(arguments)


if __name__ == "__main__":
    main()

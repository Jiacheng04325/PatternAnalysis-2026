"""Evaluate a slice-wise 2D U-Net on complete held-out HipMRI volumes.

Author: Jiacheng Yu (s4916091).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from dataset import HipMRIDataset, discover_volume_pairs, split_by_patient
from metrics import aggregate_case_metrics, evaluate_segmentation
from modules import UNet2D
from train import _resolve_device
from visualization import save_segmentation_figure


NUM_CLASSES = 6
MODEL_NAME = "baseline_2d"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--slice-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hd95", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int)
    return parser


def predict_volume(
    model: UNet2D,
    image_volume: torch.Tensor,
    slice_batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    if image_volume.ndim != 4 or image_volume.shape[0] != 1:
        raise ValueError(
            "Expected image shape [1, depth, height, width], "
            f"received {tuple(image_volume.shape)}"
        )
    image_slices = image_volume.movedim(1, 0)
    predictions = []
    with torch.inference_mode():
        for start in range(0, image_slices.shape[0], slice_batch_size):
            image = image_slices[start : start + slice_batch_size]
            with torch.amp.autocast(
                device_type=image.device.type,
                enabled=use_amp,
            ):
                logits = model(image)[0]
            predictions.append(torch.argmax(logits, dim=1).cpu())
    if not predictions:
        raise RuntimeError("Slice-wise inference generated no predictions")
    return torch.cat(predictions).numpy().astype(np.uint8, copy=False)


def evaluate_checkpoint(arguments: argparse.Namespace) -> None:
    _validate_arguments(arguments)
    device = _resolve_device(arguments.device)
    use_amp = arguments.amp and device.type == "cuda"
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=True,
    )
    base_channels, seed = _checkpoint_configuration(checkpoint)
    model = UNet2D(base_channels=base_channels).to(device)
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
    figure_directory = output_directory / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    if arguments.save_predictions:
        prediction_directory.mkdir(parents=True, exist_ok=True)
    if arguments.save_figures:
        figure_directory.mkdir(parents=True, exist_ok=True)

    case_results = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image_tensor = _require_tensor(sample, "image")
        image_volume = image_tensor[0].numpy()
        image = image_tensor.to(device)
        target = _require_tensor(sample, "label").numpy()
        spacing_xyz = _require_tensor(sample, "spacing_xyz").tolist()
        spacing_zyx = tuple(reversed(spacing_xyz))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        prediction = predict_volume(
            model,
            image,
            arguments.slice_batch_size,
            use_amp,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_start
        classes = evaluate_segmentation(
            prediction,
            target,
            NUM_CLASSES,
            spacing_zyx,
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
        figure_path = None
        figure_slice = None
        if arguments.save_figures:
            figure_path = figure_directory / f"{case_name}_COMPARISON.png"
            figure_slice = save_segmentation_figure(
                image_volume,
                target,
                prediction,
                figure_path,
                focus_class=5,
            )
        case_result = {
            "case": case_name,
            "patient_id": patient_id,
            "visit": visit,
            "inference_seconds": inference_seconds,
            "prediction_path": None if prediction_path is None else str(prediction_path),
            "figure_path": None if figure_path is None else str(figure_path),
            "figure_slice_zyx": figure_slice,
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
        "model": MODEL_NAME,
        "base_channels": base_channels,
        "split": arguments.split,
        "cases": len(case_results),
        "slice_batch_size": arguments.slice_batch_size,
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


def _checkpoint_configuration(checkpoint: dict[str, Any]) -> tuple[int, int]:
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Checkpoint model must be {MODEL_NAME}, received {checkpoint.get('model_name')}"
        )
    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Checkpoint has no valid configuration")
    base_channels = configuration.get("base_channels")
    seed = configuration.get("seed")
    if not isinstance(base_channels, int) or base_channels <= 0:
        raise ValueError(f"Checkpoint has invalid base_channels: {base_channels}")
    if not isinstance(seed, int):
        raise ValueError(f"Checkpoint has invalid seed: {seed}")
    return base_channels, seed


def _save_prediction(
    prediction_zyx: np.ndarray,
    source_image_path: str,
    output_path: Path,
) -> None:
    source = nib.load(source_image_path)
    prediction_xyz = np.ascontiguousarray(prediction_zyx.transpose(2, 1, 0))
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(prediction_xyz, source.affine, header), output_path)


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
    if arguments.slice_batch_size <= 0:
        raise ValueError(
            f"slice_batch_size must be positive, received {arguments.slice_batch_size}"
        )
    if arguments.max_cases is not None and arguments.max_cases <= 0:
        raise ValueError(f"max_cases must be positive, received {arguments.max_cases}")


def main() -> None:
    arguments = build_argument_parser().parse_args()
    evaluate_checkpoint(arguments)


if __name__ == "__main__":
    main()

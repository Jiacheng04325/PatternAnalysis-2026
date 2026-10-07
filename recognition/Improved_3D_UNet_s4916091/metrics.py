from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage


def evaluate_segmentation(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    spacing_zyx: Sequence[float],
    calculate_hd95: bool = True,
) -> list[dict[str, Any]]:
    _validate_inputs(prediction, target, num_classes, spacing_zyx)
    results = []
    for class_id in range(num_classes):
        predicted_mask = prediction == class_id
        target_mask = target == class_id
        predicted_voxels = int(predicted_mask.sum())
        target_voxels = int(target_mask.sum())
        intersection = int(np.logical_and(predicted_mask, target_mask).sum())
        union = predicted_voxels + target_voxels - intersection
        denominator = predicted_voxels + target_voxels

        if denominator == 0:
            dice = None
            overlap_status = "both_empty"
        else:
            dice = 2.0 * intersection / denominator
            overlap_status = "defined"
        iou = None if union == 0 else intersection / union

        hd95_mm = None
        hd95_status = "not_requested"
        if calculate_hd95 and class_id > 0:
            hd95_mm, hd95_status = hausdorff_distance_95(
                predicted_mask,
                target_mask,
                spacing_zyx,
            )

        results.append(
            {
                "class_id": class_id,
                "predicted_voxels": predicted_voxels,
                "target_voxels": target_voxels,
                "intersection": intersection,
                "union": union,
                "denominator": denominator,
                "dice": dice,
                "iou": iou,
                "overlap_status": overlap_status,
                "hd95_mm": hd95_mm,
                "hd95_status": hd95_status,
            }
        )
    return results


def hausdorff_distance_95(
    predicted_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing_zyx: Sequence[float],
) -> tuple[float | None, str]:
    if predicted_mask.shape != target_mask.shape:
        raise ValueError(
            f"Mask shapes differ: prediction={predicted_mask.shape}, "
            f"target={target_mask.shape}"
        )
    predicted_present = bool(predicted_mask.any())
    target_present = bool(target_mask.any())
    if not predicted_present and not target_present:
        return None, "both_empty"
    if predicted_present != target_present:
        return None, "one_empty"

    structure = ndimage.generate_binary_structure(predicted_mask.ndim, 1)
    predicted_surface = np.logical_and(
        predicted_mask,
        np.logical_not(
            ndimage.binary_erosion(
                predicted_mask,
                structure=structure,
                border_value=0,
            )
        ),
    )
    target_surface = np.logical_and(
        target_mask,
        np.logical_not(
            ndimage.binary_erosion(
                target_mask,
                structure=structure,
                border_value=0,
            )
        ),
    )
    distance_to_target = ndimage.distance_transform_edt(
        np.logical_not(target_surface),
        sampling=spacing_zyx,
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        np.logical_not(predicted_surface),
        sampling=spacing_zyx,
    )
    surface_distances = np.concatenate(
        (
            distance_to_target[predicted_surface],
            distance_to_prediction[target_surface],
        )
    )
    return float(np.percentile(surface_distances, 95)), "defined"


def aggregate_case_metrics(
    cases: Sequence[dict[str, Any]],
    num_classes: int,
) -> list[dict[str, Any]]:
    if not cases:
        raise ValueError("At least one case is required for metric aggregation")
    aggregated = []
    for class_id in range(num_classes):
        class_entries = [case["classes"][class_id] for case in cases]
        intersection = sum(entry["intersection"] for entry in class_entries)
        denominator = sum(entry["denominator"] for entry in class_entries)
        union = sum(entry["union"] for entry in class_entries)
        hd95_values = [
            entry["hd95_mm"]
            for entry in class_entries
            if entry["hd95_mm"] is not None
        ]
        hd95_status_counts = {
            status: sum(entry["hd95_status"] == status for entry in class_entries)
            for status in ("defined", "both_empty", "one_empty", "not_requested")
        }
        aggregated.append(
            {
                "class_id": class_id,
                "dice": None if denominator == 0 else 2.0 * intersection / denominator,
                "iou": None if union == 0 else intersection / union,
                "mean_hd95_mm": (
                    None if not hd95_values else float(np.mean(hd95_values))
                ),
                "median_hd95_mm": (
                    None if not hd95_values else float(np.median(hd95_values))
                ),
                "hd95_status_counts": hd95_status_counts,
            }
        )
    return aggregated


def _validate_inputs(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    spacing_zyx: Sequence[float],
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Segmentation shapes differ: prediction={prediction.shape}, "
            f"target={target.shape}"
        )
    if prediction.ndim != 3:
        raise ValueError(f"Expected 3D segmentations, received {prediction.shape}")
    if num_classes <= 1:
        raise ValueError(f"num_classes must exceed one, received {num_classes}")
    if len(spacing_zyx) != 3 or any(spacing <= 0 for spacing in spacing_zyx):
        raise ValueError(f"Invalid voxel spacing: {spacing_zyx}")
    for name, values in (("prediction", prediction), ("target", target)):
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} must contain integer labels, received {values.dtype}")
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < 0 or maximum >= num_classes:
            raise ValueError(
                f"{name} range [{minimum}, {maximum}] is outside "
                f"[0, {num_classes - 1}]"
            )

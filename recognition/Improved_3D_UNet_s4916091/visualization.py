from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt


CLASS_NAMES = (
    "Background",
    "Body",
    "Bone",
    "Urinary bladder",
    "Rectum",
    "Prostate",
)
CLASS_COLOURS = (
    "#000000",
    "#d9d9d9",
    "#f0e442",
    "#56b4e9",
    "#009e73",
    "#cc79a7",
)


def plot_training_history(
    metrics_path: Path,
    output_path: Path,
    selection_class: int,
) -> None:
    if not 0 <= selection_class < len(CLASS_NAMES):
        raise ValueError(f"Invalid selection class: {selection_class}")
    records = _read_json_lines(metrics_path)
    if not records:
        raise ValueError(f"No training records found in {metrics_path}")

    epochs = [record["epoch"] for record in records]
    training_loss = [record["train"]["loss"] for record in records]
    validation_loss = [record["validation"]["loss"] for record in records]
    training_dice = [
        record["train"]["per_class_dice"][selection_class]
        for record in records
    ]
    validation_dice = [
        record["validation"]["per_class_dice"][selection_class]
        for record in records
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, training_loss, label="Train")
    axes[0].plot(epochs, validation_loss, label="Validation")
    axes[0].set_title("Dice and cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, training_dice, label="Train")
    axes[1].plot(epochs, validation_dice, label="Validation")
    axes[1].set_title(f"{CLASS_NAMES[selection_class]} Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_segmentation_figure(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    focus_class: int = 5,
) -> int:
    _validate_volumes(image, target, prediction, focus_class)
    slice_index = _select_slice(target, prediction, focus_class)
    image_slice = image[slice_index]
    target_slice = target[slice_index]
    prediction_slice = prediction[slice_index]
    display_image = _scale_image(image_slice)
    false_positive = np.logical_and(
        prediction_slice == focus_class,
        target_slice != focus_class,
    )
    false_negative = np.logical_and(
        target_slice == focus_class,
        prediction_slice != focus_class,
    )

    colour_map = ListedColormap(CLASS_COLOURS)
    figure, axes = plt.subplots(1, 5, figsize=(18, 4))
    for axis in axes:
        axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.axis("off")

    axes[0].set_title("Input MRI")
    _show_label_overlay(axes[1], target_slice, colour_map)
    axes[1].set_title("Ground truth")
    _show_label_overlay(axes[2], prediction_slice, colour_map)
    axes[2].set_title("Prediction")
    axes[3].imshow(
        np.ma.masked_where(~false_positive, false_positive),
        cmap=ListedColormap(("#ff0000",)),
        alpha=0.85,
    )
    axes[3].set_title("Prostate false positive")
    axes[4].imshow(
        np.ma.masked_where(~false_negative, false_negative),
        cmap=ListedColormap(("#00bfff",)),
        alpha=0.85,
    )
    axes[4].set_title("Prostate false negative")
    figure.suptitle(f"Axial slice {slice_index}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return slice_index


def save_failure_analysis_figure(
    image: np.ndarray,
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    proposed_prediction: np.ndarray,
    output_path: Path,
    focus_class: int = 5,
) -> dict[str, int]:
    _validate_volumes(image, target, baseline_prediction, focus_class)
    _validate_volumes(image, target, proposed_prediction, focus_class)
    selections = _select_failure_analysis_slices(
        target,
        baseline_prediction,
        proposed_prediction,
        focus_class,
    )
    colour_map = ListedColormap(CLASS_COLOURS)
    figure, axes = plt.subplots(
        len(selections),
        6,
        figsize=(22, 3.6 * len(selections)),
        squeeze=False,
    )
    column_titles = (
        "Input MRI",
        "Ground truth",
        "Baseline",
        "Improved",
        "Baseline errors",
        "Improved errors",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)

    for row, (selection_name, slice_index) in enumerate(selections.items()):
        display_image = _scale_image(image[slice_index])
        target_slice = target[slice_index]
        baseline_slice = baseline_prediction[slice_index]
        proposed_slice = proposed_prediction[slice_index]
        for axis in axes[row]:
            axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
        axes[row, 0].text(
            -0.06,
            0.5,
            f"{selection_name}\nz={slice_index}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
        _show_label_overlay(axes[row, 1], target_slice, colour_map)
        _show_label_overlay(axes[row, 2], baseline_slice, colour_map)
        _show_label_overlay(axes[row, 3], proposed_slice, colour_map)
        _show_error_overlay(
            axes[row, 4],
            target_slice,
            baseline_slice,
            focus_class,
        )
        _show_error_overlay(
            axes[row, 5],
            target_slice,
            proposed_slice,
            focus_class,
        )

    figure.suptitle(
        "Prostate failure analysis: red=false positive, cyan=false negative",
        y=1.0,
    )
    figure.tight_layout(rect=(0.08, 0.0, 1.0, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return selections


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected object at {path}:{line_number}, received {type(record)}"
                )
            records.append(record)
    return records


def _validate_volumes(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    focus_class: int,
) -> None:
    if image.shape != target.shape or image.shape != prediction.shape:
        raise ValueError(
            "Visualization volume shapes differ: "
            f"image={image.shape}, target={target.shape}, prediction={prediction.shape}"
        )
    if image.ndim != 3:
        raise ValueError(f"Expected 3D volumes, received {image.shape}")
    if not 0 < focus_class < len(CLASS_NAMES):
        raise ValueError(f"Invalid focus class: {focus_class}")


def _select_slice(
    target: np.ndarray,
    prediction: np.ndarray,
    focus_class: int,
) -> int:
    target_counts = np.sum(target == focus_class, axis=(1, 2))
    if target_counts.max() > 0:
        return int(np.argmax(target_counts))
    prediction_counts = np.sum(prediction == focus_class, axis=(1, 2))
    if prediction_counts.max() > 0:
        return int(np.argmax(prediction_counts))
    return target.shape[0] // 2


def _select_failure_analysis_slices(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    proposed_prediction: np.ndarray,
    focus_class: int,
) -> dict[str, int]:
    target_mask = target == focus_class
    target_counts = np.sum(target_mask, axis=(1, 2))
    target_slices = np.flatnonzero(target_counts)
    if target_slices.size == 0:
        raise ValueError(f"Focus class {focus_class} is absent from the target")

    selections: dict[str, int] = {}
    selected_indices: set[int] = set()

    def add_selection(name: str, index: int) -> None:
        if index not in selected_indices:
            selections[name] = index
            selected_indices.add(index)

    add_selection("Apex extent", int(target_slices[0]))
    add_selection("Largest target", int(np.argmax(target_counts)))
    add_selection("Base extent", int(target_slices[-1]))

    baseline_false_positive = np.logical_and(
        baseline_prediction == focus_class,
        ~target_mask,
    )
    proposed_false_positive = np.logical_and(
        proposed_prediction == focus_class,
        ~target_mask,
    )
    combined_false_positive = np.logical_or(
        baseline_false_positive,
        proposed_false_positive,
    )
    distance_from_target = distance_transform_edt(~target_mask)
    false_positive_distances = np.where(
        combined_false_positive,
        distance_from_target,
        0.0,
    )
    farthest_false_positive = np.max(false_positive_distances, axis=(1, 2))
    _add_highest_error_slice(
        selections,
        selected_indices,
        "Most distant false positive",
        farthest_false_positive,
    )

    baseline_false_negative = np.logical_and(
        target_mask,
        baseline_prediction != focus_class,
    )
    proposed_false_negative = np.logical_and(
        target_mask,
        proposed_prediction != focus_class,
    )
    false_negative_counts = np.maximum(
        np.sum(baseline_false_negative, axis=(1, 2)),
        np.sum(proposed_false_negative, axis=(1, 2)),
    )
    _add_highest_error_slice(
        selections,
        selected_indices,
        "Maximum false negative",
        false_negative_counts,
    )
    return selections


def _add_highest_error_slice(
    selections: dict[str, int],
    selected_indices: set[int],
    name: str,
    counts: np.ndarray,
) -> None:
    for index in np.argsort(counts)[::-1]:
        candidate = int(index)
        if counts[candidate] <= 0:
            return
        if candidate not in selected_indices:
            selections[name] = candidate
            selected_indices.add(candidate)
            return


def _scale_image(image: np.ndarray) -> np.ndarray:
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        raise ValueError("Visualization image has no finite values")
    lower, upper = np.percentile(finite_values, (1.0, 99.0))
    if not lower < upper:
        raise ValueError(
            f"Visualization intensity range is invalid: lower={lower}, upper={upper}"
        )
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def _show_label_overlay(
    axis: Any,
    labels: np.ndarray,
    colour_map: ListedColormap,
) -> None:
    overlay = np.ma.masked_where(labels == 0, labels)
    axis.imshow(
        overlay,
        cmap=colour_map,
        vmin=0,
        vmax=len(CLASS_NAMES) - 1,
        alpha=0.65,
        interpolation="nearest",
    )


def _show_error_overlay(
    axis: Any,
    target: np.ndarray,
    prediction: np.ndarray,
    focus_class: int,
) -> None:
    false_positive = np.logical_and(
        prediction == focus_class,
        target != focus_class,
    )
    false_negative = np.logical_and(
        target == focus_class,
        prediction != focus_class,
    )
    axis.imshow(
        np.ma.masked_where(~false_positive, false_positive),
        cmap=ListedColormap(("#ff0000",)),
        alpha=0.85,
    )
    axis.imshow(
        np.ma.masked_where(~false_negative, false_negative),
        cmap=ListedColormap(("#00bfff",)),
        alpha=0.85,
    )

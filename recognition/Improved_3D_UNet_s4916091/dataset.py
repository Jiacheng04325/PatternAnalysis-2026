from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset


IMAGE_SUFFIX = "_LFOV.nii.gz"
LABEL_SUFFIX = "_SEMANTIC.nii.gz"
CASE_PATTERN = re.compile(r"^(?P<patient_id>[A-Z]\d{3})_(?P<visit>Week\d+)$")
NUM_CLASSES = 6
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class VolumePair:
    patient_id: str
    visit: str
    image_path: Path
    label_path: Path


def discover_volume_pairs(data_root: str | Path) -> tuple[VolumePair, ...]:
    root = Path(data_root).expanduser().resolve()
    image_directory = root / "semantic_MRs"
    label_directory = root / "semantic_labels_only"

    if not image_directory.is_dir():
        raise FileNotFoundError(f"MRI directory not found: {image_directory}")
    if not label_directory.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_directory}")

    image_index = _index_cases(image_directory, IMAGE_SUFFIX)
    label_index = _index_cases(label_directory, LABEL_SUFFIX)
    missing_labels = sorted(image_index.keys() - label_index.keys())
    missing_images = sorted(label_index.keys() - image_index.keys())

    if missing_labels or missing_images:
        raise ValueError(
            "Unmatched HipMRI files: "
            f"missing_labels={missing_labels}, missing_images={missing_images}"
        )

    pairs = []
    for case_key in sorted(image_index):
        match = CASE_PATTERN.fullmatch(case_key)
        if match is None:
            raise ValueError(f"Unexpected HipMRI case name: {case_key}")
        pairs.append(
            VolumePair(
                patient_id=match.group("patient_id"),
                visit=match.group("visit"),
                image_path=image_index[case_key],
                label_path=label_index[case_key],
            )
        )

    if not pairs:
        raise ValueError(f"No HipMRI image-label pairs found under {root}")

    return tuple(pairs)


def split_by_patient(
    pairs: Sequence[VolumePair],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 3710,
) -> dict[str, tuple[VolumePair, ...]]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected three split ratios, received {ratios}")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError(f"Split ratios must be positive, received {ratios}")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, received {ratios}")

    grouped: dict[str, list[VolumePair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.patient_id].append(pair)

    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("At least three patients are required for patient-level splitting")

    rng = random.Random(seed)
    patient_items = [
        (patient_id, tuple(sorted(items, key=lambda item: item.visit)), rng.random())
        for patient_id, items in sorted(grouped.items())
    ]
    patient_items.sort(key=lambda item: (-len(item[1]), item[2], item[0]))

    target_volumes = [len(pairs) * ratio for ratio in ratios]
    target_patients = [len(grouped) * ratio for ratio in ratios]
    allocated: list[list[VolumePair]] = [[] for _ in SPLIT_NAMES]
    allocated_patients: list[set[str]] = [set() for _ in SPLIT_NAMES]

    for split_index, (patient_id, patient_pairs, _) in enumerate(
        patient_items[: len(SPLIT_NAMES)]
    ):
        allocated[split_index].extend(patient_pairs)
        allocated_patients[split_index].add(patient_id)

    for patient_id, patient_pairs, _ in patient_items[len(SPLIT_NAMES) :]:
        candidate_scores = []
        for split_index in range(len(SPLIT_NAMES)):
            projected_volume_fill = (
                len(allocated[split_index]) + len(patient_pairs)
            ) / target_volumes[split_index]
            projected_patient_fill = (
                len(allocated_patients[split_index]) + 1
            ) / target_patients[split_index]
            candidate_scores.append(
                (projected_volume_fill + 0.25 * projected_patient_fill, split_index)
            )
        selected_split = min(candidate_scores)[1]
        allocated[selected_split].extend(patient_pairs)
        allocated_patients[selected_split].add(patient_id)

    splits = {
        split_name: tuple(
            sorted(
                split_pairs,
                key=lambda pair: (pair.patient_id, pair.visit),
            )
        )
        for split_name, split_pairs in zip(SPLIT_NAMES, allocated, strict=True)
    }
    validate_patient_splits(splits, expected_pairs=pairs)
    return splits


def validate_patient_splits(
    splits: dict[str, Sequence[VolumePair]],
    expected_pairs: Sequence[VolumePair] | None = None,
) -> None:
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Expected split names {SPLIT_NAMES}, received {tuple(splits)}")

    patient_sets = {
        split_name: {pair.patient_id for pair in split_pairs}
        for split_name, split_pairs in splits.items()
    }
    for index, first_name in enumerate(SPLIT_NAMES):
        for second_name in SPLIT_NAMES[index + 1 :]:
            overlap = patient_sets[first_name] & patient_sets[second_name]
            if overlap:
                raise ValueError(
                    f"Patient leakage between {first_name} and {second_name}: "
                    f"{sorted(overlap)}"
                )

    split_cases = [
        (pair.patient_id, pair.visit)
        for split_name in SPLIT_NAMES
        for pair in splits[split_name]
    ]
    if len(split_cases) != len(set(split_cases)):
        raise ValueError("Duplicate cases detected across patient splits")

    if expected_pairs is not None:
        expected_cases = {(pair.patient_id, pair.visit) for pair in expected_pairs}
        actual_cases = set(split_cases)
        missing_cases = sorted(expected_cases - actual_cases)
        unexpected_cases = sorted(actual_cases - expected_cases)
        if missing_cases or unexpected_cases:
            raise ValueError(
                "Split coverage mismatch: "
                f"missing_cases={missing_cases}, unexpected_cases={unexpected_cases}"
            )


def write_split_manifest(
    splits: dict[str, Sequence[VolumePair]],
    output_path: str | Path,
) -> None:
    validate_patient_splits(splits)
    manifest = {
        split_name: [
            {
                "patient_id": pair.patient_id,
                "visit": pair.visit,
                "image": pair.image_path.name,
                "label": pair.label_path.name,
            }
            for pair in splits[split_name]
        ]
        for split_name in SPLIT_NAMES
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class HipMRIDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        pairs: Sequence[VolumePair],
        patch_size: tuple[int, int, int] | None = None,
        training: bool = False,
        foreground_labels: tuple[int, ...] = (3, 4, 5),
        foreground_probability: float = 0.75,
        augment: bool = False,
    ) -> None:
        if not pairs:
            raise ValueError("HipMRIDataset requires at least one image-label pair")
        if patch_size is not None and (
            len(patch_size) != 3 or any(size <= 0 for size in patch_size)
        ):
            raise ValueError(f"Invalid patch size: {patch_size}")
        if any(label <= 0 or label >= NUM_CLASSES for label in foreground_labels):
            raise ValueError(f"Invalid foreground labels: {foreground_labels}")
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError(
                "Foreground probability must be between 0 and 1, "
                f"received {foreground_probability}"
            )

        self.pairs = tuple(pairs)
        self.patch_size = patch_size
        self.training = training
        self.foreground_labels = foreground_labels
        self.foreground_probability = foreground_probability
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        image_nifti = nib.load(pair.image_path)
        label_nifti = nib.load(pair.label_path)

        if image_nifti.shape != label_nifti.shape:
            raise ValueError(
                f"Shape mismatch for {pair.patient_id}_{pair.visit}: "
                f"image={image_nifti.shape}, label={label_nifti.shape}"
            )
        if not np.allclose(image_nifti.affine, label_nifti.affine):
            raise ValueError(f"Affine mismatch for {pair.patient_id}_{pair.visit}")

        image_array = np.asarray(image_nifti.dataobj, dtype=np.float32)
        label_array = np.asarray(label_nifti.dataobj)
        _validate_arrays(image_array, label_array, pair)
        image_array = _normalise_image(image_array)

        image = torch.from_numpy(np.ascontiguousarray(image_array.transpose(2, 1, 0)))
        label = torch.from_numpy(
            np.ascontiguousarray(label_array.transpose(2, 1, 0))
        ).long()

        if self.patch_size is not None:
            image, label = self._extract_patch(image, label)
        if self.training and self.augment:
            image, label = _random_flip(image, label)

        spacing = tuple(float(value) for value in image_nifti.header.get_zooms()[:3])
        return {
            "image": image.unsqueeze(0),
            "label": label,
            "patient_id": pair.patient_id,
            "visit": pair.visit,
            "spacing_xyz": torch.tensor(spacing, dtype=torch.float32),
            "image_path": str(pair.image_path),
            "label_path": str(pair.label_path),
        }

    def _extract_patch(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.patch_size is None:
            raise RuntimeError("Patch extraction requires a configured patch size")

        image, label = _pad_to_patch_size(image, label, self.patch_size)
        if not self.training:
            starts = tuple(
                (dimension - patch) // 2
                for dimension, patch in zip(image.shape, self.patch_size, strict=True)
            )
            return _crop_patch(image, label, starts, self.patch_size)

        use_foreground = torch.rand(()) < self.foreground_probability
        if use_foreground:
            selected_label_index = int(torch.randint(len(self.foreground_labels), (1,)))
            selected_label = self.foreground_labels[selected_label_index]
            candidates = torch.nonzero(label == selected_label, as_tuple=False)
            if candidates.numel() == 0:
                raise ValueError(f"Foreground label {selected_label} is absent from the volume")
            center = candidates[int(torch.randint(len(candidates), (1,)))].tolist()
            starts = tuple(
                min(
                    max(int(center_value) - patch // 2, 0),
                    dimension - patch,
                )
                for center_value, dimension, patch in zip(
                    center,
                    image.shape,
                    self.patch_size,
                    strict=True,
                )
            )
        else:
            starts = tuple(
                int(torch.randint(dimension - patch + 1, (1,)))
                for dimension, patch in zip(image.shape, self.patch_size, strict=True)
            )

        return _crop_patch(image, label, starts, self.patch_size)


def _index_cases(directory: Path, suffix: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        case_key = path.name[: -len(suffix)]
        if case_key in index:
            raise ValueError(f"Duplicate HipMRI case key {case_key} in {directory}")
        index[case_key] = path
    return index


def _validate_arrays(
    image: np.ndarray,
    label: np.ndarray,
    pair: VolumePair,
) -> None:
    case_name = f"{pair.patient_id}_{pair.visit}"
    if image.ndim != 3 or label.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays for {case_name}, "
            f"received image={image.shape}, label={label.shape}"
        )
    if not np.isfinite(image).all():
        invalid_count = int(image.size - np.isfinite(image).sum())
        raise ValueError(f"Image {case_name} has {invalid_count} non-finite voxels")
    if not np.issubdtype(label.dtype, np.integer):
        if not np.equal(label, np.rint(label)).all():
            raise ValueError(f"Label {case_name} contains non-integer values")
        label[:] = np.rint(label)
    label_min = int(label.min())
    label_max = int(label.max())
    if label_min < 0 or label_max >= NUM_CLASSES:
        raise ValueError(
            f"Label range for {case_name} is [{label_min}, {label_max}], "
            f"expected [0, {NUM_CLASSES - 1}]"
        )


def _normalise_image(image: np.ndarray) -> np.ndarray:
    nonzero = image != 0
    if not nonzero.any():
        raise ValueError("MRI volume contains no non-zero voxels")
    values = image[nonzero]
    lower, upper = np.percentile(values, (0.5, 99.5))
    if not lower < upper:
        raise ValueError(
            f"MRI intensity percentiles are invalid: lower={lower}, upper={upper}"
        )
    clipped = np.clip(image, lower, upper)
    clipped_values = clipped[nonzero]
    mean = float(clipped_values.mean())
    standard_deviation = float(clipped_values.std())
    if standard_deviation <= 0:
        raise ValueError(f"MRI standard deviation is invalid: {standard_deviation}")
    normalised = (clipped - mean) / standard_deviation
    normalised[~nonzero] = 0
    return normalised.astype(np.float32, copy=False)


def _pad_to_patch_size(
    image: torch.Tensor,
    label: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    padding = []
    for dimension, patch in reversed(tuple(zip(image.shape, patch_size, strict=True))):
        missing = max(patch - dimension, 0)
        before = missing // 2
        after = missing - before
        padding.extend((before, after))
    if any(padding):
        image = functional.pad(image, padding, mode="constant", value=0)
        label = functional.pad(label, padding, mode="constant", value=0)
    return image, label


def _crop_patch(
    image: torch.Tensor,
    label: torch.Tensor,
    starts: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    slices = tuple(
        slice(start, start + size)
        for start, size in zip(starts, patch_size, strict=True)
    )
    return image[slices], label[slices]


def _random_flip(
    image: torch.Tensor,
    label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for dimension in range(3):
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(dimension,))
            label = torch.flip(label, dims=(dimension,))
    return image, label


def _summarise_splits(splits: dict[str, Sequence[VolumePair]]) -> dict[str, object]:
    return {
        split_name: {
            "patients": len({pair.patient_id for pair in split_pairs}),
            "volumes": len(split_pairs),
        }
        for split_name, split_pairs in splits.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=3710)
    arguments = parser.parse_args()

    pairs = discover_volume_pairs(arguments.data_root)
    splits = split_by_patient(pairs, seed=arguments.seed)
    summary = {
        "total_patients": len({pair.patient_id for pair in pairs}),
        "total_volumes": len(pairs),
        "splits": _summarise_splits(splits),
    }
    print(json.dumps(summary, indent=2))
    if arguments.manifest is not None:
        write_split_manifest(splits, arguments.manifest)


if __name__ == "__main__":
    main()

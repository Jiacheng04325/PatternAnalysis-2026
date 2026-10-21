"""Generate paired qualitative autopsies for selected HipMRI cases.

Author: Jiacheng Yu (s4916091).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from dataset import HipMRIDataset, VolumePair, discover_volume_pairs
from visualization import save_failure_analysis_figure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--proposed-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--focus-class", type=int, default=5)
    return parser


def create_failure_analysis(arguments: argparse.Namespace) -> None:
    output_directory = arguments.output_dir.expanduser().resolve()
    summary_path = output_directory / "failure_analysis.json"
    if summary_path.exists():
        raise FileExistsError(f"Failure analysis already exists: {summary_path}")

    pairs = discover_volume_pairs(arguments.data_root)
    pair_index = {
        f"{pair.patient_id}_{pair.visit}": pair
        for pair in pairs
    }
    requested_cases = list(dict.fromkeys(arguments.cases))
    missing_cases = sorted(set(requested_cases) - pair_index.keys())
    if missing_cases:
        raise ValueError(f"Unknown HipMRI cases: {missing_cases}")

    output_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for case_name in requested_cases:
        pair = pair_index[case_name]
        sample = HipMRIDataset((pair,), training=False)[0]
        image = _require_tensor(sample, "image")[0].numpy()
        target = _require_tensor(sample, "label").numpy()
        baseline_prediction = _load_prediction(
            arguments.baseline_predictions,
            pair,
        )
        proposed_prediction = _load_prediction(
            arguments.proposed_predictions,
            pair,
        )
        figure_path = output_directory / f"{case_name}_FAILURE_ANALYSIS.png"
        slices = save_failure_analysis_figure(
            image=image,
            target=target,
            baseline_prediction=baseline_prediction,
            proposed_prediction=proposed_prediction,
            output_path=figure_path,
            focus_class=arguments.focus_class,
        )
        result = {
            "case": case_name,
            "figure_path": str(figure_path),
            "slices_zyx": slices,
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "focus_class": arguments.focus_class,
        "baseline_predictions": str(
            arguments.baseline_predictions.expanduser().resolve()
        ),
        "proposed_predictions": str(
            arguments.proposed_predictions.expanduser().resolve()
        ),
        "cases": results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_prediction(directory: Path, pair: VolumePair) -> np.ndarray:
    case_name = f"{pair.patient_id}_{pair.visit}"
    prediction_path = (
        directory.expanduser().resolve()
        / f"{case_name}_PREDICTION.nii.gz"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction not found: {prediction_path}")

    prediction_nifti = nib.load(prediction_path)
    source_nifti = nib.load(pair.image_path)
    if prediction_nifti.shape != source_nifti.shape:
        raise ValueError(
            f"Prediction shape mismatch for {case_name}: "
            f"prediction={prediction_nifti.shape}, source={source_nifti.shape}"
        )
    if not np.allclose(prediction_nifti.affine, source_nifti.affine):
        raise ValueError(f"Prediction affine mismatch for {case_name}")

    prediction_xyz = np.asarray(prediction_nifti.dataobj)
    if not np.issubdtype(prediction_xyz.dtype, np.integer):
        if not np.equal(prediction_xyz, np.rint(prediction_xyz)).all():
            raise ValueError(f"Prediction contains non-integer values for {case_name}")
        prediction_xyz = np.rint(prediction_xyz)
    return np.ascontiguousarray(prediction_xyz.transpose(2, 1, 0)).astype(
        np.int64,
        copy=False,
    )


def _require_tensor(sample: dict[str, object], key: str) -> torch.Tensor:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Sample field {key} must be a tensor")
    return value


def main() -> None:
    arguments = build_argument_parser().parse_args()
    create_failure_analysis(arguments)


if __name__ == "__main__":
    main()

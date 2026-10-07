from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import nn

from modules import SegmentationOutput


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        cross_entropy_weight: float = 0.5,
        class_weights: Sequence[float] | None = None,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if dice_weight < 0 or cross_entropy_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if dice_weight + cross_entropy_weight <= 0:
            raise ValueError("At least one loss weight must be positive")
        if smooth <= 0:
            raise ValueError(f"smooth must be positive, received {smooth}")
        weights = (
            None
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32)
        )
        if weights is not None and (weights.ndim != 1 or torch.any(weights <= 0)):
            raise ValueError("class_weights must be a positive one-dimensional sequence")
        self.register_buffer("class_weights", weights)
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_logits_and_target(logits, target)
        if self.class_weights is not None and self.class_weights.numel() != logits.shape[1]:
            raise ValueError(
                f"Expected {logits.shape[1]} class weights, "
                f"received {self.class_weights.numel()}"
            )
        cross_entropy = functional.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
        )
        probabilities = torch.softmax(logits, dim=1)
        one_hot = functional.one_hot(target, num_classes=logits.shape[1])
        one_hot = one_hot.movedim(-1, 1).to(dtype=probabilities.dtype)
        reduction_dimensions = (0, 2, 3, 4)
        intersection = torch.sum(probabilities * one_hot, dim=reduction_dimensions)
        denominator = torch.sum(
            probabilities + one_hot,
            dim=reduction_dimensions,
        )
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        dice_loss = 1.0 - dice.mean()
        total_weight = self.dice_weight + self.cross_entropy_weight
        return (
            self.dice_weight * dice_loss
            + self.cross_entropy_weight * cross_entropy
        ) / total_weight


class DeepSupervisionLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        auxiliary_weights: Sequence[float] = (0.125, 0.25, 0.5),
    ) -> None:
        super().__init__()
        if any(weight < 0 for weight in auxiliary_weights):
            raise ValueError("Auxiliary loss weights must be non-negative")
        self.base_loss = base_loss
        self.auxiliary_weights = tuple(float(weight) for weight in auxiliary_weights)

    def forward(
        self,
        outputs: SegmentationOutput,
        target: torch.Tensor,
    ) -> torch.Tensor:
        logits, auxiliary_logits = outputs
        if auxiliary_logits and len(auxiliary_logits) != len(self.auxiliary_weights):
            raise ValueError(
                f"Expected {len(self.auxiliary_weights)} auxiliary outputs, "
                f"received {len(auxiliary_logits)}"
            )
        loss = self.base_loss(logits, target)
        active_weights = self.auxiliary_weights[: len(auxiliary_logits)]
        for weight, auxiliary in zip(
            active_weights,
            auxiliary_logits,
            strict=True,
        ):
            loss = loss + weight * self.base_loss(auxiliary, target)
        return loss / (1.0 + sum(active_weights))


def dice_statistics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_logits_and_target(logits, target)
    prediction = torch.argmax(logits, dim=1)
    prediction_one_hot = functional.one_hot(
        prediction,
        num_classes=logits.shape[1],
    ).movedim(-1, 1)
    target_one_hot = functional.one_hot(
        target,
        num_classes=logits.shape[1],
    ).movedim(-1, 1)
    reduction_dimensions = (0, 2, 3, 4)
    intersection_twice = 2 * torch.sum(
        prediction_one_hot * target_one_hot,
        dim=reduction_dimensions,
    )
    denominator = torch.sum(
        prediction_one_hot + target_one_hot,
        dim=reduction_dimensions,
    )
    return intersection_twice.to(torch.float64), denominator.to(torch.float64)


def dice_from_statistics(
    intersection_twice: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    if intersection_twice.shape != denominator.shape:
        raise ValueError(
            "Dice statistic shapes differ: "
            f"intersection={intersection_twice.shape}, denominator={denominator.shape}"
        )
    missing = denominator == 0
    dice = intersection_twice / denominator.clamp_min(1)
    return dice.masked_fill(missing, torch.nan)


def _validate_logits_and_target(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if logits.ndim != 5:
        raise ValueError(
            "Expected logits shape [batch, classes, depth, height, width], "
            f"received {tuple(logits.shape)}"
        )
    expected_target_shape = (logits.shape[0], *logits.shape[2:])
    if target.shape != expected_target_shape:
        raise ValueError(
            f"Expected target shape {expected_target_shape}, received {tuple(target.shape)}"
        )
    if target.dtype != torch.long:
        raise TypeError(f"Expected torch.long target, received {target.dtype}")
    if target.numel() == 0:
        raise ValueError("Target cannot be empty")
    target_min = int(target.min())
    target_max = int(target.max())
    if target_min < 0 or target_max >= logits.shape[1]:
        raise ValueError(
            f"Target range [{target_min}, {target_max}] is outside "
            f"[0, {logits.shape[1] - 1}]"
        )

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import nn


SegmentationOutput = tuple[torch.Tensor, tuple[torch.Tensor, ...]]


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)
        outputs = self.activation(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.activation(outputs + residual)


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        block_factory: Callable[[int, int], nn.Module],
    ) -> None:
        super().__init__()
        blocks = []
        current_channels = in_channels
        for out_channels in channels:
            blocks.append(block_factory(current_channels, out_channels))
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = []
        outputs = inputs
        for index, block in enumerate(self.blocks):
            outputs = block(outputs)
            features.append(outputs)
            if index < len(self.blocks) - 1:
                outputs = self.pool(outputs)
        return tuple(features)


class DecoderStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block_factory: Callable[[int, int], nn.Module],
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.block = block_factory(out_channels + skip_channels, out_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.upsample(inputs)
        if outputs.shape[2:] != skip.shape[2:]:
            outputs = functional.interpolate(
                outputs,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return self.block(torch.cat((skip, outputs), dim=1))


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        _validate_configuration(in_channels, num_classes, base_channels)
        channels = tuple(base_channels * 2**index for index in range(5))
        self.encoder = Encoder3D(in_channels, channels, ConvBlock3D)
        self.decoder = _build_decoder(channels, ConvBlock3D)
        self.segmentation_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> SegmentationOutput:
        _validate_input(inputs)
        features = self.encoder(inputs)
        outputs = features[-1]
        for stage, skip in zip(self.decoder, reversed(features[:-1]), strict=True):
            outputs = stage(outputs, skip)
        return self.segmentation_head(outputs), ()


class ResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_channels: int = 16,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        _validate_configuration(in_channels, num_classes, base_channels)
        channels = tuple(base_channels * 2**index for index in range(5))
        self.encoder = Encoder3D(in_channels, channels, ResidualBlock3D)
        self.decoder = _build_decoder(channels, ResidualBlock3D)
        self.segmentation_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.deep_supervision = deep_supervision
        self.auxiliary_heads = nn.ModuleList(
            nn.Conv3d(channel, num_classes, kernel_size=1)
            for channel in reversed(channels[1:-1])
        )

    def forward(self, inputs: torch.Tensor) -> SegmentationOutput:
        _validate_input(inputs)
        features = self.encoder(inputs)
        outputs = features[-1]
        decoder_features = []
        for stage, skip in zip(self.decoder, reversed(features[:-1]), strict=True):
            outputs = stage(outputs, skip)
            decoder_features.append(outputs)

        logits = self.segmentation_head(outputs)
        if not self.deep_supervision:
            return logits, ()

        target_size = logits.shape[2:]
        auxiliary_logits = tuple(
            functional.interpolate(
                head(feature),
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            for head, feature in zip(
                self.auxiliary_heads,
                decoder_features[:-1],
                strict=True,
            )
        )
        return logits, auxiliary_logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _build_decoder(
    channels: tuple[int, ...],
    block_factory: Callable[[int, int], nn.Module],
) -> nn.ModuleList:
    stages = []
    for in_channels, skip_channels in zip(
        reversed(channels[1:]),
        reversed(channels[:-1]),
        strict=True,
    ):
        stages.append(
            DecoderStage3D(
                in_channels=in_channels,
                skip_channels=skip_channels,
                out_channels=skip_channels,
                block_factory=block_factory,
            )
        )
    return nn.ModuleList(stages)


def _validate_configuration(
    in_channels: int,
    num_classes: int,
    base_channels: int,
) -> None:
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, received {in_channels}")
    if num_classes <= 1:
        raise ValueError(f"num_classes must exceed one, received {num_classes}")
    if base_channels <= 0:
        raise ValueError(f"base_channels must be positive, received {base_channels}")


def _validate_input(inputs: torch.Tensor) -> None:
    if inputs.ndim != 5:
        raise ValueError(
            "Expected input shape [batch, channels, depth, height, width], "
            f"received {tuple(inputs.shape)}"
        )
    if min(inputs.shape[2:]) < 32:
        raise ValueError(
            f"Every spatial dimension must be at least 32, received {inputs.shape[2:]}"
        )

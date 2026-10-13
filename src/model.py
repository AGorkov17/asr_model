from typing import List

import torch


class Conv1dNet(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        kernels: List[int],
        strides: List[int],
        channels: List[int],
        activation: torch.nn.Module,
        hidden_size: int,
    ):
        super().__init__()

        features = in_features

        module_list = []

        for kernel_size, stride, chs in zip(kernels, strides, channels):

            module_list.extend(
                [
                    torch.nn.Conv1d(
                        in_channels=features,
                        out_channels=chs,
                        kernel_size=kernel_size,
                        stride=stride,
                        groups=chs,
                    ),
                    activation,
                    torch.nn.Conv1d(in_channels=chs, out_channels=chs, kernel_size=1),
                    torch.nn.BatchNorm1d(num_features=chs),
                    activation,
                    torch.nn.MaxPool1d(kernel_size=stride),
                ]
            )

            features = chs

        module_list.extend(
            [
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(features, hidden_size),
                activation,
                torch.nn.Linear(hidden_size, n_classes),
                torch.nn.LogSoftmax(-1),
            ]
        )

        self.model = torch.nn.Sequential(*module_list)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)


class DepthwiseResidualBlock(torch.nn.Module):
    """A cheap temporal residual block with a configurable receptive field."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            torch.nn.BatchNorm1d(channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(channels),
            torch.nn.Dropout(dropout),
        )
        self.activation = torch.nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class TinyTCResNet(torch.nn.Module):
    """A sub-10k-parameter residual temporal network for one-second KWS."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        channels: int = 32,
        stem_groups: int = 16,
        dilations: List[int] = (1, 2, 4, 8, 16),
        hidden_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        if in_features % stem_groups != 0 or channels % stem_groups != 0:
            raise ValueError(
                "in_features and channels must be divisible by stem_groups"
            )

        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_features,
                channels,
                kernel_size=5,
                stride=2,
                padding=2,
                groups=stem_groups,
                bias=False,
            ),
            torch.nn.BatchNorm1d(channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm1d(channels),
            torch.nn.SiLU(),
        )
        self.blocks = torch.nn.Sequential(
            *[
                DepthwiseResidualBlock(channels, dilation, dropout)
                for dilation in dilations
            ]
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * channels, hidden_size),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, n_classes),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(spectrogram))
        mean = features.mean(dim=-1)
        variance = (features - mean.unsqueeze(-1)).square().mean(dim=-1)
        std = torch.sqrt(variance.clamp_min(1e-5))
        pooled = torch.cat([mean, std], dim=1)
        return self.classifier(pooled)

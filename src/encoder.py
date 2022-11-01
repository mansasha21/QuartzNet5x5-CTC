import torch
from torch import nn


class QuartzNetBlock(torch.nn.Module):
    def __init__(
        self,
        feat_in: int,
        filters: int,
        repeat: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        residual: bool,
        separable: bool,
        dropout: float,
    ):

        super().__init__()
        if dilation > 1:
            padding = (dilation * kernel_size) // 2 - 1
        else:
            padding = kernel_size // 2

        if residual:
            self.res = nn.Sequential(
                nn.Conv1d(feat_in, filters, kernel_size=1), nn.BatchNorm1d(filters),
            )

        layers = nn.ModuleList()
        for _ in range(repeat):
            if separable:
                layers.append(
                    nn.Conv1d(
                        feat_in,
                        feat_in,
                        kernel_size,
                        stride=stride,
                        dilation=dilation,
                        groups=feat_in,
                        padding=padding,
                    )
                )
                layers.append(nn.Conv1d(feat_in, filters, kernel_size=1))
            else:
                layers.append(nn.Conv1d(feat_in, filters, kernel_size))
            feat_in = filters

            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.residual = residual
        self.conv = nn.Sequential(*layers[:-2])
        self.out = nn.Sequential(*layers[-2:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            x = self.res(x) + self.conv(x)
        else:
            x = self.conv(x)
        return self.out(x)


class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride ** block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features)
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len

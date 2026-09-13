"""Compact causal TCN, GRU, and encoder-only Transformer. Direct 20-step heads."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation {name}")


class CausalConv1d(nn.Module):
    """Left-padded dilated convolution. Output at t uses inputs 0..t only."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(in_ch, out_ch, self.kernel_size, dilation=self.dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act = _activation(activation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.conv(x)))


class CompactTCN(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        channels: list[int],
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        width = int(channels[-1])
        self.proj = nn.Conv1d(n_in, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TCNResidualBlock(width, kernel_size, int(d), dropout, activation) for d in dilations]
        )
        self.head = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, :, -1])


class CompactGRU(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden_size: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        gru_drop = float(dropout) if self.n_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=n_in,
            hidden_size=hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=gru_drop,
        )
        self.head = nn.Linear(hidden_size, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _out, hidden = self.gru(x)
        return self.head(hidden[-1])


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CompactTransformer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
        pooling: str,
        lookback: int = 100,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.pooling = pooling
        self.input_proj = nn.Linear(n_in, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=lookback + 8)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * ff_multiplier),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        kwargs: dict[str, Any] = {"encoder_layer": layer, "num_layers": n_layers}
        try:
            self.encoder = nn.TransformerEncoder(**kwargs, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(**kwargs)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.input_proj(x))
        h = self.encoder(h)
        pooled = h[:, -1, :] if self.pooling == "last" else h.mean(dim=1)
        return self.head(pooled)


def build_model(
    architecture: str,
    config: dict[str, Any],
    *,
    n_in: int,
    n_out: int = 20,
    lookback: int = 100,
) -> nn.Module:
    arch = architecture.upper()
    if arch == "TCN":
        return CompactTCN(
            n_in=n_in,
            n_out=n_out,
            channels=[int(v) for v in config["channels"]],
            kernel_size=int(config["kernel_size"]),
            dilations=[int(v) for v in config["dilations"]],
            dropout=float(config["dropout"]),
            activation=str(config.get("activation", "relu")),
        )
    if arch == "GRU":
        return CompactGRU(
            n_in=n_in,
            n_out=n_out,
            hidden_size=int(config["hidden_size"]),
            n_layers=int(config["n_layers"]),
            dropout=float(config["dropout"]),
        )
    if arch in {"TRANSFORMER", "TR"}:
        return CompactTransformer(
            n_in=n_in,
            n_out=n_out,
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]),
            ff_multiplier=int(config["ff_multiplier"]),
            dropout=float(config["dropout"]),
            pooling=str(config.get("pooling", "last")),
            lookback=lookback,
        )
    raise ValueError(f"unknown architecture {architecture}")

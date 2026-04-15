"""
DexiNed — Dense Extreme Inception Network for Edge Detection.

Implementácia architektúry podľa:
  Soria, X. et al. "DexiNed: Dense Extreme Inception Network for Edge Detection" (2020).
  https://github.com/xavysp/DexiNed

Modul je importovateľný bez stiahnutých váh — váhy sa načítajú separátne cez load_state_dict().
Aktivácia: ReLU (kompatibilná s publikovanými váhami).
State-dict kľúče: block_1, block_2, dblock_3 .. dblock_6, side_1 .. side_6, fuse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + ReLU (každá zložka voliteľná)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_features) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConvBlock(nn.Module):
    """Dve za sebou idúce ConvBlock vrstvy (používané v block_1 a block_2)."""

    def __init__(
        self,
        in_features: int,
        mid_features: int,
        out_features: int | None = None,
        stride: tuple[int, int] = (1, 1),
        use_act: bool = True,
    ) -> None:
        super().__init__()
        if out_features is None:
            out_features = mid_features
        self.conv1 = ConvBlock(in_features, mid_features, stride=stride[0])
        self.conv2 = ConvBlock(mid_features, out_features, use_act=use_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class _DenseBlock(nn.Module):
    """
    Séria ConvBlock vrstiev kde výstup každej vrstvy je vstupom pre ďalšiu.
    Používaný v dblock_3 .. dblock_6.
    """

    def __init__(
        self,
        num_layers: int,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_features
        for _ in range(num_layers):
            layers.append(ConvBlock(current, out_features))
            current = out_features
        self.dense_block_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.dense_block_layers:
            x = layer(x)
        return x


class SingleConvBlock(nn.Module):
    """1×1 Conv2d + BatchNorm2d — používaný pre side output vetvy."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=1, stride=stride, bias=False
        )
        self.bn = nn.BatchNorm2d(out_features) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class DexiNed(nn.Module):
    """
    DexiNed model.

    Vstup:  float32 tensor (B, 3, H, W), hodnoty normalizované
    Výstup: list[Tensor] — 6 side outputs + 1 fused output, každý (B, 1, H, W)
            Posledný prvok (index -1) je fused výstup používaný na detekciu hrán.
            Hodnoty NIE SÚ sigmoid-ované — aplikuj sigmoid pred prahovovaním.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder bloky
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=(2, 2))
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256)
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Side output vetvy (každá: N kanálov → 1 kanál)
        self.side_1 = SingleConvBlock(64, 1, 1)
        self.side_2 = SingleConvBlock(128, 1, 1)
        self.side_3 = SingleConvBlock(256, 1, 1)
        self.side_4 = SingleConvBlock(512, 1, 1)
        self.side_5 = SingleConvBlock(512, 1, 1)
        self.side_6 = SingleConvBlock(256, 1, 1)

        # Fúzna vrstva: 6 side outputs → 1 výstup
        self.fuse = nn.Conv2d(6, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        H, W = x.shape[2], x.shape[3]

        # Forward cez encoder
        b1 = self.block_1(x)
        b2 = self.block_2(b1)
        b3 = self.dblock_3(self.maxpool(b2))
        b4 = self.dblock_4(self.maxpool(b3))
        b5 = self.dblock_5(self.maxpool(b4))
        b6 = self.dblock_6(self.maxpool(b5))

        # Side outputs — upsample na pôvodnú veľkosť
        def _up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                t, size=(H, W), mode="bilinear", align_corners=False
            )

        s1 = _up(self.side_1(b1))
        s2 = _up(self.side_2(b2))
        s3 = _up(self.side_3(b3))
        s4 = _up(self.side_4(b4))
        s5 = _up(self.side_5(b5))
        s6 = _up(self.side_6(b6))

        fused = self.fuse(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))

        return [s1, s2, s3, s4, s5, s6, fused]

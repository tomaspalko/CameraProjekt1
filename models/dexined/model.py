"""
DexiNed — Dense Extreme Inception Network for Edge Detection.

Architektúra prispôsobená presne podľa štruktúry stiahnutých váh (Google Drive).
State-dict kľúče: block_1, block_2, dblock_3..6, side_1..5,
                  pre_dense_2..6, up_block_1..6, block_cat.

Referencia:
  Soria, X. et al. "DexiNed: Dense Extreme Inception Network for Edge Detection" (2020).
  https://github.com/xavysp/DexiNed
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stavebné bloky
# ---------------------------------------------------------------------------

class _DenseLayer(nn.Module):
    """
    Jedna vrstva v dense bloku.
    Kľúče: conv1, norm1, conv2, norm2.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, 3, padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        return x


class _DenseBlock(nn.Sequential):
    """
    Séria _DenseLayer pomenovaných denselayer1, denselayer2, ...
    Kľúče: dblock_N.denselayerM.conv1/norm1/conv2/norm2.
    """

    def __init__(self, num_layers: int, in_features: int, out_features: int) -> None:
        super().__init__()
        for i in range(num_layers):
            in_f = in_features if i == 0 else out_features
            self.add_module(f"denselayer{i + 1}", _DenseLayer(in_f, out_features))


class DoubleConvBlock(nn.Module):
    """
    Dve za sebou idúce konvolúcie. Plochá štruktúra: conv1, bn1, conv2, bn2.
    """

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
        self.conv1 = nn.Conv2d(
            in_features, mid_features, 3, stride=stride[0], padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(
            mid_features, out_features, 3, stride=stride[1], padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_features)
        self.use_act = use_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        if self.use_act:
            x = F.relu(x, inplace=True)
        return x


class SingleConvBlock(nn.Module):
    """1×1 konvolúcia + voliteľná BN. Kľúče: conv, bn."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_features) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class UpConvBlock(nn.Module):
    """
    Postupné upsampling: log2(up_scale) trojíc [Conv1×1 + ReLU + ConvTranspose].
    Každá ConvTranspose robí presne x2 (stride=2, kernel=up_scale, padding=up_scale//2-1).
    Celkový upscale = 2^n = up_scale.

    Kľúče: features.0, features.2, features.3, features.5, ...
    (ReLU na nepárnych indexoch — bez parametrov.)
    """

    def __init__(self, in_features: int, up_scale: int) -> None:
        super().__init__()
        n = int(math.log2(up_scale))
        constant = 16
        padding = up_scale // 2 - 1   # zabezpečí presný x2 per-ConvTranspose
        layers: list[nn.Module] = []
        for i in range(n):
            in_f = in_features if i == 0 else constant
            out_f = 1 if i == n - 1 else constant
            layers.append(nn.Conv2d(in_f, out_f, 1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.ConvTranspose2d(out_f, out_f, up_scale, stride=2, padding=padding, bias=True)
            )
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ---------------------------------------------------------------------------
# Hlavný model
# ---------------------------------------------------------------------------

class DexiNed(nn.Module):
    """
    DexiNed model.

    Vstup:  float32 tensor (B, 3, H, W), ImageNet-normalizovaný.
    Výstup: list[Tensor] — 6 branch outputs + 1 fused, každý (B, 1, H, W).
            Posledný prvok (index -1) je fused výstup.
            Hodnoty NIE SÚ sigmoid-ované.
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Encoder ---
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=(2, 1))    # → H/2, 64ch
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)       # → H/2, 128ch

        # Dense encoder bloky (každý predchádza maxpool)
        self.dblock_3 = _DenseBlock(2, 128, 256)   # H/4,  256ch
        self.dblock_4 = _DenseBlock(3, 256, 512)   # H/8,  512ch
        self.dblock_5 = _DenseBlock(3, 512, 512)   # H/16, 512ch
        self.dblock_6 = _DenseBlock(3, 512, 256)   # H/16, 256ch (bez maxpool)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Residuálne projekcie pred dense blokmi ---
        # pre_dense_N mapuje vstup dense bloku N na rovnaký počet kanálov ako výstup,
        # aby sa dal pridať ako residuál (skip connection).
        self.pre_dense_2 = SingleConvBlock(128, 256, stride=2)  # cross-block skip: H/4 → H/8
        self.pre_dense_3 = SingleConvBlock(128, 256)            # residuál pre dblock_3 (H/4)
        self.pre_dense_4 = SingleConvBlock(256, 512)            # residuál pre dblock_4 (H/8)
        self.pre_dense_5 = SingleConvBlock(512, 512)            # residuál pre dblock_5 (H/16)
        self.pre_dense_6 = SingleConvBlock(512, 256)            # residuál pre dblock_6 (H/16)

        # --- Side-output projekcie so stride=2 pre priestorové prispôsobenie ---
        self.side_1 = SingleConvBlock(64,  128, stride=2)  # H/2 → H/4
        self.side_2 = SingleConvBlock(128, 256, stride=2)  # H/4 → H/8
        self.side_3 = SingleConvBlock(256, 512, stride=2)  # H/8 → H/16
        self.side_4 = SingleConvBlock(512, 512)            # H/16 → H/16 (bez stride)
        self.side_5 = SingleConvBlock(512, 256)            # nepoužívané v inferencii

        # --- Upsampling bloky → 1-kanálový výstup pri H/2 ---
        self.up_block_1 = UpConvBlock(64,  up_scale=2)
        self.up_block_2 = UpConvBlock(128, up_scale=2)
        self.up_block_3 = UpConvBlock(256, up_scale=4)
        self.up_block_4 = UpConvBlock(512, up_scale=8)
        self.up_block_5 = UpConvBlock(512, up_scale=16)
        self.up_block_6 = UpConvBlock(256, up_scale=16)

        # --- Fúzna vrstva: 6 → 1 kanál ---
        self.block_cat = SingleConvBlock(6, 1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        H, W = x.shape[2], x.shape[3]

        # --- Block 1 a 2 ---
        b1 = self.block_1(x)                          # B, 64,  H/2, W/2
        b1_side = self.side_1(b1)                     # B, 128, H/4, W/4  (stride=2)

        b2 = self.block_2(b1)                         # B, 128, H/2, W/2
        b2_down = self.maxpool(b2)                    # B, 128, H/4, W/4
        b2_add = b2_down + b1_side                    # B, 128, H/4, W/4
        b2_side = self.side_2(b2_add)                 # B, 256, H/8, W/8  (stride=2)

        # --- Block 3 (dense) ---
        b3_pre = self.pre_dense_3(b2_down)            # B, 256, H/4, W/4
        b3 = self.dblock_3(b2_add) + b3_pre           # B, 256, H/4, W/4
        b3_down = self.maxpool(b3)                    # B, 256, H/8, W/8
        b3_add = b3_down + b2_side                    # B, 256, H/8, W/8
        b3_side = self.side_3(b3_add)                 # B, 512, H/16, W/16 (stride=2)

        # --- Block 4 (dense, cross-block skip z b2_down) ---
        b2_resize = self.pre_dense_2(b2_down)         # B, 256, H/8, W/8  (stride=2)
        b4_pre = self.pre_dense_4(b3_down + b2_resize)  # B, 512, H/8, W/8
        b4 = self.dblock_4(b3_add) + b4_pre           # B, 512, H/8, W/8
        b4_down = self.maxpool(b4)                    # B, 512, H/16, W/16
        b4_add = b4_down + b3_side                    # B, 512, H/16, W/16
        b4_side = self.side_4(b4_add)                 # B, 512, H/16, W/16 (stride=1)

        # --- Block 5 (dense) ---
        b5_pre = self.pre_dense_5(b4_down)            # B, 512, H/16, W/16
        b5 = self.dblock_5(b4_add) + b5_pre           # B, 512, H/16, W/16
        b5_add = b5 + b4_side                         # B, 512, H/16, W/16

        # --- Block 6 (dense) ---
        b6_pre = self.pre_dense_6(b5)                 # B, 256, H/16, W/16
        b6 = self.dblock_6(b5_add) + b6_pre           # B, 256, H/16, W/16

        # --- Upsampling každej vetvy na pôvodnú veľkosť ---
        def _up(t: torch.Tensor) -> torch.Tensor:
            return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)

        o1 = _up(self.up_block_1(b1))   # B, 1, H, W
        o2 = _up(self.up_block_2(b2))   # B, 1, H, W
        o3 = _up(self.up_block_3(b3))   # B, 1, H, W
        o4 = _up(self.up_block_4(b4))   # B, 1, H, W
        o5 = _up(self.up_block_5(b5))   # B, 1, H, W
        o6 = _up(self.up_block_6(b6))   # B, 1, H, W

        # --- Fúzia ---
        fused = self.block_cat(torch.cat([o1, o2, o3, o4, o5, o6], dim=1))

        return [o1, o2, o3, o4, o5, o6, fused]

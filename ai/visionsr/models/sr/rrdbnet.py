"""RRDBNet — the backbone behind ESRGAN, Real-ESRGAN x4plus and BSRGAN.

Residual-in-Residual Dense Blocks: dense connections inside each block, residual
scaling (0.2) at two levels to keep deep-network training stable.

The network always upsamples 4x internally. Scales 1 and 2 are reached by
pixel-unshuffling the input first (2x -> shuffle by 2 -> 4x conv -> net 2x), which
is why ``num_in_ch`` is multiplied below — the published checkpoints are shaped
this way and the state dict will not load otherwise.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from ...core.registry import register_architecture


class ResidualDenseBlock(nn.Module):
    """Five convs, each seeing every previous output. Growth channels stay small."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            init.kaiming_normal_(conv.weight, a=0, mode="fan_in")
            conv.weight.data *= 0.1  # ESRGAN's residual-scaling init
            if conv.bias is not None:
                init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # 0.2 is the residual scale from the ESRGAN paper — not a tunable here,
        # the pretrained weights assume it.
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Three dense blocks with an outer residual."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


@register_architecture("rrdbnet")
class RRDBNet(nn.Module):
    """ESRGAN / Real-ESRGAN generator.

    Args:
        num_in_ch: input channels (3 for RGB).
        num_out_ch: output channels.
        scale: 1, 2 or 4. See module docstring for how 1 and 2 are realised.
        num_feat: width. 64 in every public checkpoint.
        num_block: RRDB count. 23 for x4plus, 6 for the anime variant.
        num_grow_ch: dense growth channels.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError(f"RRDBNet supports scale 1, 2 or 4, got {scale}.")

        self.scale = scale
        # Fold the spatial shortfall into channels so the fixed 4x tail still lands
        # on the requested scale.
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))

        # Nearest + conv, not transposed conv: avoids checkerboard artefacts.
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

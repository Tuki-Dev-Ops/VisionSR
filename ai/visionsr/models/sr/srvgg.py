"""SRVGGNetCompact — Real-ESRGAN's small, fast generator.

A plain VGG-style stack with a single pixel-shuffle tail and a global nearest-
neighbour residual. ~1.2M params against RRDBNet's 16.7M, so it is the model to
reach for on a 4GB card or for video/batch work where throughput dominates.

Backs the `realesr-general-x4v3` and `realesr-animevideov3` checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...core.registry import register_architecture


def _activation(act_type: str, num_feat: int) -> nn.Module:
    # A fresh instance per call: PReLU carries learnable parameters, so sharing
    # one object across layers would silently tie weights and break the load.
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unknown activation {act_type!r}; use relu, prelu or leakyrelu.")


@register_architecture("srvgg")
class SRVGGNetCompact(nn.Module):
    """Compact SR generator.

    Args:
        num_in_ch: input channels.
        num_out_ch: output channels.
        num_feat: width (64 in the public checkpoints).
        num_conv: body convs. 32 for general-x4v3, 16 for animevideov3.
        upscale: output scale factor.
        act_type: relu | prelu | leakyrelu.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.upscale = upscale

        body: list[nn.Module] = [
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            _activation(act_type, num_feat),
        ]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(_activation(act_type, num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

        # ModuleList, not Sequential: the published state dicts key on `body.N`,
        # and Sequential would produce the same keys only by coincidence of order.
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)

        # The net predicts a residual on top of a naive upsample, so it never has
        # to relearn the identity.
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base

"""GFPGAN — blind face restoration with a generative facial prior.

Why a prior at all: a face crop from an old or heavily compressed photo has lost
information a general SR network cannot invent — pore texture, eyelash structure,
iris detail. GFPGAN borrows it from a StyleGAN2 trained on FFHQ, then keeps the
result faithful to the actual person via SFT conditioning from a U-Net encoder.

Shape contract: input is a 512x512 aligned face crop, output is 512x512. Alignment
is the caller's job (see ``pipelines.face``), not the network's.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...core.registry import register_architecture
from .stylegan2_clean import ResBlock, StyleGAN2GeneratorCSFT


@register_architecture("gfpgan")
class GFPGANv1Clean(nn.Module):
    """U-Net encoder + StyleGAN2 decoder, joined by SFT conditions.

    Args:
        out_size: face resolution. 512 for GFPGANv1.4.
        num_style_feat: style vector width.
        channel_multiplier: decoder width multiplier (2 for v1.3/v1.4).
        num_mlp: style-mapping depth.
        input_is_latent: True => the encoder emits W+ directly, skipping style_mlp.
        different_w: True => predict a distinct style vector per generator layer (W+).
        narrow: global width scale.
        sft_half: condition only half the decoder channels.
    """

    def __init__(
        self,
        out_size: int = 512,
        num_style_feat: int = 512,
        channel_multiplier: int = 2,
        num_mlp: int = 8,
        input_is_latent: bool = True,
        different_w: bool = True,
        narrow: float = 1.0,
        sft_half: bool = True,
    ) -> None:
        super().__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        # The encoder runs at half the decoder's width — it only has to describe
        # the input, not synthesise it.
        unet_narrow = narrow * 0.5
        channels = {
            "4": int(512 * unet_narrow),
            "8": int(512 * unet_narrow),
            "16": int(512 * unet_narrow),
            "32": int(512 * unet_narrow),
            "64": int(256 * channel_multiplier * unet_narrow),
            "128": int(128 * channel_multiplier * unet_narrow),
            "256": int(64 * channel_multiplier * unet_narrow),
            "512": int(32 * channel_multiplier * unet_narrow),
            "1024": int(16 * channel_multiplier * unet_narrow),
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2 ** int(math.log(out_size, 2))

        # -- encoder (downsampling half of the U-Net) --------------------------
        self.conv_body_first = nn.Conv2d(3, channels[f"{first_out_size}"], 1)

        in_channels = channels[f"{first_out_size}"]
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f"{2 ** (i - 1)}"]
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode="down"))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, channels["4"], 3, 1, 1)

        # -- decoder half of the U-Net (produces the SFT conditions) -----------
        in_channels = channels["4"]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode="up"))
            in_channels = out_channels

        # Intermediate RGB outputs — a training-time supervision signal. Kept so
        # the checkpoint loads cleanly; unused at inference.
        self.toRGB = nn.ModuleList(
            [nn.Conv2d(channels[f"{2**i}"], 3, 1) for i in range(3, self.log_size + 1)]
        )

        linear_out_channel = (
            (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat if different_w else num_style_feat
        )
        self.final_linear = nn.Linear(channels["4"] * 4 * 4, linear_out_channel)

        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half,
        )

        # Per-resolution (scale, shift) heads. These are the SFT conditions.
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            sft_out_channels = out_channels if sft_half else out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1),
                )
            )

    def forward(self, x: torch.Tensor, randomize_noise: bool = False) -> torch.Tensor:
        """Restore an aligned face.

        Args:
            x: NCHW, 512x512, **[-1, 1]**. The backend hands over [0,1]; the face
                pipeline rescales, because that is where the crop is prepared.
            randomize_noise: resample StyleGAN noise. False keeps a batch of tiles
                (or repeated runs on one photo) deterministic.
        Returns:
            NCHW, 512x512, [-1, 1].
        """
        conditions: list[torch.Tensor] = []
        unet_skips: list[torch.Tensor] = []

        feat = F.leaky_relu(self.conv_body_first(x), negative_slope=0.2)
        for down in self.conv_body_down:
            feat = down(feat)
            unet_skips.insert(0, feat)
        feat = F.leaky_relu(self.final_conv(feat), negative_slope=0.2)

        style_code = self.final_linear(feat.reshape(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        for i, up in enumerate(self.conv_body_up):
            feat = feat + unet_skips[i]
            feat = up(feat)
            # Order matters: the decoder reads conditions pairwise as (scale, shift).
            conditions.append(self.condition_scale[i](feat).clone())
            conditions.append(self.condition_shift[i](feat).clone())

        return self.stylegan_decoder(
            [style_code],
            conditions,
            input_is_latent=self.input_is_latent,
            randomize_noise=randomize_noise,
        )

"""StyleGAN2 generator, "clean" formulation.

The reference StyleGAN2 relies on two custom CUDA kernels (`fused_leaky_relu`,
`upfirdn2d`). Those need a compiler toolchain at install time and only run on
CUDA — which would break the CPU, DirectML and OpenVINO backends this project is
required to support, and make Windows installs fragile.

This variant replaces them with stock ops:

* ``fused_leaky_relu``  -> ``LeakyReLU(0.2)`` followed by a ``sqrt(2)`` rescale
* ``upfirdn2d`` resample -> bilinear ``F.interpolate``

That is exactly the substitution the official `GFPGANv1Clean` weights were
trained/converted for, so ``GFPGANv1.4`` loads into it as-is.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class NormStyleCode(nn.Module):
    """Pixel-norm on the style vector."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class ModulatedConv2d(nn.Module):
    """Conv whose kernel is scaled per-sample by a style vector, then demodulated.

    The per-sample weight is realised as a grouped conv (groups=batch) over a
    batch-flattened input — that is what lets one conv apply B different kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: str | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        nn.init.kaiming_normal_(self.modulation.weight, a=0, mode="fan_in", nonlinearity="linear")
        nn.init.ones_(self.modulation.bias)  # start as a no-op modulation

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
            / math.sqrt(in_channels * kernel_size**2)
        )
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.sample_mode == "upsample":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        elif self.sample_mode == "downsample":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        return out.view(b, self.out_channels, *out.shape[2:4])


class StyleConv(nn.Module):
    """Modulated conv + noise injection + bias + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        sample_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
        )
        self.weight = nn.Parameter(torch.zeros(1))  # noise strength, learned
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        # sqrt(2) restores the variance the original fused_leaky_relu applied.
        out = self.modulated_conv(x, style) * math.sqrt(2)

        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()

        out = out + self.weight * noise
        out = out + self.bias
        return self.activate(out)


class ToRGB(nn.Module):
    """Project features to RGB and add the upsampled skip from the level below."""

    def __init__(self, in_channels: int, num_style_feat: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.modulated_conv(x, style) + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    """The learned 4x4 seed every StyleGAN2 image grows from."""

    def __init__(self, num_channel: int, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch: int) -> torch.Tensor:
        return self.weight.repeat(batch, 1, 1, 1)


class StyleGAN2GeneratorCSFT(nn.Module):
    """StyleGAN2 generator with Spatial Feature Transform conditioning.

    GFPGAN's contribution: the degraded face is encoded by a U-Net, and the
    U-Net's per-resolution features become (scale, shift) pairs that modulate the
    generator's activations. The generator supplies the facial prior, the SFT
    conditions keep the output faithful to *this* identity rather than a plausible
    stranger.

    With ``sft_half=True`` only half the channels are conditioned; the rest run
    free, which is what preserves generative detail in skin and hair.
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        narrow: float = 1.0,
        sft_half: bool = False,
    ) -> None:
        super().__init__()
        self.num_style_feat = num_style_feat
        self.sft_half = sft_half

        layers: list[nn.Module] = [NormStyleCode()]
        for _ in range(num_mlp):
            layers.append(nn.Linear(num_style_feat, num_style_feat, bias=True))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.style_mlp = nn.Sequential(*layers)

        channels = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels["4"], size=4)
        self.style_conv1 = StyleConv(
            channels["4"], channels["4"], 3, num_style_feat, demodulate=True, sample_mode=None
        )
        self.to_rgb1 = ToRGB(channels["4"], num_style_feat, upsample=False)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            self.noises.register_buffer(
                f"noise{layer_idx}", torch.randn(1, 1, resolution, resolution)
            )

        in_channels = channels["4"]
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    3,
                    num_style_feat,
                    demodulate=True,
                    sample_mode="upsample",
                )
            )
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    3,
                    num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                )
            )
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels

    def forward(
        self,
        styles: list[torch.Tensor],
        conditions: list[torch.Tensor],
        input_is_latent: bool = False,
        noise: list[torch.Tensor | None] | None = None,
        randomize_noise: bool = True,
    ) -> torch.Tensor:
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]

        if noise is None:
            noise = (
                [None] * self.num_layers
                if randomize_noise
                else [getattr(self.noises, f"noise{i}") for i in range(self.num_layers)]
            )

        latent = styles[0]
        if latent.ndim < 3:
            latent = latent.unsqueeze(1).repeat(1, self.num_latent, 1)

        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2],
            self.style_convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
            strict=False,
        ):
            out = conv1(out, latent[:, i], noise=noise1)

            # The encoder only produces conditions for the lower resolutions; the
            # top levels are left to the generator prior.
            if i < len(conditions):
                if self.sft_half:
                    out_free, out_sft = torch.split(out, out.size(1) // 2, dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_free, out_sft], dim=1)
                else:
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        return skip


class ResBlock(nn.Module):
    """Residual block that also resamples. Used by the GFPGAN U-Net."""

    def __init__(self, in_channels: int, out_channels: int, mode: str = "down") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.scale_factor = 0.5 if mode == "down" else 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)

        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return out + self.skip(x)

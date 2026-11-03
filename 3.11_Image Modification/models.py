# U-Net: network that gets an image 256x256 and returns an image of the same size.
# Used for denoising (task 2.2); can be used for super-resolution too
# (if the low-resolution image is first enlarged to 256x256).
import torch
import torch.nn as nn
import math


def double_conv(in_ch, out_ch):
    # two convolutions 3x3 with ReLU - basic building block of U-Net
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        # encoder: image gets smaller (256 -> 128 -> 64 -> 32), number of channels grows
        self.enc1 = double_conv(3, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.bottom = double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        # decoder: image gets bigger again, we also use outputs of the encoder (skip connections)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = double_conv(base * 2, base)

        self.out = nn.Conv2d(base, 3, 1)   # back to 3 channels (RGB)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottom(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # the network learns only the correction: result = input + correction
        out = x + self.out(d1)
        return torch.clamp(out, 0.0, 1.0)


# EDSR
class EDSRBlock(nn.Module):
    def __init__(self, channels=256, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, upscale_factor=4, num_channels=256, num_res_blocks=32, res_scale=0.1):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.head = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)

        # Residual blocks + closing conv
        self.body_blocks = nn.Sequential(
            *[EDSRBlock(channels=num_channels, res_scale=res_scale) for _ in range(num_res_blocks)]
        )
        self.body_end = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        num_upsamples = int(math.log2(upscale_factor))
        upsample_layers = []
        for _ in range(num_upsamples):
            upsample_layers.extend([
                nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            ])
        self.upsample = nn.Sequential(*upsample_layers)

        # Final reconstruction to RGB
        self.tail = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x_feat = self.head(x)
        res = self.body_end(self.body_blocks(x_feat))
        res = res + x_feat 
        out = self.tail(self.upsample(res))
        return torch.clamp(out, 0.0, 1.0)

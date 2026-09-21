import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.encoder1 = ConvBlock(
            in_channels,
            c1,
        )
        self.encoder2 = ConvBlock(
            c1,
            c2,
        )
        self.encoder3 = ConvBlock(
            c2,
            c3,
        )
        self.encoder4 = ConvBlock(
            c3,
            c4,
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            c4,
            c5,
        )

        self.up4 = nn.ConvTranspose2d(
            c5,
            c4,
            kernel_size=2,
            stride=2,
        )
        self.decoder4 = ConvBlock(
            c4 + c4,
            c4,
        )

        self.up3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=2,
            stride=2,
        )
        self.decoder3 = ConvBlock(
            c3 + c3,
            c3,
        )

        self.up2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = ConvBlock(
            c2 + c2,
            c2,
        )

        self.up1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = ConvBlock(
            c1 + c1,
            c1,
        )

        self.head = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool(e1)
        )

        e3 = self.encoder3(
            self.pool(e2)
        )

        e4 = self.encoder4(
            self.pool(e3)
        )

        bottleneck = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(bottleneck)
        d4 = torch.cat(
            [d4, e4],
            dim=1,
        )
        d4 = self.decoder4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )
        d1 = self.decoder1(d1)

        return self.head(d1)

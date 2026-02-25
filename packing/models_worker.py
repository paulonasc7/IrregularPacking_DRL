import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UNet14(nn.Module):
    """7 double-conv blocks => 14 conv layers."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.inc = _DoubleConv(in_ch, 32)
        self.down1 = _DoubleConv(32, 64)
        self.down2 = _DoubleConv(64, 128)
        self.down3 = _DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.up1_t = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = _DoubleConv(256, 128)
        self.up2_t = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = _DoubleConv(128, 64)
        self.up3_t = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up3 = _DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))

        u1 = self.up1_t(x4)
        if u1.shape[-2:] != x3.shape[-2:]:
            u1 = nn.functional.interpolate(u1, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, x3], dim=1))

        u2 = self.up2_t(u1)
        if u2.shape[-2:] != x2.shape[-2:]:
            u2 = nn.functional.interpolate(u2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, x2], dim=1))

        u3 = self.up3_t(u2)
        if u3.shape[-2:] != x1.shape[-2:]:
            u3 = nn.functional.interpolate(u3, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(torch.cat([u3, x1], dim=1))
        return self.out(u3)


class WorkerQNet(nn.Module):
    """Q-network for placement scoring.

    Two modes:
    - Legacy MLP: pass `input_dim` only, forward(x)
    - CNN+Scalar encoder: pass `map_channels` and `scalar_dim`, forward(map_x, scalar_x)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        map_channels: int | None = None,
        scalar_dim: int | None = None,
    ):
        super().__init__()
        self.use_map_encoder = map_channels is not None and scalar_dim is not None

        if self.use_map_encoder:
            # Paper-style worker backbone: U-Net score-map predictor.
            self.map_encoder = _UNet14(int(map_channels))
            self.scalar_encoder = nn.Sequential(
                nn.Linear(int(scalar_dim), hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            if input_dim is None:
                raise ValueError("input_dim must be provided when not using map encoder.")
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    @staticmethod
    def _gather_xy(score_map: torch.Tensor, scalar_x: torch.Tensor) -> torch.Tensor:
        # scalar_x stores normalized x,y in first two dimensions.
        b, _c, h, w = score_map.shape
        px = torch.clamp(scalar_x[:, 0], 0.0, 1.0)
        py = torch.clamp(scalar_x[:, 1], 0.0, 1.0)
        ix = torch.clamp((px * (h - 1)).round().long(), 0, h - 1)
        iy = torch.clamp((py * (w - 1)).round().long(), 0, w - 1)
        batch_idx = torch.arange(b, device=score_map.device)
        return score_map[batch_idx, 0, ix, iy]

    def score_map(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_map_encoder:
            raise ValueError("score_map() is only available when use_map_encoder=True")
        return self.map_encoder(x)

    def forward(self, x: torch.Tensor, scalar_x: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_map_encoder:
            if scalar_x is None:
                raise ValueError("scalar_x is required when use_map_encoder=True")
            score_map = self.map_encoder(x)
            map_score = self._gather_xy(score_map, scalar_x)
            scalar_score = self.scalar_encoder(scalar_x).squeeze(-1)
            return map_score + scalar_score
        return self.net(x).squeeze(-1)

import torch
import torch.nn as nn
try:
    import torchvision
except Exception:  # pragma: no cover - optional dependency guard
    torchvision = None


class ManagerQNet(nn.Module):
    """Q-network for object-selection scoring.

    Two modes:
    - Legacy MLP: pass `input_dim` only, forward(x)
    - CNN+Scalar encoder: pass `map_channels` and `scalar_dim`, forward(map_x, scalar_x)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 128,
        map_channels: int | None = None,
        scalar_dim: int | None = None,
    ):
        super().__init__()
        self.use_map_encoder = map_channels is not None and scalar_dim is not None

        if self.use_map_encoder:
            self.map_encoder = self._make_resnet18_encoder(int(map_channels))
            self.scalar_encoder = nn.Sequential(
                nn.Linear(int(scalar_dim), 64),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                # Paper-style manager head: ResNet18 features + 3-layer FC scorer.
                nn.Linear(512 + 64, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
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
    def _make_resnet18_encoder(in_channels: int) -> nn.Module:
        if torchvision is None:
            return nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        try:
            model = torchvision.models.resnet18(weights=None)
        except TypeError:
            model = torchvision.models.resnet18(pretrained=False)

        if in_channels != 3:
            old = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

        model.fc = nn.Identity()
        return model

    def forward(self, x: torch.Tensor, scalar_x: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_map_encoder:
            if scalar_x is None:
                raise ValueError("scalar_x is required when use_map_encoder=True")
            map_feat = self.map_encoder(x)
            scalar_feat = self.scalar_encoder(scalar_x)
            feat = torch.cat([map_feat, scalar_feat], dim=1)
            return self.head(feat).squeeze(-1)
        return self.net(x).squeeze(-1)

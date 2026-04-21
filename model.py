import torch
import torch.nn as nn


class AtriumUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)

        self.dec = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        pooled = self.pool(enc1)
        enc2 = self.enc2(pooled)
        upsampled = self.up(enc2)
        merged = torch.cat([upsampled, enc1], dim=1)
        return self.dec(merged)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.forward_from_features(features)


UNet = AtriumUNet

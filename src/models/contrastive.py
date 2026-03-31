import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, proj_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.projector = ProjectionHead()

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z

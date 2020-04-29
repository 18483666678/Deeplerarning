import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.sequential(x) + x


if __name__ == '__main__':
    resBlock = ResBlock()
    x = torch.randn(1,16,55,55)
    y = resBlock(x)
    print(y.shape)
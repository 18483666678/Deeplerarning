import torch
import torch.nn as nn

config = [
    [-1, 32, 1, 2],
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1]
]


class Block(nn.Module):

    def __init__(self, p_c, t, c, i, s):
        super().__init__()
        self._s = s if i == 0 else 1
        _p_c = p_c * t
        self.layer = nn.Sequential(
            nn.Conv2d(p_c, _p_c, 1, self._s, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, _p_c, 3, 1, padding=1, groups=_p_c, bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c, c, 1, 1, bias=False),
            nn.BatchNorm2d(c)
        )

    def forward(self, x):
        print(self.layer(x).shape,x.shape)
        if self._s > 1:
            return self.layer(x) + x

        else:
            return self.layer(x) + x


class MobilenetV2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        self.blocks = []
        p_c = config[0][1]
        for t, c, n, s in config[1:]:
            for i in range(n):
                self.blocks.append(Block(p_c, t, c, i, s))
        self.hidden_layer = nn.Sequential(*self.blocks)

        self.output_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(7, 1),
            nn.Conv2d(1280, 10, 1, 1, bias=False)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.hidden_layer(h)
        h = self.output_layer(h)
        return h


if __name__ == '__main__':
    net = MobilenetV2(config)
    y = net(torch.randn(1, 3, 224, 224))
    print(y.shape)

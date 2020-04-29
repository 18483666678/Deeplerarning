from torch import nn
import torch

class NetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3,16,3,1,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2,1),
        )
        self.output = nn.Sequential(
            nn.Linear(64*8*8,10),
        )

    def forward(self,x):
        h = self.seq(x)
        h = h.reshape(-1,64*8*8)
        h = self.output(h)
        return h

# 初始化卷积和全连接
def weight_init(m):
    if (isinstance(m,nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif(isinstance(m,nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class NetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Conv2d(3,16,3,1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,3,1,padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64*8*8,10)
        )

        # 权重初始化
        self.apply(weight_init)

    def forward(self,x):
        h = self.seq(x)
        h = h.reshape(-1,64*8*8)
        h = self.output(h)
        return h


if __name__ == '__main__':
    net = NetV2()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())